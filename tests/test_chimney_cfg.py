"""CPU invariants for braced corridor climbing (no GPU needed)."""
from __future__ import annotations

import torch

import mjlab_microduck.tasks  # noqa: F401  (registry)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.tasks import chimney_mdp as ch

TASK = "Mjlab-Chimney-MicroDuck"


def test_corridor_stays_inside_the_measured_band():
    """The bracing measurement: 12 cm wedges at friction 0.3, 14 cm needs 0.8,
    16 cm never held.  The curriculum must not leave that band."""
    assert 0.11 <= cs.WIDTH_MIN_M < cs.WIDTH_MAX_M <= 0.15
    cfg = load_env_cfg(TASK)
    lo, hi = cfg.events["set_chimney_state"].params["width_range"]
    assert cs.WIDTH_MIN_M <= lo <= hi <= cs.WIDTH_MAX_M
    for spec_fn in (lambda: cs._wall_spec(cs.WALL_LEFT, "1 1 1 1"),):
        model = spec_fn().compile()
        assert model.nmocap == 1
        # the wall marks are OFF by default (user asked for them removed once
        # clips carried a live height readout instead)
        marks = int(cs.WALL_HEIGHT_M / cs.MARKER_SPACING_M) if cs.WALL_MARKS else 0
        assert model.ngeom == 1 + marks
        assert model.geom_contype[0] != 0
        # and if they are switched back on they must stay invisible to physics
        assert all(int(c) == 0 for c in model.geom_contype[1:])
        assert all(int(c) == 0 for c in model.geom_conaffinity[1:])
    # a wall must out-reach the tallest spawn, or the climb runs off the top
    assert cs.WALL_HEIGHT_M > 2.0


def test_reward_signs_and_terms():
    cfg = load_env_cfg(TASK)
    r = cfg.rewards
    assert not any(n.startswith("long_jump") for n in r)
    assert r["ch_progress"].weight > 0 and r["ch_brace"].weight > 0
    # self-negating penalties carry POSITIVE weights (mdp sign convention)
    assert r["ch_slip"].weight > 0 and r["ch_fall_cost"].weight > 0
    # height is paid as PROGRESS, so holding still cannot be farmed
    assert r["ch_progress"].func is ch.climb_progress
    # The brace is a foothold, not the goal - and the comparison has to be made
    # over an EPISODE, not per step, which the first version of this test got
    # wrong: 0.5/step x 300 steps paid 150 for hanging still while climbing 5 cm
    # paid 40, so holding won by 4x and run w3 duly learned to hang.
    steps = int(cfg.episode_length_s / 0.02)
    hold_pays = r["ch_brace"].weight * steps
    climb_5cm_pays = r["ch_progress"].weight * 0.05
    assert hold_pays < climb_5cm_pays / 2
    # ... and hanging still must actively cost something
    assert r["ch_stall"].weight > 0
    stall_cost = r["ch_stall"].weight * steps * 0.9
    assert stall_cost > hold_pays
    assert cfg.terminations["fell"].func is ch.climb_fell
    assert {cs.WALL_LEFT, cs.WALL_RIGHT} <= set(cfg.scene.entities)


def test_progress_pays_only_for_new_height():
    """Potential-based: the same height pays once, and sliding back earns
    nothing on the way up again."""
    from types import SimpleNamespace

    class _Env:
        num_envs = 2
        device = "cpu"
        scene = {"robot": SimpleNamespace()}

    env = _Env()
    st = ch._st(env)
    st["held_z"][:] = torch.tensor([0.50, 0.50])   # braced height, not the peak
    st["paid"][:] = torch.tensor([0.40, 0.50])
    orig = ch._update
    ch._update = lambda e, a: st
    try:
        first = ch.climb_progress(env, asset_cfg=SimpleNamespace(name="robot"))
        again = ch.climb_progress(env, asset_cfg=SimpleNamespace(name="robot"))
    finally:
        ch._update = orig
    assert torch.allclose(first, torch.tensor([0.10, 0.00]))
    assert torch.allclose(again, torch.zeros(2))


def test_spawn_starts_braced_up_the_corridor():
    """Reverse curriculum: without mid-corridor spawns only the bottom of the
    climb would ever be practised."""
    cfg = load_env_cfg(TASK)
    p = cfg.events["set_chimney_state"].params
    lo, hi = p["height_range"]
    assert lo >= 0.2 and hi >= 0.4
    ov = p["brace_overrides"]
    # hips rotated across the gap (the measured brace), mirrored left/right
    assert abs(ov[2]) > 1.5 and abs(ov[11]) > 1.5 and ov[2] * ov[11] < 0
    play = load_env_cfg(TASK, play=True)
    assert play.events["set_chimney_state"].params["height_range"][1] <= 0.5


def test_head_on_a_wall_is_priced_not_fatal():
    """In a 12 cm corridor the head spans the gap by construction: measured at
    spawn, 98 % correctly wedged and 89 % with the head touching.  Terminating
    on it ended every episode in one step (run w1, 1.09 steps, 3254 falls per
    window)."""
    from types import SimpleNamespace

    class _Env:
        num_envs = 3
        device = "cpu"
        common_step_counter = 0
        scene = {"robot": SimpleNamespace()}

    env = _Env()
    env._ch_contacts = {
        "step": 0,
        "on_floor": torch.tensor([False, True, False]),
        "head_touch": torch.tensor([True, False, False]),
        "feet_left": torch.tensor([False, False, False]),
        "feet_right": torch.tensor([True, True, True]),
        "body_left": torch.tensor([True, True, True]),
        "body_right": torch.tensor([False, False, False]),
    }
    st = ch._st(env)
    st["max_z"][:] = 0.5
    orig = ch._update
    ch._update = lambda e, a: st
    try:
        pen = ch.climb_head_penalty(env, asset_cfg=SimpleNamespace(name="robot"))
    finally:
        ch._update = orig
    assert pen[0] < 0 and pen[1] == 0          # priced ...
    cfg = load_env_cfg(TASK)
    assert cfg.rewards["ch_head"].weight > 0   # ... with the self-negating sign convention
    assert cfg.rewards["ch_head"].weight < cfg.rewards["ch_fall_cost"].weight


def test_the_brace_is_the_neutral_action():
    """The action offset is the robot's DEFAULT joint pose.  With the HOME
    default (legs down) a zero action drives the legs out of the wedge in a few
    control steps - run w2 fell out of every brace in ~0.2 s and gained no
    height.  The brace has to BE the default, so holding still holds the wedge.
    """
    import re

    import mjlab_microduck.tasks.microduck_chimney_env_cfg as m
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    cfg = load_env_cfg(TASK)
    jp = cfg.scene.entities["robot"].init_state.joint_pos

    def value(name):
        hit = 0.0
        for pat, v in jp.items():
            if re.match(pat, name):
                hit = v
        return hit

    # hips rotated across the gap, mirrored, and NOT the standing default
    assert abs(value("left_hip_pitch")) > 1.5
    assert value("left_hip_pitch") * value("right_hip_pitch") < 0
    home_hip = [v for pat, v in HOME_FRAME.joint_pos.items() if re.match(pat, "left_hip_pitch")][-1]
    assert abs(value("left_hip_pitch") - home_hip) > 1.0
    # and the spawn override agrees with the default, or the two fight
    assert abs(value("left_hip_pitch") - m.BRACE_OVERRIDES[2]) < 0.01


def test_only_braced_height_is_paid():
    """Paying the instantaneous peak was a jackpot: w4 lunged, banked 14 cm of
    transient height and fell, every episode (braced 17 % of the time, surviving
    1.3 s).  Height must only count while the robot is wedged."""
    from types import SimpleNamespace

    class _Env:
        num_envs = 2
        device = "cpu"
        common_step_counter = 3
        step_dt = 0.02
        scene = {"robot": SimpleNamespace()}

    env = _Env()
    st = ch._st(env)
    st["held_z"][:] = torch.tensor([0.60, 0.60])
    st["paid"][:] = torch.tensor([0.50, 0.60])
    orig = ch._update
    ch._update = lambda e, a: st
    try:
        paid = ch.climb_progress(env, asset_cfg=SimpleNamespace(name="robot"))
    finally:
        ch._update = orig
    assert torch.allclose(paid, torch.tensor([0.10, 0.0]))
    # the source of truth for progress must be the BRACED height, not the peak
    import inspect

    src = inspect.getsource(ch.climb_progress)
    assert "held_z" in src and "max_z" not in src


def test_wall_friction_is_randomisable_per_environment():
    """The wedge is geometric - 12 cm holds at friction 0.3 while 14 cm needs
    0.8 - so how slippery the wall may be is THE sim2real question here, and
    every run up to w8 only ever saw the 0.9 the walls are built with."""
    cfg = load_env_cfg(TASK)
    for wall in (cs.WALL_LEFT, cs.WALL_RIGHT):
        term = cfg.events[f"{wall}_friction"]
        assert term.mode == "startup"
        assert term.params["asset_cfg"].name == wall
        assert term.params["asset_cfg"].geom_names == (f"{wall}_collision",)
        lo, hi = term.params["ranges"]
        assert 0.0 < lo <= hi <= 1.5
    # and the default must leave the walls where they were built
    import mjlab_microduck.tasks.microduck_chimney_env_cfg as m
    import os as _os
    if not _os.getenv("MICRODUCK_CH_WALL_FRIC"):
        assert m.WALL_FRICTION == (0.9, 0.9)


def test_a_ground_start_is_not_killed_by_touching_the_floor():
    """Every run through w9 spawned the robot already wedged 25 cm to 1.3 m up,
    so entering the corridor from the floor was never trained.  A ground start
    begins ON the floor, so floor contact cannot end the episode until it has
    been wedged at least once — otherwise every floor spawn dies on its first
    step, the way the flip's fatal-hop rule killed itself on the spawn bounce."""
    import inspect

    src = inspect.getsource(ch._update)
    line = [l for l in src.splitlines() if l.strip().startswith("floor_fatal =")]
    assert len(line) == 1, line
    assert "ground_start" in line[0] and "ever_braced" in line[0], line[0]
    # the knob exists and defaults to off, so every existing run is unchanged
    import os as _os

    import mjlab_microduck.tasks.microduck_chimney_env_cfg as m
    if not _os.getenv("MICRODUCK_CH_GROUND_PROB"):
        assert m.GROUND_PROB == 0.0
    cfg = load_env_cfg(TASK)
    assert "ground_prob" in cfg.events["set_chimney_state"].params


def test_the_climb_height_label_reads_cm_then_m():
    """Corridor-climb clips carry the live height top left: whole centimetres
    below a metre, metres to one decimal above it."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "add_climb_height_overlay.py"
    src = path.read_text()
    ns: dict = {}
    # exec just the formatter, so the test needs no imageio/PIL
    start = src.index("def label_for(")
    end = src.index("def _font(")
    exec(src[start:end], ns)
    label_for = ns["label_for"]
    assert label_for(0.27) == "27cm"
    assert label_for(0.274) == "27cm"
    assert label_for(0.994) == "99cm"
    assert label_for(0.999) == "1.0m"      # switches on the ROUNDED value
    assert label_for(1.0) == "1.0m"
    assert label_for(1.24) == "1.2m"
    assert label_for(8.18) == "8.2m"


def test_an_upright_handover_start_is_available():
    """A separate approach policy would walk in and hand over the duck STANDING
    mid-corridor, legs down.  The climb's own floor starts are already braced,
    so that handover state is outside its distribution unless trained here.  A
    policy switch only works where the sender's terminal states lie inside the
    receiver's initial ones."""
    import inspect
    import os as _os

    import mjlab_microduck.tasks.microduck_chimney_env_cfg as m

    src = inspect.getsource(ch.reset_chimney_state)
    assert "upright_prob" in src and "is_upright" in src
    # upright starts stand at standing height, midway between the walls, no pitch
    assert "upright_height" in src
    assert "torch.where(is_upright, torch.zeros" in src
    # and the standing joint angles must be written EXPLICITLY: the robot's
    # default pose IS the brace, so leaving the joints alone spawns a braced
    # duck that merely stands at standing height (caught on video 2026-09-19)
    assert "upright_overrides" in src
    assert "is_upright, env.sim.data.qpos" not in src
    import mjlab_microduck.tasks.microduck_chimney_env_cfg as mm
    assert set(mm.UPRIGHT_OVERRIDES) == set(mm.BRACE_OVERRIDES)
    # the hips and ankles are what make a brace a brace; the knees are near
    # zero in both poses, so only these need to differ
    for j in (2, 11, 4, 13):
        assert abs(mm.UPRIGHT_OVERRIDES[j] - mm.BRACE_OVERRIDES[j]) > 0.3, j
    # hips swing from across-the-gap to under the body
    assert abs(mm.BRACE_OVERRIDES[2]) > 1.5 and abs(mm.UPRIGHT_OVERRIDES[2]) < 0.6
    cfg = load_env_cfg(TASK)
    assert "upright_prob" in cfg.events["set_chimney_state"].params
    if not _os.getenv("MICRODUCK_CH_UPRIGHT_PROB"):
        assert m.UPRIGHT_PROB == 0.0


def test_parked_walls_stay_below_the_floor_at_any_height():
    """A fixed parking depth breaks for tall walls: a 25 m wall parked at -10 m
    has its top 2.5 m ABOVE the floor and collides with every spawn, which
    surfaces as an unrelated-looking nconmax overflow."""
    import importlib
    import os

    from mjlab_microduck.robot import corridor_stage as cs

    for h in ("2.4", "10.0", "25.0"):
        os.environ["MICRODUCK_CH_WALL_H"] = h
        importlib.reload(cs)
        top = cs.PARKED_POS[2] + 0.5 * cs.WALL_HEIGHT_M
        assert top < -1.0, f"parked wall of height {h} reaches z={top:.2f}"
    os.environ.pop("MICRODUCK_CH_WALL_H", None)
    importlib.reload(cs)


def test_handover_bank_is_applied_after_the_normal_state():
    """The bank replaces only the POSE; every other per-env field (max_z,
    held_z, paid, ground_start, width) must already be set, or the climb's
    reward and termination read stale values from the previous episode."""
    import inspect

    from mjlab_microduck.tasks import chimney_mdp as ch

    src = inspect.getsource(ch.reset_chimney_state)
    bank_at = src.index("handover_bank and handover_prob")
    width_at = src.index('st["width"][env_ids] = width')
    assert bank_at > width_at, "the bank override must come after the normal state"

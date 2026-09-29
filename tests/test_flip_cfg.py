"""CPU invariants for the platform flip task."""
import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot import flip_stage
from mjlab_microduck.tasks.bounded_actions import BoundedJointPositionActionCfg

TASK = "Mjlab-Flip-MicroDuck"


def test_flip_cfg_builds():
    cfg = load_env_cfg(TASK, play=False)
    assert isinstance(cfg.actions["joint_pos"], BoundedJointPositionActionCfg)
    assert flip_stage.PLATFORM_ENTITY in cfg.scene.entities and flip_stage.MAT_ENTITY in cfg.scene.entities
    names = {s.name for s in cfg.scene.sensors}
    assert {"feet_mat_contact", "body_mat_contact", "head_mat_contact", "body_floor_contact", "self_collision"} <= names
    actor = cfg.observations["actor"].terms
    assert actor["body_command"].params["dim"] == 6 and actor["head_command"].params["dim"] == 4


def test_flip_reward_signs():
    r = load_env_cfg(TASK, play=False).rewards
    for name in ("flip_stand_tax", "flip_fall_cost", "flip_body_contact", "flip_dawdle", "flip_lateral", "gentle_landing"):
        assert r[name].weight >= 0.0, name
    for name in ("action_rate_l2", "self_collisions", "body_ang_vel"):
        assert r[name].weight <= 0.0, name
    assert r["flip_progress"].weight > 0 and r["flip_stand"].weight > 0
    assert "upright" not in r and not any(n.startswith("long_jump") for n in r)


def test_flip_play_has_no_midflip_spawns():
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.events["set_flip_state"].params["midflip_prob"] == 0.0
    assert cfg.events["set_flip_state"].params["direction"] in ("back", "front")


def test_a_preparatory_hop_does_not_steal_the_takeoff():
    """User 2026-09-18, watching the clip: "it first does a small jump and then
    does the actual jump".  The first instant airborne used to latch jump_ok,
    the takeoff vz and the tilt, so every takeoff metric described the HOP and
    the real jump was never measured.  Landing back on the platform must return
    the episode to GROUND, and the hop must be priced."""
    from mjlab.tasks.registry import load_env_cfg

    from mjlab_microduck.tasks import flip_mdp as fl

    cfg = load_env_cfg("Mjlab-Flip-MicroDuck")
    assert cfg.rewards["flip_hop"].func is fl.flip_hop_penalty
    # self-negating penalties carry POSITIVE weights (mdp sign convention)
    assert cfg.rewards["flip_hop"].weight > 0
    # ... and the hop must cost less than a fall, or it will refuse to jump
    assert cfg.rewards["flip_hop"].weight < cfg.rewards["flip_fall_cost"].weight
    assert cfg.rewards["flip_hops"].func is fl.flip_hops


def test_play_mode_really_disables_mid_flip_spawns():
    """The reset event sets midflip_prob to 0 for play — and the spawn-mix
    curriculum's step-0 stage set it straight back to MIDFLIP_PROB, so every
    play rollout (renders, probes, batteries) was spawning about a quarter of
    its environments already in mid-air.  Measured 2026-09-18: 27.3 % of the
    "flips" in a play probe had takeoff vz exactly 0.000 and jump_ok preset
    true, which is the mid-flip spawn signature (reset_flip_state sets
    jump_ok = is_mid and leaves takeoff_vz at 0), not a flip at all."""
    play = load_env_cfg(TASK, play=True)
    assert play.events["set_flip_state"].params["midflip_prob"] == 0.0
    # ... and nothing may put it back
    assert "flip_spawn_mix" not in play.curriculum
    for name, term in play.curriculum.items():
        for stage in (term.params.get("param_stages") or []):
            assert "midflip_prob" not in stage.get("params", {}), (name, stage)
    # training keeps the mix
    train = load_env_cfg(TASK, play=False)
    assert "flip_spawn_mix" in train.curriculum
    assert train.events["set_flip_state"].params["midflip_prob"] > 0


def test_the_hop_discount_multiplies_the_payoff_not_a_flat_tax():
    """Measured 2026-09-18 on genuine platform starts: all 256 flips hop, 89 %
    hop exactly three times, and hopping more buys nothing (departure vz 0.174
    at two hops vs 0.180 at three).  A flat tax did not move the count in 1250
    iterations, because a tax needs a cheaper alternative already in the
    repertoire and there is none.  The discount makes each hop removed multiply
    what the flip earns — a gradient, not the cliff a hard zero-hop gate would
    be."""
    import torch as _t

    from mjlab_microduck.tasks import flip_mdp as fl

    st = {"hops": _t.tensor([0, 1, 2, 3])}
    old = fl.HOP_DECAY
    try:
        fl.HOP_DECAY = 0.6
        d = fl._hop_discount(st)
    finally:
        fl.HOP_DECAY = old
    assert float(d[0]) == 1.0
    assert abs(float(d[1]) - 0.6) < 1e-6
    assert abs(float(d[3]) - 0.216) < 1e-6
    # strictly decreasing, and never zero — no cliff
    assert all(float(d[i]) > float(d[i + 1]) > 0.0 for i in range(3))
    # and the default must leave existing runs untouched
    import os as _os
    if not _os.getenv("MICRODUCK_FLIP_HOP_DECAY"):
        assert old == 1.0


def test_the_hop_can_be_made_impossible_rather_than_expensive():
    """Two graded pressures failed to reduce the preparatory hop: the flat tax
    3 -> 8 over 1250 iterations left the count unmoved at ~1.9, and discounting
    the payoff to 0.6 ** hops (22 % of the reward at the usual three) let it
    RISE from 2.00 to 2.12 over 1000 more.  The policy has no hop-free flip to
    fall back on, so a gradient cannot pay for the crossing.  The knob makes the
    hop fatal instead; mid-flip spawns keep reward flowing meanwhile."""
    import os as _os

    import mjlab_microduck.tasks.microduck_flip_env_cfg as m
    from mjlab_microduck.tasks import flip_mdp as fl

    cfg = load_env_cfg(TASK)
    if not _os.getenv("MICRODUCK_FLIP_HOP_FATAL"):
        # default OFF, so every existing run is untouched
        assert m.HOP_FATAL is False
        assert "hopped" not in cfg.terminations
    assert callable(fl.flip_hopped)
    # and it must fire on the hop itself, not merely on being airborne
    import inspect
    src = inspect.getsource(fl.flip_hopped)
    assert 'st["hop_step"]' in src and 'st["hops"] > 0' in src


def test_a_spawn_bounce_is_not_a_preparatory_hop():
    """Run f24 with no guard: 308 of ~390 episodes per window ended "hopped" at
    a mean length of 9.5 steps (0.19 s).  The robot settling onto the platform
    at spawn registered as a departure and a return, so the episode died before
    it could do anything.  A real hop follows a real departure."""
    import inspect

    from mjlab_microduck.tasks import flip_mdp as fl

    src = inspect.getsource(fl.flip_hopped)
    # A time window cannot separate the two: the traced hop is only ~40 ms clear
    # of the platform, and it happens ~0.16 s after the spawn, inside any
    # sensible grace period.  What separates them is the sign of the departure:
    # a hop pushes off at +0.20 m/s and lifts the trunk ~1.6 cm, a settle drops.
    assert "HOP_MIN_VZ" in src and "takeoff_vz" in src
    assert "HOP_GRACE_STEPS" not in src and "HOP_MIN_AIR_STEPS" not in src
    assert 0.0 < fl.HOP_MIN_VZ < 0.15

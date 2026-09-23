"""Corridor exit: geometry and reward invariants from the user's sketch."""
from __future__ import annotations

import pytest

from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.robot import exit_stage as ex
from mjlab_microduck.tasks import exit_mdp as em

TASK = "Mjlab-Exit-MicroDuck"


def test_platform_starts_outside_the_corridor():
    """User: "the platform would thus not be inside the corridor itself"."""
    # now clear of the slot by PLATFORM_CLEAR_M, so nothing juts into the climb
    assert ex.platform_near_y() == pytest.approx(0.5 * cs.WALL_DEPTH_M + ex.PLATFORM_CLEAR_M)
    assert ex.platform_far_y() > ex.platform_near_y()


def test_walls_overhang_the_platform():
    """User: "the walls would also extend a bit towards us ... safety walls on
    the platform", which is what lets it brace while moving horizontally."""
    assert ex.wall_depth_with_overhang() > cs.WALL_DEPTH_M
    # the far face of the extended wall stays where the climbing slot was
    far_face = ex.wall_centre_y() - 0.5 * ex.wall_depth_with_overhang()
    assert far_face == pytest.approx(-0.5 * cs.WALL_DEPTH_M)
    # and the near face reaches out over the platform, stopping at the wall end
    # - the platform now continues past it into the open region
    near_face = ex.wall_centre_y() + 0.5 * ex.wall_depth_with_overhang()
    assert near_face == pytest.approx(ex.wall_end_y())
    assert near_face < ex.platform_far_y()


def test_goal_is_out_past_the_slot_mouth():
    """Landing must mean LEFT the corridor, not hovering in its mouth."""
    assert ex.goal_y() > ex.platform_near_y()
    assert ex.goal_y() < ex.platform_far_y()


def test_platform_top_is_where_the_slab_top_is():
    _, _, cz = ex.platform_mocap_pos()
    assert cz + 0.5 * ex.PLATFORM_THICK_M == pytest.approx(ex.PLATFORM_TOP_M)


def test_spawn_is_the_climbs_terminal_state_not_a_landing():
    """The exit starts WEDGED just above platform level - if it spawned already
    on the platform the task would be solved at t=0."""
    cfg = load_env_cfg(TASK, play=True)
    assert "set_exit_state" in cfg.events
    from mjlab_microduck.tasks.microduck_chimney_env_cfg import BRACE_OVERRIDES

    assert cfg.events["set_exit_state"].params["brace_overrides"] == BRACE_OVERRIDES


def test_default_pose_is_the_BRACE_so_the_handover_is_seamless():
    """Joint actions are offsets from the default pose.  A policy that begins
    wedged must share the neutral action of the policy handing it the wedge."""
    from mjlab_microduck.tasks.microduck_chimney_env_cfg import BRACE_FRAME

    cfg = load_env_cfg(TASK, play=True)
    assert cfg.scene.entities["robot"].init_state.joint_pos == BRACE_FRAME.joint_pos


def test_penalties_positive_and_success_bootstraps():
    cfg = load_env_cfg(TASK, play=True)
    for n in ("ex_dash", "ex_fall"):
        assert cfg.rewards[n].weight > 0, f"{n} is self-negating and needs a positive weight"
    assert cfg.terminations["landed"].time_out is True
    assert cfg.terminations["fell"].time_out is False


def test_progress_is_rate_limited():
    """The lesson from the approach: potential shaping stops camping, not
    diving."""
    assert em.MAX_PAID_SPEED <= 0.35


def test_falling_is_measured_from_the_PLATFORM_not_the_floor():
    """This task lives a metre up; a robot 40 cm back down the slot has failed
    even though it is nowhere near the ground."""
    import inspect

    src = inspect.getsource(em.exit_fell)
    assert "PLATFORM_TOP_M" in src


def test_obs_slots_are_all_present():
    cfg = load_env_cfg(TASK, play=True)
    for slot in ("command", "head_command", "body_command"):
        assert slot in cfg.observations["actor"].terms


def test_the_BUILT_walls_really_overhang_the_platform():
    """Not the helper - the actual compiled geom.  corridor_stage fixes its
    depth at import, so a cfg-level env var would leave 30 cm walls behind
    while every helper claimed 75 cm."""
    import mujoco

    spec = ex.make_overhang_corridor_entity_cfgs()[cs.WALL_LEFT].spec_fn()
    m = spec.compile()
    # the LOWER segment is the plain climbing slot...
    g_low = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{cs.WALL_LEFT}_collision")
    assert 2.0 * float(m.geom_size[g_low][1]) == pytest.approx(cs.WALL_DEPTH_M)
    # ...and only the UPPER one reaches out over the platform
    g_up = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{cs.WALL_LEFT}_upper")
    depth = 2.0 * float(m.geom_size[g_up][1])
    assert depth == pytest.approx(ex.wall_depth_with_overhang()), depth
    assert depth > cs.WALL_DEPTH_M


def test_dash_penalty_does_not_bill_a_FALL():
    """A free fall is fast and vertical; falling already costs.  Charging 3-D
    speed teaches "do not move" rather than "do not skate"."""
    import inspect

    src = inspect.getsource(em.exit_dash_penalty)
    assert "vel[:, :2]" in src, "the dash penalty must measure planar speed only"


def test_spawn_mix_includes_NEARLY_DONE_episodes():
    """x1/x2 spawned only at the handover and the landing bonus fired zero
    times in 1250 iterations - the goal state was never visited, so the critic
    had nothing to back up.  Some episodes must start nearly finished."""
    cfg = load_env_cfg(TASK, play=True)
    p = cfg.events["set_exit_state"].params
    assert p["nearly_done_prob"] > 0.0
    assert p["frontier_prob"] > 0.0
    assert p["nearly_done_prob"] + p["frontier_prob"] < 1.0, "keep real handover spawns too"


def test_nearly_done_spawn_writes_STANDING_joints_explicitly():
    """The default pose IS the brace, so a nearly-done spawn that leaves the
    joints alone puts a BRACED duck over the platform, not a standing one.  The
    same mistake produced a withdrawn handover result on the climb."""
    from mjlab_microduck.tasks.microduck_chimney_env_cfg import BRACE_OVERRIDES, UPRIGHT_OVERRIDES

    cfg = load_env_cfg(TASK, play=True)
    p = cfg.events["set_exit_state"].params
    assert p["stand_overrides"] == UPRIGHT_OVERRIDES
    assert p["stand_overrides"] != BRACE_OVERRIDES


def test_frontier_spawns_carry_FORWARD_VELOCITY():
    """A traverse delivers the robot moving; a frontier spawn from rest teaches
    a landing the robot never experiences.  x3 reached the platform 99.6 % of
    the time from the real handover and still landed 2 %, while scoring 67 %
    when the battery included from-rest frontier spawns."""
    import inspect

    src = inspect.getsource(em.reset_exit_state)
    assert "frontier_vy_range" in src
    assert "qvel[env_ids, 1]" in src, "the frontier spawn must be given +y velocity"


def test_the_platform_extends_PAST_the_wall_end():
    """User 2026-09-20: the walls stop part way along the platform so there is
    a region with only platform, and the duck has to walk out onto it."""
    assert ex.OPEN_M > 0.0
    assert ex.wall_end_y() < ex.platform_far_y()
    assert ex.wall_end_y() == pytest.approx(ex.platform_near_y() + ex.OVERHANG_M)


def test_the_goal_is_in_the_OPEN_part():
    """Standing while still braced between the overhanging walls is the
    half-way point, not the finish."""
    assert ex.goal_y() > ex.wall_end_y()
    assert ex.goal_y() < ex.platform_far_y()


def test_success_requires_being_clear_of_the_walls():
    import inspect

    src = inspect.getsource(em.exit_landed_done)
    assert "clear_of_the_walls" in src


def test_the_built_platform_covers_overhang_AND_open():
    import mujoco

    spec = ex.make_exit_stage_entity_cfgs()[ex.PLATFORM].spec_fn()
    m = spec.compile()
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{ex.PLATFORM}_collision")
    depth = 2.0 * float(m.geom_size[g][1])
    assert depth == pytest.approx(ex.OVERHANG_M + ex.OPEN_M)


def test_the_clear_line_is_not_at_the_wall_edge():
    """x8 cleared a line drawn AT the wall end 100 % of the time and stopped
    2 cm past it: crossing ends the episode, so the argmax is to cross by a
    centimetre.  The line has to sit out in the open region."""
    assert ex.clear_line_y() > ex.wall_end_y() + 0.10
    assert ex.clear_line_y() <= ex.platform_far_y()


def test_the_finish_out_pays_loitering_on_the_platform():
    """Measured: at land 3 / clear 8, sitting under the overhang for a 10 s
    episode paid 1200 while walking out and finishing paid 500 - because
    finishing ENDS the episode and forfeits the rest of the intermediate."""
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    episode_steps = cfgmod.EPISODE_LENGTH_S / 0.02
    hold_steps = 0.5 / 0.02
    loiter = cfgmod.LAND_W * episode_steps * 0.8          # land early, sit
    finish = cfgmod.LAND_W * episode_steps * 0.2 + cfgmod.CLEAR_W * hold_steps
    assert finish > loiter, f"loitering pays {loiter:.0f} against a finish of {finish:.0f}"


def test_the_finish_out_pays_loitering():
    """At land 3 / clear 8 the arithmetic favoured sitting under the overhang:
    finishing ends the episode and forfeits the remaining per-step reward."""
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    steps = cfgmod.EPISODE_LENGTH_S / 0.02
    hold = 0.5 / 0.02
    loiter = cfgmod.LAND_W * steps * 0.8
    finish = cfgmod.LAND_W * steps * 0.2 + cfgmod.CLEAR_W * hold
    assert finish > loiter, f"loitering {loiter:.0f} vs finish {finish:.0f}"


def test_standing_tolerance_is_tight_enough_to_look_like_standing():
    """Measured: with a 40 deg tilt tolerance the policy finished at 32.7 deg -
    leaning hard enough to read as a sprawl.  A tolerance is a target."""
    assert em.TILT_MAX_DEG <= 30.0, "40 deg finished at 32.7 - a sprawl"
    assert em.TILT_MAX_DEG >= 22.0, "18 deg removed the reward and the run collapsed"
    # The trunk band used to be the anti-sprawl guard (<= 2.8 cm). That job now
    # belongs to head carriage (`standing_tall`), and a LIVE standing policy
    # leaves a 2.6 cm band within 0.5 s (measured 2026-09-22), so the band is
    # wide and the posture test is what must be tight.
    assert em.STAND_BAND <= 0.08
    assert em.HEAD_CARRY_MIN >= 0.12


def test_contact_buffer_is_big_enough_for_a_braced_duck():
    """The default 35 overflows ("increase nconmax to 46") and an overflow
    silently drops contacts - the wall support the brace depends on."""
    for task in ("Mjlab-Exit-MicroDuck", "Mjlab-Chimney-MicroDuck", "Mjlab-Approach-MicroDuck"):
        cfg = load_env_cfg(task, play=False)
        assert cfg.sim.nconmax >= 120, f"{task} nconmax={cfg.sim.nconmax}"


def test_success_must_be_HELD_long_enough_to_be_standing():
    """With a 0.5 s hold the duck struck the pose and toppled straight after:
    the episode ended at the first qualifying instant, so nothing trained the
    seconds that follow."""
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.terminations["landed"].params["hold_s"] >= 1.5


def test_the_stepped_wall_is_placed_ON_THE_FLOOR():
    """Its two segments carry their own heights inside the spec, so offsetting
    the body by half a wall height lifts the whole corridor off the ground."""
    import inspect

    src = inspect.getsource(em.reset_exit_state)
    assert "stepped_wall_mocap_z" in src
    assert "cs.wall_z()" not in src
    assert ex.stepped_wall_mocap_z() == 0.0


def test_platform_is_clear_of_the_climbing_slot():
    """Flush with the slot mouth, the platform underside snagged the duck on
    the way past."""
    assert ex.platform_near_y() > 0.5 * cs.WALL_DEPTH_M + 0.03


def test_the_corridor_ENDS_above_the_platform():
    assert ex.wall_top_z() == pytest.approx(ex.PLATFORM_TOP_M + ex.WALL_ABOVE_M)
    assert 1.2 * 0.25 < ex.WALL_ABOVE_M < 1.8 * 0.25, "about a duck and a half"


def test_arrival_bank_is_applied_after_the_normal_state():
    """Same rule as the chimney's bank: it replaces only the pose, after every
    other per-env field is set, or reward/termination read stale values."""
    import inspect

    src = inspect.getsource(em.reset_exit_state)
    assert src.index("handover_bank and handover_prob") > src.index('st["spawn_z"][env_ids] = z')
    assert "PLATFORM_TOP_M - float(bank[\"platform_top\"])" in src, "bank must re-base z onto the current platform"


def test_head_carriage_separates_standing_from_the_sprawl():
    """Root tilt cannot tell the two apart; head carriage must.

    Measured 2026-09-21 over 256 envs: jaw-above-ankle is 0.186 m on a braced
    arrival and 0.069 m once the policy settles into the sprawl, while ROOT
    tilt reads 16.7 deg and 22.2 deg - indistinguishable. The old success test
    was position-only and the old fall test tripped at 80 deg of root tilt, so
    the sprawl scored as a win.
    """
    from mjlab_microduck.tasks import exit_mdp as em

    assert em.HEAD_CARRY_COLLAPSE < 0.069 + 0.02
    assert em.HEAD_CARRY_COLLAPSE < em.HEAD_CARRY_MIN < em.HEAD_CARRY_STAND
    # the gate must reject the measured sprawl and accept the measured stand
    assert 0.069 < em.HEAD_CARRY_MIN
    assert em.HEAD_CARRY_MIN < 0.186


def test_collapse_is_a_termination_and_carriage_is_paid():
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    assert "collapsed" in cfg.terminations
    assert "ex_carry" in cfg.rewards
    assert cfg.rewards["ex_carry"].weight > 0.0


def test_contact_sensor_maxmatch_is_raised_everywhere():
    """nconmax was raised; contact_sensor_maxmatch was not, and defaults to 64.

    The approach logged "increase Option.contact_sensor_maxmatch to 81" on
    2026-09-21 - contact SENSOR matches were being dropped in the same silent
    way broadphase contacts once were, which cost the chain 95 points before.
    """
    from mjlab_microduck.tasks import (
        microduck_approach_env_cfg as apc,
        microduck_chimney_env_cfg as chc,
        microduck_exit_env_cfg as exc,
    )

    for make in (apc.make_microduck_approach_env_cfg,
                 chc.make_microduck_chimney_env_cfg,
                 exc.make_microduck_exit_env_cfg):
        cfg = make(play=False)
        assert cfg.sim.nconmax >= 200
        assert cfg.sim.contact_sensor_maxmatch >= 200, make


def test_exit_takes_over_just_above_the_platform():
    """The handover band must sit low, or the climb overshoots to reach it.

    Measured 2026-09-21: with the old (0.02, 0.22) band the climb handed over
    with its feet 18-29 cm above the platform. Clamping only the render gate
    made it worse (4/8 -> 2/8) because feet +5 cm is below anything the exit
    was trained on, so the band itself has to come down.
    """
    from mjlab_microduck.tasks import exit_mdp as em

    lo, hi = em.ABOVE_PLATFORM_RANGE
    assert lo < 0.02, "the low end must reach below the old floor"
    assert hi <= 0.14, "the high end is what the climb overshoots to"
    assert lo < hi


def test_collapse_needs_the_duck_out_of_the_slot():
    """Inside the slot the braced head is low by design; the collapse test
    must not fire there. Ungated it ended 112/120 episodes in 0.7 s (x21)."""
    import inspect

    from mjlab_microduck.tasks import exit_mdp as em

    src = inspect.getsource(em.collapsed)
    assert "wall_end_y" in src
    assert '"stood"' in src, "collapse must be latched on having stood first"


def test_carriage_shaping_has_gradient_at_the_sprawl():
    """A sprawl (0.069) must score visibly, or the shaping is dead (measured:
    std 0.045 gave ex_carry 0.001 for 34 iterations)."""
    import math

    from mjlab_microduck.tasks import exit_mdp as em

    # The std is a curriculum knob: 0.10 while the policy sprawls (0.069),
    # 0.05 once it carries >= 0.12 - at 0.10 a 0.14 crouch already paid 0.81
    # and the policy parked there (2026-09-22). Require gradient at a crouch.
    g = math.exp(-(((0.12 - em.HEAD_CARRY_STAND) / em.HEAD_CARRY_STD) ** 2))
    assert g > 0.15


def test_every_policy_triggerable_termination_is_priced():
    """A termination the policy can trigger with zero cost becomes the cheapest
    policy (measured three times on 2026-09-22: a12 squat-death, x21 falls,
    then x21 collapses once falls were priced)."""
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    assert cfg.rewards["ex_collapse"].weight >= 100
    assert cfg.rewards["ex_fall"].weight > 0


def test_success_is_reachable_from_the_success_spawn():
    """The nearly-done spawn must satisfy the static part of the success test.

    Measured 2026-09-22: with goal == clear line and spawn y = goal +- 0.05,
    only 12 % started past the line, and a 2.6 cm trunk band was left by a live
    policy in 0.5 s - `landed` never fired in any x21 run.
    """
    from mjlab_microduck.robot import exit_stage as ex
    from mjlab_microduck.tasks import exit_mdp as em

    assert em.STAND_BAND >= 0.05
    assert ex.goal_y() + 0.01 > ex.clear_line_y()


def test_stillness_is_shaped_not_only_gated():
    """`landed` requires 2 s below SLOW_MPS; without per-step pay for slowing
    the conjunction never fires (x21: landed 0.0000 at carriage 1.06)."""
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    assert cfg.rewards["ex_still"].weight > 0.0


def test_finishing_beats_camping():
    """Per-step shaping must not be available short of the goal, and the
    terminal must outweigh the stream it ends (x21 camped short of the line:
    carriage 0.198, past-line 0.0 %, 2026-09-22)."""
    import inspect

    from mjlab_microduck.tasks import exit_mdp as em
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    assert "clear_line_y" in inspect.getsource(em.carriage_reward)
    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    per_step_stream = (cfg.rewards["ex_carry"].weight + cfg.rewards["ex_still"].weight) * cfg.episode_length_s
    assert cfg.rewards["ex_landed_bonus"].weight > per_step_stream


def test_upright_is_shaped_and_carriage_minimum_is_not_a_crouch():
    """x21 parked at carriage 0.14 (min 0.13) in a >26 deg lean: the minimum
    must sit nearer the measured stand (0.186) and tilt must be shaped."""
    from mjlab_microduck.tasks import exit_mdp as em
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    assert em.HEAD_CARRY_MIN >= 0.15
    assert em.HEAD_CARRY_STD <= 0.06
    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    assert cfg.rewards["ex_level"].weight > 0.0


def test_the_resting_lean_is_a_costed_termination():
    """x21 parked in a >26 deg lean past the line, tall and still, for whole
    episodes (2026-09-22). A free rest pose beats any shaping; it must end
    the episode at a cost, and the finish must pay more than a fall costs."""
    from mjlab_microduck.tasks import microduck_exit_env_cfg as cfgmod

    cfg = cfgmod.make_microduck_exit_env_cfg(play=False)
    assert "leaning" in cfg.terminations
    assert cfg.rewards["ex_lean"].weight >= 1000

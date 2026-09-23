"""Corridor approach: the invariants that would silently break the task."""
from __future__ import annotations

import math

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.tasks import approach_mdp as ap
from mjlab_microduck.tasks.microduck_approach_env_cfg import STAND_OVERRIDES, WIDTH

TASK = "Mjlab-Approach-MicroDuck"


def test_obs_contract_is_61d_and_slots_kept():
    cfg = load_env_cfg(TASK, play=True)
    terms = cfg.observations["actor"].terms
    for slot in ("command", "head_command", "body_command"):
        assert slot in terms, f"{slot} must stay in the obs contract, zero-padded if unused"
    assert terms["body_command"].params["dim"] == 6


def test_penalties_have_positive_weights():
    """The self-negating penalties take POSITIVE weights; a negative weight
    double-negates into a reward for the violation."""
    cfg = load_env_cfg(TASK, play=True)
    for name in ("ap_jam", "ap_fall"):
        assert cfg.rewards[name].weight > 0, f"{name} is self-negating and needs a positive weight"


def test_arrival_bootstraps_but_falling_does_not():
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.terminations["arrived"].time_out is True, \
        "success must bootstrap, or reaching the goal is valued like falling over"
    assert cfg.terminations["fell"].time_out is False


def test_default_width_matches_the_climb_clip():
    """The user asked the entry to work at the width used in the climb video."""
    assert WIDTH == pytest.approx(0.125)


def test_stand_overrides_are_the_climb_handover_pose():
    from mjlab_microduck.tasks.microduck_chimney_env_cfg import UPRIGHT_OVERRIDES

    assert STAND_OVERRIDES == UPRIGHT_OVERRIDES, \
        "the approach must finish in exactly the pose the climb starts from"


def test_spawn_is_outside_the_throat():
    """A spawn inside the corridor would make the task trivially solved."""
    cfg = load_env_cfg(TASK, play=True)
    lo, hi = cfg.events["set_approach_state"].params["distance_range"]
    assert lo > 0.0 and hi >= lo
    # reset places the robot at -(distance + half throat), so it always starts
    # clear of the walls
    assert lo + ap.INSIDE_Y_M > 0.5 * cs.WALL_DEPTH_M


def test_handover_pose_needs_every_component():
    """A duck squeezed off the floor must NOT count as arrived - the entry
    probe once scored a trunk at 30 cm as a 100 % handover."""
    assert ap.STAND_BAND < 0.05
    assert ap.STAND_Z - ap.STAND_BAND > 0.08
    # 30 cm is far outside the band
    assert abs(0.30 - ap.STAND_Z) > ap.STAND_BAND


def test_gait_reward_is_gated_on_distance_not_the_stale_command():
    """The walking recipe gates air-time on the twist command, which this task
    never asks the policy to follow - so it would be paid to march on the spot
    after arriving."""
    cfg = load_env_cfg(TASK, play=True)
    air = cfg.rewards["air_time"]
    assert air.func is ap.gait_gate, "air_time must be re-gated for the approach"
    assert air.params["far_m"] > 0.0
    assert air.params["base_params"]["command_threshold"] == 0.0


def test_progress_is_rate_limited_so_diving_cannot_win():
    """a1 crossed 37 cm in 0.9 s - 38 cm/s average against a MEASURED sustained
    maximum of 30 - because an uncapped progress reward at weight 300 made
    hurrying the argmax."""
    assert ap.MAX_PAID_SPEED <= 0.40, "the cap must be near real walking speed"
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.rewards["ap_dash"].weight > 0, "the dash penalty is self-negating"


def test_capped_progress_pays_no_more_for_a_dive_than_for_a_walk():
    dt = 0.02
    walk = min(0.20 * dt, ap.MAX_PAID_SPEED * dt)
    dive = min(0.76 * dt, ap.MAX_PAID_SPEED * dt)
    assert dive == pytest.approx(ap.MAX_PAID_SPEED * dt)
    assert dive / walk < 2.0, "a dive must not out-earn a walk by much"


def test_the_speed_penalty_is_SMOOTHED_not_instantaneous():
    """Measured: a normal gait peaks at 0.5-0.9 m/s instantaneous while
    sustaining under 0.30, so an instantaneous threshold charges ordinary
    walking and no amount of walking better escapes it."""
    import inspect

    src = inspect.getsource(ap.approach_dash_penalty)
    assert "speed_ema" in src, "charge a smoothed speed, not the instantaneous one"
    assert ap.SUSTAINED_CAP >= 0.30, "the cap must sit above the measured sustained maximum"


def test_approach_and_climb_share_a_ROBOT_MODEL():
    """A policy chain whose halves run different bodies is not a handover.  The
    velocity recipe brings the walk model and the climb the collision-covered
    one; the approach must follow the climb."""
    ap_cfg = load_env_cfg(TASK, play=True)
    ch_cfg = load_env_cfg("Mjlab-Chimney-MicroDuck", play=True)
    ap_spec = ap_cfg.scene.entities["robot"].spec_fn
    ch_spec = ch_cfg.scene.entities["robot"].spec_fn
    name = lambda f: getattr(f, "__qualname__", str(f))  # noqa: E731
    assert name(ap_spec) == name(ch_spec), (
        f"approach uses {name(ap_spec)} but the climb uses {name(ch_spec)}"
    )


def test_approach_default_pose_is_the_STANCE_not_the_brace():
    """Sharing the model must not drag in the climb's default pose: actions are
    offsets from it and this policy has to walk."""
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    cfg = load_env_cfg(TASK, play=True)
    # compared by VALUE: the registry deep-copies cfgs, so identity never holds
    assert cfg.scene.entities["robot"].init_state.joint_pos == HOME_FRAME.joint_pos


def test_the_head_can_actually_hit_a_wall():
    """The walk model carries collision geometry on the FEET ONLY - two geoms -
    so the beak passes straight through a corridor wall (user 2026-09-20: "he
    walks with his head straight through the wall").  The corridor tasks need
    the collision-covered model, where the head and jaw are solid."""
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        m = env.sim.mj_model
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""  # noqa: E731
        wall = next(g for g in range(m.ngeom) if "corridor_left" in name(g))
        head = [g for g in range(m.ngeom)
                if name(g).startswith("robot/") and ("jaw" in name(g) or "head" in name(g))]
        assert head, "no head geoms on the robot at all"

        def can_collide(a, b):
            return bool((int(m.geom_contype[a]) & int(m.geom_conaffinity[b]))
                        or (int(m.geom_contype[b]) & int(m.geom_conaffinity[a])))

        assert any(can_collide(g, wall) for g in head), \
            "no head geom can collide with a corridor wall"
    finally:
        del env


def test_handover_requires_the_CLIMBS_STANCE_at_rest():
    """Measured: with only a trunk-speed test the approach "arrived" 99.6 % of
    the time mid-stride - hips 19-23 deg off the climb's stance, joints
    swinging at 1.12 rad/s - and the climb then gained 1.9 cm instead of
    climbing.  A slow trunk is not a robot standing still."""
    from mjlab_microduck.tasks.microduck_chimney_env_cfg import UPRIGHT_OVERRIDES

    assert ap._STAND_TARGETS == UPRIGHT_OVERRIDES, \
        "the handover pose must be the pose the climb actually trains from"
    assert ap.STAND_JOINT_TOL < 0.35, "tolerance must be tighter than the measured 0.33-0.41 rad error"
    assert ap.STAND_JOINT_VEL <= 1.1, "must exclude the measured 1.12 rad/s mid-stride handover"


def test_handover_requires_FACING_ACROSS_the_slot():
    """Measured: a nose-first approach hands over at |yaw| = 74 deg and the
    climb then gains nothing.  The climb braces back-to-one-wall and
    feet-to-the-other, so it must face ACROSS the corridor - which is why the
    entry has to be side-on."""
    import math

    assert ap.STAND_YAW_TOL < math.radians(30.0)
    import inspect

    src = inspect.getsource(ap.in_handover_pose)
    assert "facing_across" in src


def test_spawn_faces_across_so_the_entry_is_SIDE_ON():
    import inspect

    src = inspect.getsource(ap.reset_approach_state)
    assert "math.pi / 2.0 + (torch.rand" not in src, "spawn must not face along the slot"


def test_spawn_mix_includes_nearly_done_so_the_bonus_can_FIRE():
    """a7 added the yaw requirement and scored 0 % arrivals: the handover bonus
    never fired, so nothing pulled the policy off its nose-first habit."""
    cfg = load_env_cfg(TASK, play=True)
    p = cfg.events["set_approach_state"].params
    assert p["nearly_done_prob"] > 0.0 and p["frontier_prob"] > 0.0
    assert p["nearly_done_prob"] + p["frontier_prob"] < 1.0


def test_orientation_is_shaped_every_step_not_only_at_the_goal():
    """a8 reached the handover 0 % of the time from a real outside spawn and
    ended at |yaw| = 94 deg: progress pays for distance whichever way the duck
    faces, so turning and walking forwards beat side-stepping."""
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.rewards["ap_yaw"].weight > 0, "self-negating, so a positive weight"


def test_handover_must_be_DEEP_enough_in_the_throat():
    """Measured: the climb goes 0 % from y = -13 cm, 73 % from -9, and 92-96 %
    from -5 to +13.  a9 stopped at -13.1 - just inside the throat by the old
    test and exactly where the climb cannot start."""
    assert ap.HANDOVER_Y_M <= 0.08, "the old +-15 cm let the approach stop at the lip"
    assert ap.HANDOVER_Y_M < ap.INSIDE_Y_M


def test_entry_side_flips_the_spawn_sign(monkeypatch):
    """MICRODUCK_AP_SIDE=+1 puts the spawn on the camera side.

    Rotating the duck 180 degrees instead of retraining looked free and is not:
    the duck side-steps toward ONE body direction and the exit's command is
    world-frame, so a rotated duck exits off the back of the platform (measured
    2026-09-21: 0 of 8 seeds survived). Entering on the exit's side is a
    different gait, so it is a trained variant, not a render-time transform.
    """
    import importlib

    import mjlab_microduck.tasks.approach_mdp as ap

    monkeypatch.setenv("MICRODUCK_AP_SIDE", "1")
    importlib.reload(ap)
    assert ap.SIDE == 1.0
    monkeypatch.setenv("MICRODUCK_AP_SIDE", "-1")
    importlib.reload(ap)
    assert ap.SIDE == -1.0


def test_approach_trains_under_the_hardware_command_bounds():
    """The approach runs as a guest in the exit env, whose action term is the
    bounded one. Trained unbounded, a11 hands over 99.8 % at home and 84 % as
    a guest (2026-09-22). Bounds in training AND deployment - the standing rule."""
    from mjlab_microduck.tasks import microduck_approach_env_cfg as apc
    from mjlab_microduck.tasks import microduck_exit_env_cfg as exc

    a = apc.make_microduck_approach_env_cfg(play=False).actions["joint_pos"]
    e = exc.make_microduck_exit_env_cfg(play=False).actions["joint_pos"]
    assert type(a) is type(e), (type(a), type(e))
    assert "Bounded" in type(a).__name__

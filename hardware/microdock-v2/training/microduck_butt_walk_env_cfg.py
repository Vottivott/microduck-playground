"""Microduck *butt walk* task — seated locomotion on command.

The duck sits in the sitstand SIT pose and scoots on its bottom: forward, BACKWARD, turning, and still at the all-zero
command. Made for the seated MicroDock (v2): the duck walks up to the dock, turns around, sits with the official
sitstand policy, then butt-walks backward until the magnetic tip in its USB-C port meets the dock's adapter, and rests
seated while it charges (its legs carry almost no load, so it draws far less power than standing).

    twist = [vx, vy, wz] in the trunk frame (the usual velocity command; all-zero = sit still, the deployment idle)

The predecessor is the uncommanded "butt scoot" actor (reports/butt-scoot-policy-20260914), cut from the desk-climb
route: it scoots forward only, with all 13 command slots zero. This env adds the command, so the same skill runs
backward and stops. Its actor can warm-start this one (same 61D obs, 14 actions, 512-256-128 ELU).

Built on the sitstand env (same full-collision robot, seated contact-solver hardening, velocity-parity DR / obs noise /
delays, commandable head, NaN guard, BAM friction fields), with:
  - the posture flag replaced by a UniformVelocityCommand on the same twist slot (obs layout unchanged, 61D);
  - every reset seated (SIT keyframe + joint/tilt noise);
  - the posture stack replaced by velocity tracking + a seated band: a height Gaussian at SEATED_Z, a one-sided L1
    above SEATED_Z_SOFT and a termination above SEATED_Z_MAX ("stood up": walking on the feet is the easy way to move);
  - stillness at the zero command (stopping at the dock and resting there is the deployment case);
  - the three sitstand terms that read the sit/stand flag dropped (seated head jerk, seated tip, neck strain when
    leaned): they gate on the posture command, which this env no longer has.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG, get_standup_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import (
    SITTING_TARGET_OVERRIDES,
    _LEG_JOINTS,
    MicroduckSitStandRlCfg,
    make_microduck_sitstand_env_cfg,
)

EPISODE_LENGTH_S = 12.0

# ── Command ranges (trunk frame) ───────────────────────────────────────────────
# Biased backward: backing into the dock is the job. vy keeps a tiny range so its input stays alive (dead-weight rule).
LIN_VEL_X_RANGE = (-0.10, 0.06)
LIN_VEL_Y_RANGE = (-0.01, 0.01)
ANG_VEL_Z_RANGE = (-0.30, 0.30)
ZERO_COMMAND_FRAC = 0.2              # exact all-zero commands: sit still (the deployment idle and the docked rest)
COMMAND_RESAMPLE_S = (2.5, 6.0)

# ── Seated band (trunk z, m) ───────────────────────────────────────────────────
# SIT_Z = 0.060 is the measured seated rest; the butt-scoot actor rides 0.067-0.083 while moving.
SEATED_Z = 0.065
SEATED_Z_SOFT = 0.080                # L1 above this
SEATED_Z_MAX = 0.095                 # terminate above this (standing is 0.115)

# Sit UPRIGHT: the dock's hood meets the trunk in the trunk frame, so a consistent, near-zero seated pitch keeps the
# magnetic tip square to the adapter (the official sit leans back ~7 deg). Sharp Gaussian on tilt (std 0.07 = ~4 deg).
ENABLE_SEATED_UPRIGHT = True
SEATED_UPRIGHT_STD = 0.07
SEATED_UPRIGHT_WEIGHT = 1.5

# Seat friction. The seat slides on the floor all the time, so its friction matters: a plastic shell on PLA or wood is
# ~0.3-0.5, while the model's collision geoms default to 1.0. The export leaves them unnamed (only the soles are named),
# which mjlab's CollisionCfg and friction DR can't address, so _seated_spec names them <body>_<k>_collision first.
# Every collision geom then gets priority 1 (its own friction is used against the terrain) and the non-foot ones a
# friction randomized per env over SEAT_FRICTION_RANGE.
_FEET = r"^(left|right)_foot_collision$"
SEAT_FRICTION_RANGE = (0.3, 1.0)
SEATED_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FEET: 3, ".*_collision": 3},
    priority={_FEET: 1, ".*_collision": 1},
    friction={_FEET: (1.0,), ".*_collision": (sum(SEAT_FRICTION_RANGE)/2,)},
    disable_other_geoms=False,          # the unnamed self-collision-only geoms keep their settings
)


def _seated_spec():
    """The standup robot (robot_groundcontact.xml) with its unnamed terrain-colliding geoms named <body>_<k>_collision."""
    spec = get_standup_spec()
    counts: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.name or geom.contype != 1:
            continue
        body = geom.parent.name; k = counts.get(body, 0); counts[body] = k + 1
        geom.name = f"{body}_{k}_collision"
    return spec


# Discovering the stroke: a slippery seat first, then the realistic range (iterations).
ENABLE_SEAT_FRICTION_CURRICULUM = True
SEAT_FRICTION_STAGES = ((0, (0.05, 0.25)), (400, (0.15, 0.45)), (900, (0.25, 0.7)), (1500, SEAT_FRICTION_RANGE))
# Progress along the commanded vx / wz (linear, capped): the Gaussians are flat at the small docking speeds.
ENABLE_PROGRESS = True

# Head up. Left to the Gaussian head tracking alone, the policy bowed its head ~85 deg forward (neck pitch -1.15 rad)
# to put its weight over its feet and unload the seat: the camera an inch off the floor. A linear price and a hard
# limit keep the head at the commanded pose (small balance motions are still free).
HEAD_L1_WEIGHT = 5.0
HEAD_MAX_ERROR = 0.8                 # rad

TRACK_LIN_STD = 0.04                 # m/s: the errors that matter at docking speeds (~0.05 m/s)
TRACK_ANG_STD = 0.25                 # rad/s


def make_microduck_butt_walk_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_sitstand_env_cfg(play=play, rough=rough)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.scene.entities = {"robot": replace(MICRODUCK_STANDUP_ROBOT_CFG, spec_fn=_seated_spec, collisions=(SEATED_COLLISION,))}
    cfg.events["seat_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(?!(left|right)_foot_collision$).*_collision$",)),
                "operation": "abs", "ranges": SEAT_FRICTION_STAGES[0][1] if ENABLE_SEAT_FRICTION_CURRICULUM else SEAT_FRICTION_RANGE,
                "shared_random": True},
    )
    if ENABLE_SEAT_FRICTION_CURRICULUM:
        cfg.curriculum["seat_friction_range"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={"event_name": "seat_friction",
                    "param_stages": [{"step": it*24, "params": {"ranges": rng}} for it, rng in SEAT_FRICTION_STAGES]},
        )

    # ── Command: the usual velocity twist on the same slot ─────────────────────
    sit_cmd = cfg.commands["twist"]
    cfg.commands["twist"] = UniformVelocityCommandCfg(
        entity_name=sit_cmd.entity_name,
        resampling_time_range=COMMAND_RESAMPLE_S,
        rel_standing_envs=ZERO_COMMAND_FRAC,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        debug_vis=play,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=LIN_VEL_X_RANGE, lin_vel_y=LIN_VEL_Y_RANGE, ang_vel_z=ANG_VEL_Z_RANGE, heading=None,
        ),
    )

    # ── Drop the posture stack and every term that reads the sit/stand flag ────
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_height", "posture_height_sharp", "posture_height_l1",
                 "rise_bootstrap", "posture_stillness", "posture_composite", "upright_while_tall", "neck_strain_leaned"):
        cfg.rewards.pop(name, None)
    for name in ("seated_head_jerk", "seated_tip"):
        cfg.events.pop(name, None)
    for name in ("seated_head_jerk_prob", "seated_tip_range"):
        cfg.curriculum.pop(name, None)

    # ── Task: track the twist while seated ─────────────────────────────────────
    cfg.rewards["track_linear_velocity"] = RewardTermCfg(
        func=mdp.track_linear_velocity, weight=3.0, params={"command_name": "twist", "std": TRACK_LIN_STD},
    )
    cfg.rewards["track_angular_velocity"] = RewardTermCfg(
        func=mdp.track_angular_velocity, weight=1.5, params={"command_name": "twist", "std": TRACK_ANG_STD},
    )
    if ENABLE_PROGRESS:
        cfg.rewards["progress_lin"] = RewardTermCfg(
            func=microduck_mdp.command_progress, weight=2.0, params={"command_name": "twist", "axis": "lin_x", "min_cmd": 0.01},
        )
        cfg.rewards["progress_ang"] = RewardTermCfg(
            func=microduck_mdp.command_progress, weight=1.0, params={"command_name": "twist", "axis": "ang_z", "min_cmd": 0.05},
        )
    cfg.rewards["still_at_zero_command"] = RewardTermCfg(
        func=microduck_mdp.stillness_at_zero_command, weight=1.0,
        params={"command_name": "twist", "command_threshold": 0.01, "vel_std": 0.03},
    )
    # Seated band: Gaussian at the riding height, one-sided L1 above the soft cap (self-negating → POSITIVE weight).
    cfg.rewards["seated_height"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian, weight=1.0, params={"target_height": SEATED_Z, "std": 0.02},
    )
    cfg.rewards["seated_height_cap"] = RewardTermCfg(
        func=microduck_mdp.height_above_l1_penalty, weight=20.0, params={"max_height": SEATED_Z_SOFT},
    )
    # Loose seated leg pose: keeps the legs folded in front without blocking the scooting stroke.
    cfg.rewards["seated_leg_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=0.5,
        params={"std": 0.6, "joint_indices": _LEG_JOINTS, "target_overrides": SITTING_TARGET_OVERRIDES},
    )
    if ENABLE_SEATED_UPRIGHT:
        cfg.rewards["seated_upright"] = RewardTermCfg(
            func=mdp.upright, weight=SEATED_UPRIGHT_WEIGHT, params={"std": SEATED_UPRIGHT_STD, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        )
    # Kept from sitstand: head_pose_tracking, upright_linear, descent_speed / rise_speed (anti-hop caps), gentle_motion
    # (|a_z|), action_rate / torque_rate (with their curricula), body_ang_vel, angular_momentum, self_collisions.

    cfg.rewards["head_pose_tracking"].params.update({"fine_std": 0.15, "fine_weight": 0.5})
    cfg.rewards["head_pose_tracking"].weight = 1.5
    if HEAD_L1_WEIGHT > 0:
        cfg.rewards["head_pose_l1"] = RewardTermCfg(func=microduck_mdp.head_pose_l1, weight=HEAD_L1_WEIGHT,
                                                   params={"command_name": "head_pose"})

    # ── Terminations ───────────────────────────────────────────────────────────
    cfg.terminations["stood_up"] = TerminationTermCfg(
        func=microduck_mdp.root_height_above, params={"max_height": SEATED_Z_MAX},
    )
    if HEAD_MAX_ERROR > 0:
        cfg.terminations["head_off_command"] = TerminationTermCfg(
            func=microduck_mdp.head_off_command, params={"max_error": HEAD_MAX_ERROR, "command_name": "head_pose"},
        )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(60.0)},
    )

    # ── Resets: always seated ──────────────────────────────────────────────────
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.06, 0.075)
    gs = cfg.events["set_ground_state"].params
    gs["sitting_prob"] = 1.0
    gs["standing_prob"] = 0.0
    return cfg


MicroduckButtWalkRlCfg = deepcopy(MicroduckSitStandRlCfg)
MicroduckButtWalkRlCfg.experiment_name = "microduck_butt_walk"
MicroduckButtWalkRlCfg.run_name = "microduck_butt_walk"
MicroduckButtWalkRlCfg.save_interval = 100
MicroduckButtWalkRlCfg.max_iterations = 3000

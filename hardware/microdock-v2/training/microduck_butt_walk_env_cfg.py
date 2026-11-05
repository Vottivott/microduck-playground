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
import os
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
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
# Backing up straight, the docking's command: this share of the resamples gets no turn and |vx| >= 0.02 m/s (75 %
# backward), and a linear yaw-rate cost (mdp.yaw_rate_error_l1, positive weight) where the Gaussian has gone flat.
# Both off by default (the released policy trained without them).
STRAIGHT_FRAC = float(os.environ.get("MICRODUCK_BW_STRAIGHT", "0"))
YAW_L1_WEIGHT = float(os.environ.get("MICRODUCK_BW_YAW_L1", "0"))

# ── Seated band (trunk z, m) ───────────────────────────────────────────────────
# SIT_Z = 0.060 is the measured seated rest; the butt-scoot actor rides 0.067-0.083 while moving.
SEATED_Z = 0.065
SEATED_Z_SOFT = 0.080                # L1 above this
SEATED_Z_MAX = 0.095                 # terminate above this (standing is 0.115)

# Sit UPRIGHT: the dock's hood meets the trunk in the trunk frame, so a consistent, near-zero seated pitch keeps the
# magnetic tip square to the adapter (the official sit leans back ~7 deg). Sharp Gaussian on tilt (std 0.07 = ~4 deg).
ENABLE_SEATED_UPRIGHT = os.environ.get("MICRODUCK_BW_UPRIGHT", "1") != "0"
SEATED_UPRIGHT_STD = float(os.environ.get("MICRODUCK_BW_UPRIGHT_STD", "0.07"))
SEATED_UPRIGHT_WEIGHT = float(os.environ.get("MICRODUCK_BW_UPRIGHT_WEIGHT", "1.5"))
# ...and a linear tilt cost: the Gaussian alone has no pull at the ~9 deg the policies settled at.
TILT_L1_WEIGHT = float(os.environ.get("MICRODUCK_BW_TILT_L1", "0"))
# ...or a linear cost on pitch away from the scooting lean (one consistent seated pitch, moving or at rest).
PITCH_TARGET_DEG = float(os.environ.get("MICRODUCK_BW_PITCH_TARGET", "3.0"))
PITCH_L1_WEIGHT = float(os.environ.get("MICRODUCK_BW_PITCH_L1", "0"))
# ...only at rest: the stroke itself rocks the trunk, and any tilt cost while moving killed it.
REST_PITCH_WEIGHT = float(os.environ.get("MICRODUCK_BW_REST_PITCH", "0"))

# Seat friction. The seat slides on the floor all the time, so its friction matters: a plastic shell on PLA or wood is
# ~0.3-0.5, while the model's collision geoms default to 1.0. The export leaves them unnamed (only the soles are named),
# which mjlab's CollisionCfg and friction DR can't address, so _seated_spec names them <body>_<k>_collision first.
# Every collision geom then gets priority 1 (its own friction is used against the terrain) and the non-foot ones a
# friction randomized per env over SEAT_FRICTION_RANGE.
_FEET = r"^(left|right)_foot_collision$"
SEAT_FRICTION_RANGE = tuple(float(v) for v in os.environ.get("MICRODUCK_BW_SEAT_MU", "0.3,1.0").split(","))
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
ENABLE_SEAT_FRICTION_CURRICULUM = os.environ.get("MICRODUCK_BW_MU_CURRICULUM", "1") != "0"
SEAT_FRICTION_STAGES = ((0, (0.05, 0.25)), (400, (0.15, 0.45)), (900, (0.25, 0.7)), (1500, SEAT_FRICTION_RANGE))
# Progress along the commanded vx / wz (linear, capped): the Gaussians are flat at the small docking speeds.
ENABLE_PROGRESS = os.environ.get("MICRODUCK_BW_PROGRESS", "1") != "0"

# Head up. Left to the Gaussian head tracking alone, the policy bowed its head ~85 deg forward (neck pitch -1.15 rad)
# to put its weight over its feet and unload the seat: the camera an inch off the floor. A linear price and a hard
# limit keep the head at the commanded pose (small balance motions are still free).
HEAD_L1_WEIGHT = float(os.environ.get("MICRODUCK_BW_HEAD_L1", "5.0"))
HEAD_MAX_ERROR = float(os.environ.get("MICRODUCK_BW_HEAD_TERM", "0.8"))      # rad; 0 = no termination

# Hips off their stops. Unpriced, the policy parked both hip rolls (and a hip yaw) on their stops with targets far past
# them (+94..+340 deg): the servos pushed at their current limit even sitting still, ~18 W of motor heat in the CPU
# rehearsal against 0.5 W on the official sit policy. The price lives on the qpos side, next to the hard limits (the
# command-side overshoot is how a low-kp servo reaches its target, so it stays free); the leg torque^2 (~ motor heat)
# can be priced too, staged in once the stroke exists ("ITER:WEIGHT,...", weights as magnitudes). 0 / "" = off.
HIP_LIMIT_WEIGHT = float(os.environ.get("MICRODUCK_BW_HIP_LIMIT", "0"))
HIP_LIMIT_MARGIN = float(os.environ.get("MICRODUCK_BW_HIP_LIMIT_MARGIN", "0.15"))     # rad from each hard stop
LEG_TORQUE_STAGES = [tuple(float(v) for v in s.split(":")) for s in os.environ.get("MICRODUCK_BW_TORQUE", "").split(",") if s]
_HIPS = r"^(left|right)_hip_(roll|yaw)$"
_LEG_ACTUATORS = r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle)$"

# Undocking (off by default): this share of the episodes starts docked at the MicroDock, the magnet holding the port and
# the hood over the back (mdp.undock_forces), with a forward command for the first UNDOCK_CMD_S seconds. The magnet's
# pull is unknown (no adapter measured yet), so it is drawn per episode from a range that grows over training:
# "iteration:lo-hi,..." in newtons. The CPU docking simulation models 4 N; the released butt walk breaks free of 1 N only.
UNDOCK_PROB = float(os.environ.get("MICRODUCK_BW_UNDOCK", "0"))
UNDOCK_F_STAGES = [(int(it), tuple(float(v) for v in rng.split("-"))) for it, rng in
                   (s.split(":") for s in os.environ.get("MICRODUCK_BW_UNDOCK_F", "0:0.5-1.5,300:1-3,600:1.5-4.5,900:2-6").split(","))]
UNDOCK_VX_RANGE = tuple(float(v) for v in os.environ.get("MICRODUCK_BW_UNDOCK_VX", "0.03,0.06").split(","))
UNDOCK_CMD_S = float(os.environ.get("MICRODUCK_BW_UNDOCK_CMD_S", "5.0"))
UNDOCK_PROGRESS_WEIGHT = float(os.environ.get("MICRODUCK_BW_UNDOCK_PROGRESS", "0"))    # per mm of net progress out of the dock
UNDOCK_CRITIC = os.environ.get("MICRODUCK_BW_UNDOCK_CRITIC", "0") != "0"              # the magnet in the critic's obs
UNDOCK_YAW_K = float(os.environ.get("MICRODUCK_BW_UNDOCK_YAW_K", "2.0"))   # N m/rad: the hood's hold on the heading (0 = off)

# The sit-stand curricula the butt walk inherits harden at iteration 1500 (head-pose range 0.49 -> 0.91, pushes x2,
# action rate -1.0, rise speed 5) and the butt walk collapses there: head-off terminations x15, episodes 580 -> 250
# steps, never recovering by 2000 (the released policy is the iteration-1400 checkpoint). This drops their stages from
# this iteration on (0 = keep them all); the butt walk's own curricula (seat friction, magnet, leg torques) keep theirs.
CURRICULUM_CAP = int(os.environ.get("MICRODUCK_BW_CURRICULUM_CAP", "0"))
_OWN_CURRICULA = ("seat_friction_range", "undock_magnet", "leg_torques_weight")

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
    straight = dict(rel_straight_envs=STRAIGHT_FRAC) if STRAIGHT_FRAC > 0 else {}
    cfg.commands["twist"] = (microduck_mdp.VelocityCommandCommandOnlyCfg if straight else UniformVelocityCommandCfg)(
        **straight,
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
    if YAW_L1_WEIGHT > 0:
        cfg.rewards["yaw_rate_l1"] = RewardTermCfg(
            func=microduck_mdp.yaw_rate_error_l1, weight=YAW_L1_WEIGHT, params={"command_name": "twist"},
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
    if TILT_L1_WEIGHT > 0:
        cfg.rewards["tilt_l1"] = RewardTermCfg(func=microduck_mdp.tilt_l1, weight=TILT_L1_WEIGHT)
    if PITCH_L1_WEIGHT > 0:
        cfg.rewards["pitch_target_l1"] = RewardTermCfg(func=microduck_mdp.pitch_target_l1, weight=PITCH_L1_WEIGHT,
                                                      params={"target_deg": PITCH_TARGET_DEG})
    if REST_PITCH_WEIGHT > 0:
        cfg.rewards["rest_pitch_l1"] = RewardTermCfg(func=microduck_mdp.rest_pitch_l1, weight=REST_PITCH_WEIGHT,
                                                    params={"target_deg": PITCH_TARGET_DEG, "command_name": "twist"})
    # Kept from sitstand: head_pose_tracking, upright_linear, descent_speed / rise_speed (anti-hop caps), gentle_motion
    # (|a_z|), action_rate / torque_rate (with their curricula), body_ang_vel, angular_momentum, self_collisions.

    cfg.rewards["head_pose_tracking"].params.update({"fine_std": 0.15, "fine_weight": 0.5})
    cfg.rewards["head_pose_tracking"].weight = 1.5
    if HEAD_L1_WEIGHT > 0:
        cfg.rewards["head_pose_l1"] = RewardTermCfg(func=microduck_mdp.head_pose_l1, weight=HEAD_L1_WEIGHT,
                                                   params={"command_name": "head_pose"})

    # Hips off their stops (costs >= 0 -> NEGATIVE weights).
    if HIP_LIMIT_WEIGHT > 0:
        cfg.rewards["hip_limit_proximity"] = RewardTermCfg(
            func=microduck_mdp.joint_pos_limit_proximity, weight=-HIP_LIMIT_WEIGHT,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(_HIPS,)), "margin": HIP_LIMIT_MARGIN},
        )
    if LEG_TORQUE_STAGES:
        first_it, first_w = LEG_TORQUE_STAGES[0]
        cfg.rewards["leg_torques"] = RewardTermCfg(
            func=mdp.joint_torques_l2, weight=-first_w if first_it == 0 else 0.0,
            params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(_LEG_ACTUATORS,))},
        )
        cfg.curriculum["leg_torques_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "leg_torques", "weight_stages": [{"step": int(it)*24, "weight": -w} for it, w in LEG_TORQUE_STAGES]},
        )

    # ── Undocking: docked starts, the magnet and the hood, a forward command ───
    if UNDOCK_PROB > 0:
        cfg.events["undock_reset"] = EventTermCfg(
            mode="reset", func=microduck_mdp.undock_reset,
            params={"prob": UNDOCK_PROB, "fmax_range": UNDOCK_F_STAGES[0][1], "vx_range": UNDOCK_VX_RANGE, "cmd_s": UNDOCK_CMD_S},
        )
        cfg.events["undock_forces"] = EventTermCfg(
            mode="step", func=microduck_mdp.undock_forces,
            params={"command_name": "twist", "k_yaw": UNDOCK_YAW_K, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
        )
        if UNDOCK_PROGRESS_WEIGHT > 0:
            cfg.rewards["undock_progress"] = RewardTermCfg(func=microduck_mdp.undock_progress, weight=UNDOCK_PROGRESS_WEIGHT)
        if UNDOCK_CRITIC:
            cfg.observations["critic"].terms["undock"] = ObservationTermCfg(func=microduck_mdp.undock_critic_obs)
        cfg.curriculum["undock_magnet"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={"event_name": "undock_reset",
                    "param_stages": [{"step": it*24, "params": {"fmax_range": rng}} for it, rng in UNDOCK_F_STAGES]},
        )

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

    # ── Inherited curricula capped (CURRICULUM_CAP) ────────────────────────────
    if CURRICULUM_CAP > 0:
        for name, term in cfg.curriculum.items():
            if name in _OWN_CURRICULA:
                continue
            for k, stages in term.params.items():
                if k.endswith("_stages") and isinstance(stages, list):
                    term.params[k] = [st for st in stages if st["step"] < CURRICULUM_CAP*24]
    return cfg


MicroduckButtWalkRlCfg = deepcopy(MicroduckSitStandRlCfg)
MicroduckButtWalkRlCfg.experiment_name = "microduck_butt_walk"
MicroduckButtWalkRlCfg.run_name = "microduck_butt_walk"
MicroduckButtWalkRlCfg.save_interval = 100
MicroduckButtWalkRlCfg.max_iterations = 3000

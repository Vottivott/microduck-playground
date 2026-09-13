"""Microduck stair-ladder climbing task.

Built on the velocity recipe (robot, BAM actuators, full DR + obs-noise +
NaN-guard stack) with the flat terrain replaced by an alternating-tread
stair-ladder (see ``robot/ladder.py`` for the geometry and its rationale).

Task
----
* The twist ``vx`` command slot carries the climb command: exact zero means
  hold still on the ladder (``rel_standing_envs``), otherwise a vertical
  trunk speed in m/s.  The remaining ten command slots carry ladder
  perception (per-foot vector to the next tread, ladder geometry) so the
  actor observation stays 61-D.
* Reward is potential-based height gain plus climb-speed tracking, foot
  support and mild regularizers.  Nothing is latched or gated on a searched
  contact state; body parts leaning on the treads are allowed and only
  lightly priced.
* Episodes spawn either on the floor in front of the ladder (mount) or
  standing on two staggered treads (reverse-curriculum spawn), both from
  physically settled poses.
* Difficulty is an adaptive per-environment level over (riser, angle),
  promoted when the robot climbs and demoted when it falls early.

Environment overrides
---------------------
``MICRODUCK_LADDER_PLAY_LEVEL``, ``MICRODUCK_LADDER_RISER_MM``,
``MICRODUCK_LADDER_ANGLE_DEG`` fix the play-mode geometry;
``MICRODUCK_LADDER_FLOOR_SPAWN`` overrides the floor-spawn probability.
"""

from __future__ import annotations

from copy import deepcopy
import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as base_terminations
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.ladder import (
    LADDER_GEOMETRY,
    StairLadderGeometry,
    make_stair_ladder_entity_cfgs,
    staircase_geometry_from_json,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Adaptive geometry levels: (riser range m, angle range deg).  Every entry
# satisfies the tread-clearance rules for the default 36 mm tread (checked by
# tests/test_ladder_cfg.py).  Level 0 is a steep staircase; level 4 is a
# 60-68 degree ladder with 24-30 mm risers.  From 23 mm risers the next
# same-side tread clears the ankle shell entirely.
LADDER_LEVELS: tuple[dict, ...] = (
    {"riser": (0.015, 0.017), "angle": (45.0, 50.0)},
    {"riser": (0.017, 0.020), "angle": (50.0, 56.0)},
    {"riser": (0.020, 0.023), "angle": (55.0, 60.0)},
    {"riser": (0.023, 0.027), "angle": (58.0, 65.0)},
    {"riser": (0.024, 0.030), "angle": (60.0, 68.0)},
)

CLIMB_SPEED_RANGE = (0.02, 0.05)  # m/s vertical trunk speed command
HOLD_FRACTION = 0.2  # exact-zero command: hold still on the ladder
FLOOR_SPAWN_PROB = 0.15
EPISODE_LENGTH_S = 10.0

# --- staircase variant: mini stairs on a human staircase ---------------------
# One human riser (184.6 mm) = 8 mini risers of 23.1 mm at ~62 deg (inside
# ladder level 3).  Every 8th tread is a full-width landing (the human tread);
# the next flight starts ``flat`` metres past its nose.  Levels shrink the
# flat: the winders' inner side leaves only ~40-50 mm.
# The landing is a solid step block (closed riser): a 20 mm plate let the
# swing foot pass under it and press its toe on the rear face below the top.
# Landing foot target (landing study 2026-09-04, s12 checkpoint from a static
# stance two treads below): at the nose + 5 cm up the foot reached the nose
# line but never past it (2 % on the landing, 88 % head-on-landing
# face-plants); 2 cm past the nose + 2 cm up put the foot on the landing in
# 63 % of attempts (the robot then still toppled, which is what s13 trains).
STAIRCASE_GEOMETRY = StairLadderGeometry(
    num_treads=16,
    landing_every=8,
    rail_length_m=0.20,
    landing_thickness_m=0.20,
    landing_target_ahead_m=0.02,
    landing_target_up_m=0.02,
    landing_step_gate=True,  # s19: no landing target for a foot while the other foot is two treads below
)
# Risers >= 23 mm clear the ankle shell vertically, so any angle is allowed
# (the human riser gives 23.1 mm).
# ``lateral`` (m, to the left) and ``dyaw_deg`` shift and turn the next flight
# about the landing: the measured right-side placement on the corner
# staircase needs flats of 59-121 mm, lateral shifts of -85..+136 mm and
# turns of 0..-18 deg (presentation/stairs/staircase_geometry.json).
# Left-side placement on the corner staircase (chosen 2026-09-04): the flats
# between mini stairs are 12-31 cm long, the winders shift the next flight up
# to 25 cm sideways and turn it up to 18 deg.  The robot walks the flat along
# a curved path (ladder.walk_paths); the levels widen that walk.
STAIRCASE_LEVELS: tuple[dict, ...] = (
    {"riser": (0.023, 0.025), "angle": (58.0, 63.0), "flat": (0.12, 0.20), "lateral": (-0.03, 0.03), "dyaw_deg": (-4.0, 4.0)},
    {"riser": (0.023, 0.025), "angle": (58.0, 65.0), "flat": (0.12, 0.25), "lateral": (-0.10, 0.10), "dyaw_deg": (-10.0, 10.0)},
    {"riser": (0.023, 0.025), "angle": (60.0, 65.0), "flat": (0.15, 0.31), "lateral": (-0.18, 0.18), "dyaw_deg": (-14.0, 14.0)},
    {"riser": (0.023, 0.025), "angle": (60.0, 65.0), "flat": (0.20, 0.32), "lateral": (-0.25, 0.25), "dyaw_deg": (-18.0, 18.0)},
)
STAIRCASE_PATH_SPAWN_FRAC = 0.5  # of the landing spawns: part-way along the walk
STAIRCASE_EPISODE_LENGTH_S = 12.0
STAIRCASE_LANDING_SPAWN_PROB = 0.25


def _play_overrides() -> dict:
    overrides: dict = {}
    level = os.getenv("MICRODUCK_LADDER_PLAY_LEVEL")
    if level is not None:
        overrides["fixed_level"] = int(level)
    riser = os.getenv("MICRODUCK_LADDER_RISER_MM")
    if riser is not None:
        overrides["fixed_riser_m"] = float(riser) / 1000.0
    angle = os.getenv("MICRODUCK_LADDER_ANGLE_DEG")
    if angle is not None:
        overrides["fixed_angle_deg"] = float(angle)
    return overrides


def make_microduck_ladder_env_cfg(
    play: bool = False,
    geometry: StairLadderGeometry = LADDER_GEOMETRY,
    level_table: tuple[dict, ...] = LADDER_LEVELS,
    episode_length_s: float = EPISODE_LENGTH_S,
    landing_spawn_prob: float = 0.0,
    top_spawn_prob: float = 0.0,
    max_start_tread: int = 8,
    head_contact_weight: float = 0.0,
    path_spawn_frac: float = 0.0,
    nose_jitter_m: float = float(os.getenv("MICRODUCK_LADDER_NOSE_JITTER_M", "0.0")),
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # --- scene: all-collision robot (shins/trunk may lean on treads) + ladder
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        **make_stair_ladder_entity_cfgs(geometry),
    }
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.7
    cfg.episode_length_s = episode_length_s
    # Many small boxes and a robot that may lean on several of them: same
    # solver budget as the rough-terrain walker.
    cfg.sim.nconmax = 150
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    # --- commands: twist vx = climb speed, exact zero = hold -----------------
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = HOLD_FRACTION
    twist.rel_turn_in_place_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.resampling_time_range = (4.0, 8.0)
    twist.ranges.lin_vel_x = CLIMB_SPEED_RANGE
    # Unused slots keep a tiny non-zero range so their inputs stay alive.
    twist.ranges.lin_vel_y = (-0.005, 0.005)
    twist.ranges.ang_vel_z = (-0.01, 0.01)
    if play:
        twist.rel_standing_envs = 0.0
        twist.ranges.lin_vel_x = (0.04, 0.04)
    # The head/body pose commands are replaced by ladder perception below.
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)

    # --- observations: fill the 4-D head and 6-D body slots -----------------
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        # Twist slot = [climb speed, sin phase, cos phase] (gait clock).
        terms["command"] = deepcopy(terms["command"])
        terms["command"].func = microduck_mdp.ladder_command_with_phase
        terms["command"].params = {"command_name": "twist"}
        terms["head_command"] = deepcopy(terms["head_command"])
        terms["head_command"].func = microduck_mdp.ladder_foot_targets
        terms["head_command"].params = {"asset_cfg": SceneEntityCfg("robot")}
        terms["body_command"] = deepcopy(terms["body_command"])
        terms["body_command"].func = microduck_mdp.ladder_geometry_obs
        terms["body_command"].params = {"asset_cfg": SceneEntityCfg("robot")}

    # --- rewards ---------------------------------------------------------------
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
    ):
        cfg.rewards.pop(name, None)
    cfg.rewards.pop("upright", None)  # folded into the stance composite
    # Anti-violence pressure (AGENTS.md): impacts and thrash, not rotation
    # speed per se; raised after r10's lunging step attempts.
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["self_collisions"].weight = -1.0

    robot = SceneEntityCfg("robot")
    cfg.rewards["upward_progress"] = RewardTermCfg(
        func=microduck_mdp.ladder_upward_progress,
        weight=1000.0,  # 1 cm/step cap -> at most 10 per step; a fall costs ~-100
        params={"asset_cfg": robot, "command_name": "twist", "max_delta": 0.01},
    )
    # r1-r3 lesson: additive tracking + support + upright + alive let "hold
    # still" keep most of the return under a climb command.  One product:
    # support x upright x tracking.  Hold command (0) pays in full at rest;
    # a climb command pays only the tracking factor unless the robot climbs.
    cfg.rewards["stance"] = RewardTermCfg(
        func=microduck_mdp.ladder_stance_composite,
        weight=2.0,
        params={
            "asset_cfg": robot,
            "command_name": "twist",
            "std": 0.035,
            "tau_s": 0.3,
            "upright_std": 0.45,
            "forward_lean_allow": 0.55,
        },
    )
    # Phase-based reference gait from the leg table (alternation, landing on
    # the next same-side tread, trunk shift).  Paid only under a climb command.
    # r21: alternating-gait references switched off (weight 0, terms kept so
    # the phase clock in the twist slot keeps advancing).  Open-loop playback
    # showed the hand-built alternating reference is not feasible for these
    # servos; the policy's own discovery (rounds 10-14) was a two-footed hop
    # up one tread pair, which the overstep gate now bounds.
    cfg.rewards["gait_reference"] = RewardTermCfg(
        func=microduck_mdp.ladder_gait_reference_tracking,
        weight=0.0,
        params={"asset_cfg": robot, "command_name": "twist", "std": 0.3},
    )
    # Task-space reference for the feet (swing arc to the next same-side
    # tread, support foot on its tread); joint tracking alone was faked by
    # bobbing in place.
    # r17 lesson: with std 25 mm and weight 2 the policy marched in place
    # (arc followed only near its start).  Wider std keeps a gradient across
    # the whole 3-5 cm arc; higher weight makes the far end worth reaching.
    cfg.rewards["gait_foot_reference"] = RewardTermCfg(
        func=microduck_mdp.ladder_gait_foot_tracking,
        weight=0.0,
        params={"asset_cfg": robot, "command_name": "twist", "std": 0.03},
    )
    # Dense, potential-based direction for the swing foot (<= 1.0 per step).
    cfg.rewards["foot_target_progress"] = RewardTermCfg(
        func=microduck_mdp.ladder_foot_target_progress,
        weight=300.0,
        params={"asset_cfg": robot, "command_name": "twist", "max_delta": 0.01, "lower_foot_only": False},
    )
    if geometry.landing_every > 0:
        # Walking across a landing (staircase): remaining-path potential.
        cfg.rewards["path_progress"] = RewardTermCfg(
            func=microduck_mdp.ladder_path_progress,
            weight=300.0,
            params={"asset_cfg": robot, "command_name": "twist", "max_delta": 0.01},
        )
    # Self-negating costs: positive weights are intentional (they return <= 0).
    cfg.rewards["body_ladder_contact"] = RewardTermCfg(
        func=microduck_mdp.ladder_body_contact_penalty, weight=0.02
    )
    if head_contact_weight > 0.0:
        cfg.rewards["head_ladder_contact"] = RewardTermCfg(
            func=microduck_mdp.ladder_head_contact_penalty, weight=head_contact_weight
        )
    cfg.rewards["vertical_impact"] = RewardTermCfg(
        func=microduck_mdp.ladder_vertical_impact_penalty,
        weight=0.5,
        params={"asset_cfg": robot, "clip": 0.5},
    )
    # Keep the swing foot low: cost per metre above next tread + 25 mm.
    cfg.rewards["swing_overshoot"] = RewardTermCfg(
        func=microduck_mdp.ladder_swing_overshoot_penalty,
        weight=50.0,
        params={"asset_cfg": robot, "clearance": 0.025},
    )

    # --- events: ladder placement + spawn replaces the flat-ground reset -----
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    spawn_params = {
        "asset_cfg": robot,
        "geometry": geometry,
        "level_table": level_table,
        "floor_spawn_prob": float(os.getenv("MICRODUCK_LADDER_FLOOR_SPAWN", FLOOR_SPAWN_PROB)),
        "max_start_tread": max_start_tread,
        "landing_spawn_prob": landing_spawn_prob,
        "landing_approach_prob": 0.4 if landing_spawn_prob > 0.0 else 0.0,
        "top_spawn_prob": top_spawn_prob,
        "nose_jitter_m": nose_jitter_m,
        "path_spawn_frac": path_spawn_frac,
        "position_noise": 0.004,
        "yaw_noise_deg": 5.0,
        "tilt_noise_deg": 2.0,
        "joint_noise": 0.03,
        # r2 lesson: static-stance spawns alone plateau at "hold still".
        # Reverse-curriculum spawns partway through a step give on-policy
        # data of the landing, which is where the value of stepping is learned.
        "swing_spawn_prob": 0.3,
        "swing_fraction_range": (0.0, 1.0),  # from lift-off to just landed
        "swing_clearance": 0.012,
        "swing_lateral_shift": 0.02,
        "level_mix_prob": 0.3,
    }
    if play:
        spawn_params.update(_play_overrides())
        spawn_params["swing_spawn_prob"] = float(os.getenv("MICRODUCK_LADDER_SWING_SPAWN", 0.0))
        spawn_params["level_mix_prob"] = 0.0
    reset_term = EventTermCfg(
        func=microduck_mdp.reset_stair_ladder, mode="reset", params=spawn_params
    )
    # Run the spawn before every other reset event.
    cfg.events = {"reset_stair_ladder": reset_term, **cfg.events}
    # Pushes are introduced by curriculum once climbing exists.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }

    # --- terminations -----------------------------------------------------------
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["bad_orientation"] = TerminationTermCfg(
        func=base_terminations.bad_orientation,
        params={"limit_angle": math.radians(65.0)},
    )
    cfg.terminations["fallen"] = TerminationTermCfg(
        func=microduck_mdp.ladder_fallen,
        params={"asset_cfg": robot, "min_trunk_above_feet": 0.055, "min_trunk_height": 0.06},
    )
    cfg.terminations["off_side"] = TerminationTermCfg(
        func=microduck_mdp.ladder_off_side,
        params={"asset_cfg": robot, "max_lateral": 0.10},
    )
    cfg.terminations["foot_fling"] = TerminationTermCfg(
        func=microduck_mdp.ladder_foot_fling,
        params={"asset_cfg": robot, "clearance": 0.025, "margin": 0.05},
    )
    cfg.terminations["overstep"] = TerminationTermCfg(func=microduck_mdp.ladder_overstep)
    # A hop up one tread pair is airborne for ~0.15-0.25 s; 0.36 s grace.
    cfg.terminations["airborne"] = TerminationTermCfg(
        func=microduck_mdp.ladder_airborne, params={"grace_steps": 18}
    )
    cfg.terminations["reached_top"] = TerminationTermCfg(
        func=microduck_mdp.ladder_reached_top,
        time_out=True,
        params={"asset_cfg": robot, "margin": 0.05},
    )

    # --- curriculum -------------------------------------------------------------
    for name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "head_pose_bias_weight",
        "action_rate_weight",
    ):
        cfg.curriculum.pop(name, None)
    if not play:
        cfg.curriculum["ladder_level"] = CurriculumTermCfg(
            func=microduck_mdp.ladder_level_curriculum,
            params={
                "num_levels": len(level_table),
                "promote_rise_m": 0.08,  # ~3 risers, more than one swing landing
                "demote_rise_m": 0.01,
                "early_fraction": 0.6,
            },
        )
        # Smoothness after skill discovery, never before (AGENTS.md).
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.05},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.1},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -0.2},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -0.3},
                ],
            },
        )
        if "push_robot" in cfg.events:
            cfg.curriculum["push_range"] = CurriculumTermCfg(
                func=microduck_mdp.event_param_curriculum,
                params={
                    "event_name": "push_robot",
                    "param_stages": [
                        # r7 lesson: pushes from iteration 1500 landed while the
                        # gait was still forming; keep them off until late.
                        {"step": 0, "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                        {
                            "step": 4500 * NUM_STEPS_PER_ENV,
                            "params": {"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}},
                        },
                        {
                            "step": 6000 * NUM_STEPS_PER_ENV,
                            "params": {"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
                        },
                    ],
                },
            )
    return cfg


MicroduckLadderRlCfg = deepcopy(MicroduckRlCfg)
MicroduckLadderRlCfg.experiment_name = "ladder_climb"
MicroduckLadderRlCfg.run_name = "stair_ladder"
MicroduckLadderRlCfg.save_interval = 100
MicroduckLadderRlCfg.max_iterations = 4_000


def make_microduck_staircase_env_cfg(play: bool = False, top_spawn_prob: float = 0.0) -> ManagerBasedRlEnvCfg:
    """Mini stairs on a human staircase: two flights of seven mini treads,
    each ending on a full-width landing, with a flat run before the next
    flight.  Same 61-D observation contract as the ladder task, so ladder
    checkpoints resume directly."""
    cfg = make_microduck_ladder_env_cfg(
        play=play,
        geometry=STAIRCASE_GEOMETRY,
        level_table=STAIRCASE_LEVELS,
        episode_length_s=STAIRCASE_EPISODE_LENGTH_S,
        landing_spawn_prob=STAIRCASE_LANDING_SPAWN_PROB,
        top_spawn_prob=top_spawn_prob,
        max_start_tread=12,
        head_contact_weight=1.0,  # head on a landing block only (tripod stall); mini-tread brushes are free
        path_spawn_frac=STAIRCASE_PATH_SPAWN_FRAC,
    )
    if play:
        spawn = cfg.events["reset_stair_ladder"].params
        spawn["landing_spawn_prob"] = float(os.getenv("MICRODUCK_STAIRCASE_LANDING_SPAWN", STAIRCASE_LANDING_SPAWN_PROB))
        # Probe switches: spawn two treads below a landing (the landing step)
        # or part-way along the walking path.
        spawn["landing_approach_prob"] = float(os.getenv("MICRODUCK_STAIRCASE_APPROACH_SPAWN", spawn["landing_approach_prob"]))
        spawn["path_spawn_frac"] = float(os.getenv("MICRODUCK_STAIRCASE_PATH_SPAWN", spawn["path_spawn_frac"]))
        spawn["approach_swing_prob"] = float(os.getenv("MICRODUCK_STAIRCASE_APPROACH_SWING", spawn.get("approach_swing_prob", 0.7)))
    return cfg


MicroduckStaircaseRlCfg = deepcopy(MicroduckLadderRlCfg)
MicroduckStaircaseRlCfg.experiment_name = "staircase"
MicroduckStaircaseRlCfg.run_name = "staircase"


# --- landing-only curriculum (s12) -------------------------------------------------
# The landing-step study (2026-09-04) showed the step from the top mini tread
# onto the human tread is never learned: from two treads below, s11 reaches
# the landing 14 % of the time, and the mid-swing approach spawn used for
# 40 % of the staircase spawns is unrecoverable even for the r25 seed.  This
# variant spends most spawns on the approach (static stance, some mid-swing),
# pays a one-shot bonus for the first foot on a landing, and keeps episodes
# short so the approach is attempted often.
LANDING_APPROACH_PROB = 0.6
# s13: static approach spawns only (the mid-swing ones face-plant regardless).
LANDING_APPROACH_SWING_PROB = float(os.getenv("MICRODUCK_LANDING_APPROACH_SWING", "0.0"))
LANDING_BONUS_WEIGHT = 50.0  # s17: 1.0 one-shot (was 0.2), half a fall
LANDING_EPISODE_LENGTH_S = 8.0


def make_microduck_staircase_landing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_staircase_env_cfg(play=play, top_spawn_prob=float(os.getenv("MICRODUCK_STAIRCASE_TOP_SPAWN", "0.15")))
    cfg.episode_length_s = LANDING_EPISODE_LENGTH_S
    spawn = cfg.events["reset_stair_ladder"].params
    if not play:
        # s15: train the natural arrival.  The natural climb reaches the
        # landing with the trunk pitched forward and the head hits the block
        # (s14 probe: 93 % head contacts, last support tread 4), while the
        # static approach spawn starts upright.  Random on-ladder starts are
        # capped below the first landing so most episodes arrive naturally;
        # approach and landing spawns are forced independently of the cap.
        spawn["floor_spawn_prob"] = 0.30
        spawn["landing_spawn_prob"] = 0.15
        spawn["landing_approach_prob"] = 0.30
        spawn["max_start_tread"] = 4
        spawn["approach_swing_prob"] = LANDING_APPROACH_SWING_PROB
    cfg.rewards["landing_bonus"] = RewardTermCfg(
        func=microduck_mdp.ladder_landing_bonus, weight=LANDING_BONUS_WEIGHT, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    if not play:
        # s20 (from the s14 checkpoint): the s16-s19 penalty stack (head x10,
        # lean, stall x5) turned the landing into a state the policy wants to
        # leave (s19: 100 % face-plants right after arriving).  Keep only the
        # physically justified parts: slower arrivals, the landing-step gate
        # (geometry flag), a mild park price and the larger landing bonus.
        cfg.commands["twist"].ranges.lin_vel_x = (0.015, 0.035)
        # s25: with the arrival solved (s23/s24), a 39 % park on tread 4 is
        # timidity, not an impossible step - price it harder (2 -> 4).
        cfg.rewards["tread_stall"] = RewardTermCfg(
            func=microduck_mdp.ladder_tread_stall_penalty,
            weight=float(os.getenv("MICRODUCK_STAIR_STALL_W", "4.0")),  # s40: 8 with alive 1 (s39 parked on tread 4)
            params={"stall_s": 3.0},
        )
        # s21: approach spawns two to five treads below the landing, so the
        # arrival is trained from a climb, not from a standstill.
        cfg.events["reset_stair_ladder"].params["approach_max_below"] = 5
        # s23: natural climbs arrive at the landing turned ~40 deg from the
        # flight axis and dive; price the heading error on the mini treads
        # and spawn with wider yaw noise so the correction is trained.
        cfg.rewards["flight_heading"] = RewardTermCfg(
            func=microduck_mdp.ladder_flight_heading_penalty, weight=2.0, params={"dead_zone_deg": 8.0}
        )
        cfg.events["reset_stair_ladder"].params["yaw_noise_deg"] = 20.0
    # s29: head/jaw/neck contact with a landing block ends the episode (the
    # landing crossing was a head-first lunge; see ladder_head_touch).
    if os.getenv("MICRODUCK_STAIR_HEAD_TERM", "1") == "1":
        cfg.terminations["head_touch"] = TerminationTermCfg(func=microduck_mdp.ladder_head_touch)
    # s36 lesson: once the policy turns violent the regularizers (action
    # rate) outweigh the progress terms, the per-step return goes negative
    # and the head-touch termination becomes a free exit (episodes 14 steps,
    # 194 head touches per window).  A per-step alive pay keeps continuing
    # worth more than quitting (the tug "upright 5" rule).
    cfg.rewards["alive"] = RewardTermCfg(
        func=microduck_mdp.is_alive, weight=float(os.getenv("MICRODUCK_STAIR_ALIVE", "3.0"))
    )
    # s30: +100 once for standing upright on the top landing (reached_top).
    cfg.rewards["top_bonus"] = RewardTermCfg(
        func=microduck_mdp.ladder_top_bonus, weight=float(os.getenv("MICRODUCK_STAIR_TOP_BONUS", "100")),
        params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.05, "hold_s": 0.2},
    )
    cfg.terminations["reached_top"].params = {"asset_cfg": SceneEntityCfg("robot"), "margin": 0.05, "hold_s": 0.2}
    return cfg


MicroduckStaircaseLandingRlCfg = deepcopy(MicroduckStaircaseRlCfg)
MicroduckStaircaseLandingRlCfg.run_name = "staircase_landing"


# --- demo: the real corner staircase with right-side mini stairs ---------------
# Fixed geometry from presentation/stairs/staircase_geometry.json (playground
# repo), copied to robot/staircase_corner.json: 16 human steps, one mini flight
# per step, landings are the real (winder) tread meshes.  Play/eval only.
STAIRCASE_DEMO_SIDE = os.getenv("MICRODUCK_STAIRCASE_DEMO_SIDE", "left")


def make_microduck_staircase_demo_env_cfg(play: bool = True, side: str = STAIRCASE_DEMO_SIDE) -> ManagerBasedRlEnvCfg:
    geometry, info = staircase_geometry_from_json(side=side)
    r = info["riser"]
    level = ({"riser": (r, r), "angle": (info["angle_deg"], info["angle_deg"]), "flat": (0.1, 0.1)},)
    cfg = make_microduck_ladder_env_cfg(
        play=play,
        geometry=geometry,
        level_table=level,
        episode_length_s=150.0,
        landing_spawn_prob=0.0,
        max_start_tread=0,
        head_contact_weight=1.0,
    )
    spawn = cfg.events["reset_stair_ladder"].params
    spawn["floor_spawn_prob"] = 1.0
    spawn["swing_spawn_prob"] = 0.0
    spawn["level_mix_prob"] = 0.0
    spawn["fixed_level"] = 0
    spawn["landing_approach_prob"] = 0.0
    if not play and os.getenv("MICRODUCK_STAIRCASE_DEMO_TRAIN", "0") == "1":
        # c1 (2026-09-06): TRAIN on the real corner staircase - the s27 climber
        # (straight two-flight curriculum) stalls at the second human step,
        # where the winders shift and turn the next flight.  Spawns spread
        # over the whole staircase: floor, random treads, approaches below
        # every one of the 16 landings.
        spawn["floor_spawn_prob"] = 0.15
        spawn["max_start_tread"] = geometry.num_treads - 5
        spawn["landing_approach_prob"] = 0.5
        spawn["approach_max_below"] = 5
        # c3 probe (2026-09-06): 74 % of climbs reach the last tread and topple
        # there - the climber never practised stopping on THIS summit.  Top-
        # landing spawns (as on the straight staircase) teach the stop.
        spawn["top_spawn_prob"] = 0.15
        cfg.episode_length_s = 30.0
        # c4/c5 lesson (2026-09-07): this cfg builds on the ladder recipe, so
        # the straight staircase's summit terms were missing here - no top
        # bonus, no alive pay, no stall price.  Top spawns died in 0.2 s (the
        # landing-exit hop) for 6000 iterations with nothing paying for a
        # stop.  Mirror the staircase recipe's terms.
        cfg.rewards["alive"] = RewardTermCfg(
            func=microduck_mdp.is_alive, weight=float(os.getenv("MICRODUCK_STAIR_ALIVE", "3.0"))
        )
        cfg.rewards["top_bonus"] = RewardTermCfg(
            func=microduck_mdp.ladder_top_bonus, weight=float(os.getenv("MICRODUCK_STAIR_TOP_BONUS", "100")),
            params={"asset_cfg": SceneEntityCfg("robot"), "margin": 0.05, "hold_s": 0.2},
        )
        cfg.terminations["reached_top"].params = {"asset_cfg": SceneEntityCfg("robot"), "margin": 0.05, "hold_s": 0.2}
        cfg.rewards["tread_stall"] = RewardTermCfg(
            func=microduck_mdp.ladder_tread_stall_penalty,
            weight=float(os.getenv("MICRODUCK_STAIR_STALL_W", "4.0")),
            params={"stall_s": 3.0},
        )
    # 128 treads + 32 rails of contacts per robot: raise the contact budget.
    cfg.sim.nconmax = 300
    cfg.viewer.distance = 1.0
    return cfg


MicroduckStaircaseDemoRlCfg = deepcopy(MicroduckStaircaseRlCfg)
MicroduckStaircaseDemoRlCfg.experiment_name = "staircase_demo"
MicroduckStaircaseDemoRlCfg.run_name = "staircase_demo"

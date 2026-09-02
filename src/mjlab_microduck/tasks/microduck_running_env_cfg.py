"""Forward-running task for Microduck.

This deliberately starts from the proven velocity environment so the 61D
observation contract, BAM actuator model, delays, noise, and sim2real domain
randomization remain identical to the walking policy family.  The task is
forward-only and rewards measured progress beyond ordinary walking speed.

Two registered recipes use this file:

* ``Running`` optimizes forward progress without prescribing a gait.
* ``RunningFlight`` adds a small reward only when a controlled flight phase
  begins.  It does not pay for airtime, so a long ballistic fall is not useful.
"""

import math
import os
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg, RewardTermCfg, SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

RUNNING_PLAY_SPEED = 1.0
RUNNING_SPEED_CAP = float(os.environ.get("MICRODUCK_RUNNING_SPEED_CAP", "1.4"))
RUNNING_TARGET_MAX_SPEED = float(
    os.environ.get("MICRODUCK_RUNNING_TARGET_MAX_SPEED", "1.2")
)
RUNNING_FINAL_ACTION_RATE_WEIGHT = float(
    os.environ.get("MICRODUCK_RUNNING_ACTION_RATE_WEIGHT", "-0.10")
)
RUNNING_FORWARD_PROGRESS_WEIGHT = float(
    os.environ.get("MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT", "5.0")
)
RUNNING_HIGH_SPEED_STAGE_INTERVAL = int(
    os.environ.get("MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL", "500")
)
RUNNING_ENABLE_SYMMETRY = os.environ.get("MICRODUCK_RUNNING_ENABLE_SYMMETRY") == "1"
RUNNING_ENABLE_HEADING_FEEDBACK = (
    os.environ.get("MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK") == "1"
)
RUNNING_STANDING_FRACTION = 0.03
RUNNING_CURRICULUM_START_ITERATION = int(
    os.environ.get("MICRODUCK_RUNNING_CURRICULUM_START_ITERATION", "0")
)

# The lower edge rises too: zero-command behavior comes from the explicit
# standing bucket, rather than consuming most running samples near zero.
_RUNNING_BASE_SPEED_STAGES = (
    {"step": 0 * NUM_STEPS_PER_ENV, "min_speed": 0.20, "max_speed": 0.45},
    {"step": 1000 * NUM_STEPS_PER_ENV, "min_speed": 0.30, "max_speed": 0.55},
    {"step": 2000 * NUM_STEPS_PER_ENV, "min_speed": 0.40, "max_speed": 0.65},
    {"step": 3000 * NUM_STEPS_PER_ENV, "min_speed": 0.50, "max_speed": 0.75},
    {"step": 4000 * NUM_STEPS_PER_ENV, "min_speed": 0.60, "max_speed": 0.85},
    {"step": 5000 * NUM_STEPS_PER_ENV, "min_speed": 0.70, "max_speed": 0.95},
    {"step": 6000 * NUM_STEPS_PER_ENV, "min_speed": 0.90, "max_speed": 1.20},
)


def _running_speed_stages(
    target_max_speed: float, high_speed_stage_interval: int = 500
) -> tuple[dict, ...]:
    """Return the proven base curriculum plus an optional continuation ramp."""
    if target_max_speed < 1.2:
        raise ValueError("running target max speed must be at least 1.2 m/s")
    if high_speed_stage_interval <= 0:
        raise ValueError("running high-speed stage interval must be positive")
    if target_max_speed == 1.2:
        return _RUNNING_BASE_SPEED_STAGES
    intermediate_1 = min(1.35, target_max_speed)
    intermediate_2 = min(1.50, target_max_speed)
    intermediate_3 = min(1.65, target_max_speed)
    stages = _RUNNING_BASE_SPEED_STAGES + (
        {
            "step": 6750 * NUM_STEPS_PER_ENV,
            "min_speed": 0.95,
            "max_speed": intermediate_1,
        },
        {
            "step": 7250 * NUM_STEPS_PER_ENV,
            "min_speed": 1.05,
            "max_speed": intermediate_2,
        },
        {
            "step": 7750 * NUM_STEPS_PER_ENV,
            "min_speed": 1.15,
            "max_speed": intermediate_3,
        },
    )

    # Checkpoint 8,749 has consolidated the 1.65 m/s slice.  Targets above
    # that must arrive as new post-checkpoint stages; replacing the 7,750
    # target would make a resumed run jump immediately to its final command.
    if target_max_speed <= 1.65:
        return stages

    high_speed_targets: list[float] = []
    next_target = 1.80
    while next_target < target_max_speed - 1e-9:
        high_speed_targets.append(round(next_target, 2))
        next_target += 0.15
    high_speed_targets.append(target_max_speed)

    return stages + tuple(
        {
            "step": (8750 + index * high_speed_stage_interval)
            * NUM_STEPS_PER_ENV,
            "min_speed": round(max(1.25, max_speed - 0.55), 2),
            "max_speed": max_speed,
        }
        for index, max_speed in enumerate(high_speed_targets)
    )


RUNNING_SPEED_STAGES = _running_speed_stages(
    RUNNING_TARGET_MAX_SPEED, RUNNING_HIGH_SPEED_STAGE_INTERVAL
)


def make_microduck_running_env_cfg(
    play: bool = False, flight_reward_weight: float = 0.0
):
    """Build the flat-ground, forward-only running environment."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.episode_length_s = 12.0
    # Microduck is only ~25 cm tall; the velocity recipe's 3 m follow camera
    # hides gait and contact details in rollout videos.
    cfg.viewer.distance = 0.55
    cfg.viewer.max_extra_envs = 0

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0 if play else RUNNING_STANDING_FRACTION
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.resampling_time_range = (12.0, 12.0)
    if play:
        command.ranges.lin_vel_x = (RUNNING_PLAY_SPEED, RUNNING_PLAY_SPEED)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
    else:
        command.ranges.lin_vel_x = (
            RUNNING_SPEED_STAGES[0]["min_speed"],
            RUNNING_SPEED_STAGES[0]["max_speed"],
        )
        # Tiny non-zero lateral/yaw ranges keep those command neurons alive,
        # while still making essentially every active sample forward-running.
        command.ranges.lin_vel_y = (-0.02, 0.02)
        command.ranges.ang_vel_z = (-0.05, 0.05)

    if RUNNING_ENABLE_HEADING_FEEDBACK:
        # Slot 2 becomes signed spawn-heading error, recomputed every step.  It
        # remains in the same position and mirrors with the same sign rule as a
        # yaw-rate command, so the deployment observation contract is unchanged.
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.heading = None
        command.ranges.ang_vel_z = (0.0, 0.0)
        command = microduck_mdp.SpawnHeadingVelocityCommandCfg(
            **vars(command), heading_error_clip=1.0
        )
        cfg.commands["twist"] = command

    # Max-speed objective.  Velocity tracking remains useful as a curriculum
    # guide, but forward progress is strong enough that exceeding the command is
    # profitable.  This is what makes the optimum "as fast as possible".
    cfg.rewards["forward_progress"] = RewardTermCfg(
        func=microduck_mdp.running_forward_progress,
        weight=RUNNING_FORWARD_PROGRESS_WEIGHT,
        params={"speed_cap": RUNNING_SPEED_CAP},
    )
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.15)
    cfg.rewards["track_angular_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    cfg.rewards["planar_drift"] = RewardTermCfg(
        func=microduck_mdp.running_planar_drift_cost,
        weight=-0.05,
        params={"command_name": "twist", "lateral_weight": 4.0},
    )
    # Rate tracking alone can only say "stop turning"; once a heading error
    # exists it provides no signal for which way to steer back.  Anchor yaw to
    # each randomized spawn heading so straight running is the actual optimum.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.5,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Permit the forward lean and fast leg cycling a sprint needs.  These stay
    # non-zero only to rule out tumbling and unbounded thrash as cheap optima.
    cfg.rewards["pose"].weight = 0.15
    cfg.rewards["upright"].weight = 0.75
    cfg.rewards["upright"].params["std"] = math.sqrt(0.15)
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["foot_slip"].weight = -0.05
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["foot_clearance"].params["target_height"] = 0.015
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.015

    # Head/body slots remain present and sample tiny non-zero ranges, preserving
    # the 61D hot-swap contract, but posture precision must not block discovery.
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.rewards["head_pose_bias"].weight = 0.0
    cfg.rewards["body_pose_tracking"].weight = 0.0

    if flight_reward_weight > 0.0:
        cfg.rewards["flight_event"] = RewardTermCfg(
            func=microduck_mdp.running_flight_event,
            weight=flight_reward_weight,
            params={
                "sensor_name": "feet_ground_contact",
                "min_forward_speed": 0.30,
                "max_tilt_deg": 50.0,
                "min_airborne_steps": 3,
            },
        )

    # A clean speed-discovery phase: no random velocity kicks and no scheduled
    # widening of CoM/head commands or precision taxes.  Fixed initial DR stays
    # active, so the result is not a deterministic-sim-only policy.
    cfg.events.pop("push_robot", None)
    for name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "com_range",
        "head_com_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(name, None)

    if not play:
        cfg.curriculum["running_speed_range"] = CurriculumTermCfg(
            func=microduck_mdp.running_command_ranges_curriculum,
            params={
                "command_name": "twist",
                "speed_stages": [
                    {
                        **stage,
                        "step": stage["step"]
                        - RUNNING_CURRICULUM_START_ITERATION * NUM_STEPS_PER_ENV,
                    }
                    for stage in RUNNING_SPEED_STAGES
                ],
            },
        )
        # Smoothness is introduced only after a fast gait exists.
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {
                        "step": (0 - RUNNING_CURRICULUM_START_ITERATION)
                        * NUM_STEPS_PER_ENV,
                        "weight": -0.02,
                    },
                    {
                        "step": (2500 - RUNNING_CURRICULUM_START_ITERATION)
                        * NUM_STEPS_PER_ENV,
                        "weight": -0.05,
                    },
                    {
                        "step": (4000 - RUNNING_CURRICULUM_START_ITERATION)
                        * NUM_STEPS_PER_ENV,
                        "weight": RUNNING_FINAL_ACTION_RATE_WEIGHT,
                    },
                    {
                        "step": (5500 - RUNNING_CURRICULUM_START_ITERATION)
                        * NUM_STEPS_PER_ENV,
                        "weight": RUNNING_FINAL_ACTION_RATE_WEIGHT,
                    },
                ],
            },
        )
    else:
        cfg.curriculum.pop("action_rate_weight", None)

    return cfg


MicroduckRunningRlCfg = deepcopy(MicroduckRlCfg)
MicroduckRunningRlCfg.experiment_name = "running"
MicroduckRunningRlCfg.run_name = "running-max-speed"
MicroduckRunningRlCfg.algorithm.entropy_coef = 0.02
MicroduckRunningRlCfg.algorithm.symmetry_cfg = (
    deepcopy(SYMMETRY_CFG) if RUNNING_ENABLE_SYMMETRY else None
)
MicroduckRunningRlCfg.max_iterations = 7_500

MicroduckRunningFlightRlCfg = deepcopy(MicroduckRunningRlCfg)
MicroduckRunningFlightRlCfg.experiment_name = "running_flight"
MicroduckRunningFlightRlCfg.run_name = "running-flight-event"

"""Progressive locomotion task for platform-to-peg Microduck stilts.

Each run compiles one fixed morphology.  Checkpoints transfer between runs in
this order: support-shape blends at 20 mm, then increasing heights on the
12 mm peg.  Keeping morphology fixed within a run makes contacts and inertias
auditable and avoids non-stationary physics inside PPO rollouts.
"""

from __future__ import annotations

import os
from copy import deepcopy

from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.robot.stilt_constants import make_stilt_walk_robot_cfg
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

STILT_HEIGHT_MM = float(os.environ.get("MICRODUCK_STILT_HEIGHT_MM", "20"))
STILT_BLEND = float(os.environ.get("MICRODUCK_STILT_BLEND", "0"))
_STILT_MASS_TEXT = os.environ.get("MICRODUCK_STILT_MASS_KG")
STILT_MASS_KG = float(_STILT_MASS_TEXT) if _STILT_MASS_TEXT else None
STILT_PLAY_SPEED = float(os.environ.get("MICRODUCK_STILT_PLAY_SPEED", "0.15"))


def _stage_name(height_mm: float, blend: float) -> str:
    return f"stilt-h{height_mm:03.0f}-b{blend * 100:03.0f}"


def make_microduck_stilt_env_cfg(
    play: bool = False,
    height_mm: float = STILT_HEIGHT_MM,
    blend: float = STILT_BLEND,
    mass_kg: float | None = STILT_MASS_KG,
):
    """Build one fixed stilt morphology on the proven velocity recipe."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    cfg.scene.entities["robot"] = make_stilt_walk_robot_cfg(
        height_mm=height_mm,
        blend=blend,
        mass_kg=mass_kg,
    )
    cfg.episode_length_s = 10.0
    # Stay close at short stages; expand only enough to keep very tall stilts
    # in frame.  At 20 mm this is the same close 0.55 m view as running.
    cfg.viewer.distance = max(0.55, 0.45 + 2.0 * height_mm * 0.001)
    cfg.viewer.max_extra_envs = 0

    # The original reset height puts the old sole about 3 mm above the floor.
    # Raise both bounds by the exact stilt length so the new tips start with the
    # same clearance and no penetration impulse.
    reset_height = cfg.events["reset_base"].params["pose_range"]["z"]
    height_m = height_mm * 0.001
    cfg.events["reset_base"].params["pose_range"]["z"] = (
        reset_height[0] + height_m,
        reset_height[1] + height_m,
    )

    command = cfg.commands["twist"]
    command.resampling_time_range = (4.0, 8.0)
    command.rel_heading_envs = 0.0
    if play:
        command.rel_standing_envs = 0.0
        command.rel_turn_in_place_envs = 0.0
        command.ranges.lin_vel_x = (STILT_PLAY_SPEED, STILT_PLAY_SPEED)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
    else:
        # A large exact-zero bucket teaches quiet balance.  The active bucket
        # still spans forward/backward, lateral, and yaw motion from stage 0.
        command.rel_standing_envs = 0.25
        command.rel_turn_in_place_envs = 0.05
        command.ranges.lin_vel_x = (-0.12, 0.25)
        command.ranges.lin_vel_y = (-0.06, 0.06)
        command.ranges.ang_vel_z = (-0.35, 0.35)

    # Balance and commanded motion are the task.  Head-command slots remain
    # non-zero and observable, but posture precision must not block gait
    # discovery on a new support geometry.
    cfg.rewards["upright"].weight = 3.0
    cfg.rewards["track_linear_velocity"].weight = 2.5
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["air_time"].weight = 2.0
    cfg.rewards["head_pose_tracking"].weight = 0.25
    cfg.rewards["head_pose_bias"].weight = 0.0
    cfg.rewards["body_pose_tracking"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["foot_slip"].weight = -0.1 - 0.1 * blend

    # Discovery first: no pushes, no widening morphology-unrelated command or
    # CoM distributions, and no early smoothness tax.  Robustness is a later
    # consolidation stage after each morphology can already walk.
    cfg.events.pop("push_robot", None)
    for curriculum_name in (
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "com_range",
        "head_com_range",
        "head_pose_bias_weight",
    ):
        cfg.curriculum.pop(curriculum_name, None)

    if play:
        cfg.curriculum.pop("action_rate_weight", None)
    else:
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.02},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.05},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -0.10},
                    {"step": 4000 * NUM_STEPS_PER_ENV, "weight": -0.20},
                ],
            },
        )

    return cfg


MicroduckStiltRlCfg = deepcopy(MicroduckRlCfg)
MicroduckStiltRlCfg.experiment_name = "stilt_locomotion"
MicroduckStiltRlCfg.run_name = _stage_name(STILT_HEIGHT_MM, STILT_BLEND)
MicroduckStiltRlCfg.save_interval = 100
MicroduckStiltRlCfg.max_iterations = 4_000

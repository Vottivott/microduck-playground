"""MicroDuck self-pumped swing task on two tension-only strings.

The robot begins at the bottom of the arc with exactly zero root and joint
velocity.  There is no phase clock and no scripted impulse: PPO must discover
how to move the articulated head and legs to add pendulum energy.  The actor
keeps the standard 61D deployment observation contract; exact string/seat
kinematics are privileged to the critic and rewards only.
"""

import math
import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_SWING_ROBOT_CFG,
    SWING_ANCHOR_HEIGHT,
    SWING_ATTACHMENT_Z,
    SWING_HANG_LENGTH,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


EPISODE_LENGTH_S = 24.0
NUM_STEPS_PER_ENV = 48


def make_microduck_swing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg()

    swing_robot_cfg = MICRODUCK_SWING_ROBOT_CFG
    if os.environ.get("MICRODUCK_SWING_NOMINAL_ACTUATOR") == "1":
        # Simulation-only mechanism-limit mode: fix ordinary actuator DR at
        # conservative midpoint values. Masses, BAM electrical dynamics,
        # current limits, friction, control delay, strings and contacts remain
        # unchanged. The default task remains fully randomized.
        swing_robot_cfg = deepcopy(MICRODUCK_SWING_ROBOT_CFG)
        actuator_cfg = swing_robot_cfg.articulation.actuators[0]
        actuator_cfg.vin_range = (7.35, 7.35)
        actuator_cfg.vin_drop_gain_range = (0.10, 0.10)
        actuator_cfg.delay_min_lag = 5
        actuator_cfg.delay_max_lag = 5
    cfg.scene.entities = {"robot": swing_robot_cfg}
    cfg.scene.sensors = ()
    cfg.scene.num_envs = 4096
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    # The string anchors are world-fixed sites. Manager reset adds env_origins
    # only to free bodies, not to world sites, so a visual grid would stretch
    # every non-central string to the grid origin. MuJoCo-Warp worlds are
    # already physically isolated; overlapping their coordinates is correct.
    cfg.scene.env_spacing = 0.0
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    cfg.viewer.lookat = (0.0, 0.0, 0.39)
    cfg.viewer.distance = 1.15
    cfg.viewer.elevation = -13.0
    cfg.viewer.azimuth = 100.0
    cfg.episode_length_s = EPISODE_LENGTH_S

    joint_action = cfg.actions["joint_pos"]
    assert isinstance(joint_action, JointPositionActionCfg)
    # Keep the trained 0.7-rad contract by default.  The explicit environment
    # override supports bounded physical-limit audits and staged continuation
    # without silently changing exported policies or ordinary evaluations.
    joint_action.scale = float(os.environ.get("MICRODUCK_SWING_ACTION_SCALE", "0.7"))

    # Task objective. Convex frontier progress makes genuinely large new arcs
    # worth much more than repeated small motion; height remains dense while
    # kinetic energy is only a modest phase-learning aid.
    cfg.rewards = {
        "swing_peak_progress": RewardTermCfg(
            func=microduck_mdp.swing_bidirectional_peak_progress,
            weight=224.0,
            params={
                "target_angle": math.pi,
                "max_paid_rate": 8.0,
                "frontier_power": 2.0,
            },
        ),
        "swing_height": RewardTermCfg(
            func=microduck_mdp.swing_height_reward,
            weight=8.0,
        ),
        "swing_late_height": RewardTermCfg(
            func=microduck_mdp.swing_late_height_reward,
            weight=24.0,
            params={"time_power": 2.0},
        ),
        "swing_energy": RewardTermCfg(
            func=microduck_mdp.swing_energy_reward,
            weight=0.5,
            params={"max_equivalent_height": 2.0},
        ),
        "swing_lateral": RewardTermCfg(
            func=microduck_mdp.swing_lateral_penalty,
            weight=-3.0,
            params={"tolerance_m": 0.03},
        ),
        "swing_lateral_barrier": RewardTermCfg(
            func=microduck_mdp.swing_lateral_barrier_penalty,
            weight=-8.0,
            # Robust-consolidation envelope: begin charging before the 20 mm
            # validation gate so moderately asymmetric actuator realizations
            # cannot settle just outside it. The previous 18/10 mm barrier
            # was nearly free at the gate and only became decisive after a
            # visibly large excursion had already developed.
            params={"safe_m": 0.012, "scale_m": 0.008},
        ),
        "swing_lateral_velocity": RewardTermCfg(
            func=microduck_mdp.swing_lateral_velocity_penalty,
            weight=-3.0,
            params={"tolerance_m_s": 0.1},
        ),
        "swing_out_of_plane_angular_velocity": RewardTermCfg(
            func=microduck_mdp.swing_out_of_plane_angular_velocity_penalty,
            weight=-1.0,
            params={"tolerance_rad_s": 1.5},
        ),
        "swing_alignment": RewardTermCfg(
            func=microduck_mdp.swing_attachment_alignment_penalty,
            weight=-18.0,
        ),
        "swing_alignment_barrier": RewardTermCfg(
            func=microduck_mdp.swing_attachment_alignment_barrier_penalty,
            weight=-4.0,
            params={"safe_penalty": 0.04, "scale": 0.01},
        ),
        "swing_predictive_geometry_barrier": RewardTermCfg(
            func=microduck_mdp.swing_predictive_geometry_barrier_penalty,
            weight=-6.0,
            params={
                "horizon_s": 0.08,
                "lateral_safe_m": 0.012,
                "lateral_scale_m": 0.008,
                "alignment_safe_penalty": 0.04,
                "alignment_scale": 0.01,
                "max_penalty": 16.0,
            },
        ),
        "string_slack": RewardTermCfg(
            func=microduck_mdp.swing_string_slack_penalty,
            weight=-8.0,
        ),
        "string_extension_barrier": RewardTermCfg(
            func=microduck_mdp.swing_string_extension_barrier_penalty,
            weight=-12.0,
            params={
                "safe_length_m": 0.392,
                "scale_m": 0.002,
                "max_penalty": 16.0,
            },
        ),
        # Once a two-step strict-gate violation is latched, positive swing
        # rewards stay disabled and this cost persists through the complete
        # 24-second rollout. Its bounded geometry severity remains
        # action-sensitive, teaching recovery rather than assigning the same
        # binary cost to every possible post-latch state.
        "invalid_episode_debt": RewardTermCfg(
            func=microduck_mdp.swing_invalid_episode_debt,
            weight=-10.0,
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03),
        "joint_torques_l2": RewardTermCfg(
            func=microduck_mdp.joint_torques_l2,
            weight=-0.001,
        ),
    }

    # Keep the 61D actor layout: 48 proprioception + heading3 + head4 + body6.
    # The actor receives only deployable IMU/encoder/history signals. The
    # three otherwise-unused twist slots carry an IMU-derived swing-plane
    # heading error, enabling feedback against yaw drift without privileged
    # string/anchor sensing. The critic additionally sees exact string
    # kinematics to reduce variance.
    actor = cfg.observations["actor"]
    critic = cfg.observations["critic"]
    for group in (actor, critic):
        group.terms.pop("height_scan", None)
        group.terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4}
        )
        group.terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6}
        )
    actor.terms.pop("base_lin_vel", None)
    actor.terms["command"] = ObservationTermCfg(
        func=microduck_mdp.swing_plane_heading_observation,
    )
    for name in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
        critic.terms.pop(name, None)
    critic.terms["swing_state"] = ObservationTermCfg(
        func=microduck_mdp.swing_state_observation,
    )
    # Replace the critic's six constant body-command slots with exact
    # mechanism-validity context.  This is privileged training information
    # only; the actor's body-command slots stay zero and deployment remains
    # the same 61D IMU/encoder/history contract.
    critic.terms["body_command"] = ObservationTermCfg(
        func=microduck_mdp.swing_validity_observation,
    )
    # Replace the critic's otherwise constant zero command slots with the
    # running positive/negative frontiers and episode progress. The actor
    # command slots remain zero, preserving the 61D deployment interface.
    critic.terms["command"] = ObservationTermCfg(
        func=microduck_mdp.swing_frontier_observation,
    )

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for group in (actor, critic):
        for term_name in ("joint_pos", "joint_vel"):
            group.terms[term_name] = deepcopy(group.terms[term_name])
            group.terms[term_name].params["asset_cfg"] = deepcopy(passive_excluded)

    # Retain modest realistic sensor corruption while leaving the physical
    # spawn exactly still.  This prevents a visually perfect but brittle phase
    # detector without handing the actor privileged pivot coordinates.
    actor.terms["base_ang_vel"].noise = Unoise(n_min=-0.02, n_max=0.02)
    actor.terms["projected_gravity"].noise = Unoise(n_min=-0.006, n_max=0.006)
    actor.terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    actor.terms["joint_vel"].noise = Unoise(n_min=-0.15, n_max=0.15)

    # Command exists solely for the standard runtime layout and is always zero.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S)
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)

    # Exact stillness at the bottom. reset_root_state_uniform adds these zero
    # deltas to SWING_SEATED_FRAME and the per-environment origin.
    cfg.events = {
        "reset_base": cfg.events["reset_base"],
        "reset_robot_joints": cfg.events["reset_robot_joints"],
        "encoder_bias": cfg.events["encoder_bias"],
        "expand_bam_friction_fields": cfg.events["expand_bam_friction_fields"],
        "reset_action_history": cfg.events["reset_action_history"],
    }
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
    actor.terms["joint_pos"].params["biased"] = True
    critic.terms["joint_pos"].params["biased"] = False

    # A large swing is not a fall. Retain only horizon, numerical, and terrain
    # guards; the plane is well below the suspended mechanism.
    for name in ("fell_over", "out_of_terrain_bounds"):
        cfg.terminations.pop(name, None)
    cfg.terminations["nan_state"].func = microduck_mdp.robot_state_is_nan

    cfg.curriculum = {
        "peak_progress_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_peak_progress",
                "weight_stages": [
                    {"step": 0, "weight": 224.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": 192.0},
                    {"step": 4000 * NUM_STEPS_PER_ENV, "weight": 160.0},
                ],
            },
        ),
        "height_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_height",
                "weight_stages": [
                    {"step": 0, "weight": 8.0},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "weight": 12.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": 16.0},
                ],
            },
        ),
        "energy_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_energy",
                "weight_stages": [
                    {"step": 0, "weight": 0.5},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": 0.25},
                ],
            },
        ),
        "action_rate_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.03},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -0.04},
                    {"step": 4000 * NUM_STEPS_PER_ENV, "weight": -0.05},
                ],
            },
        ),
        "lateral_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_lateral",
                "weight_stages": [
                    {"step": 0, "weight": -3.0},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -4.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -5.0},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -6.0},
                ],
            },
        ),
        "alignment_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "swing_alignment",
                "weight_stages": [
                    {"step": 0, "weight": -18.0},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -24.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -30.0},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -36.0},
                ],
            },
        ),
        "slack_weight": CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "string_slack",
                "weight_stages": [
                    {"step": 0, "weight": -8.0},
                    {"step": 2000 * NUM_STEPS_PER_ENV, "weight": -10.0},
                    {"step": 2500 * NUM_STEPS_PER_ENV, "weight": -12.0},
                    {"step": 3000 * NUM_STEPS_PER_ENV, "weight": -15.0},
                ],
            },
        ),
    }

    if play:
        # Keep checkpoint metrics and videos reproducible while retaining one
        # representative sampled actuator/friction realization. Training
        # overrides this with its configured seed in the launcher.
        cfg.seed = 72
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("encoder_bias", None)

    return cfg


MicroduckSwingRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.002,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
        class_name="mjlab_microduck.tasks.swing_planar_ppo:SwingPlanarCorrectionPPO",
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_swing",
    run_name="microduck_swing",
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=10_000,
)

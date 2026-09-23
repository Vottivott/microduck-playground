"""Microduck standing long jump.

Episodic trick (policy switch = jump now): from a standing start, crouch,
launch forward off both feet, fly, land on both feet and stand up.  Reward
design follows the roulade recipe (dense potential-based progress, landing
terms gated on task completion, reverse-curriculum spawns), see
``long_jump_mdp.py`` for the state machine and reward semantics.

Realism measures (this env is meant to transfer to hardware):
  * Robot model ``MICRODUCK_LONG_JUMP_ROBOT_CFG``: every visible part has a
    convex-hull collision geom (trunk shells, thighs, neck, servos, ankle
    brackets), explicit thigh/shin and hip/thigh contact pairs, no spurious
    self-contacts at HOME (``scripts/audit_collision_coverage.py``).
  * Joint commands bounded to the joint travel by
    ``BoundedJointPositionAction`` (bounded command fed back into the action
    history; ``scripts/export.py`` bakes the same bounds into the ONNX).
  * BAM actuators (voltage + torque limits), full velocity-recipe DR, sensor
    noise and delays, |a_z| impact cost from step 0, self-collision cost,
    non-foot floor contact cost, head-on-floor termination.
"""
from __future__ import annotations

import math
import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.long_jump_robot import MICRODUCK_LONG_JUMP_ROBOT_CFG
from mjlab_microduck.tasks import long_jump_mdp as lj
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.bounded_actions import bounded_cfg_from, command_saturation_penalty
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

ENABLE_SYMMETRY = True

ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# crouch 0.5 s + launch + flight ~0.3 s + landing + 2 s of standing
EPISODE_LENGTH_S = 4.0
STAND_Z = 0.115  # measured standing trunk height (standup/roulade)

# Leg fold used by the crouch / flight spawns (servo-index keyed; the
# roulade tuck anchor without the chin tuck).  fold=1 is a deep squat.
CROUCH_OVERRIDES = {
    2: -1.15,   # left hip_pitch
    3: 1.25,    # left knee
    4: 1.05,    # left ankle
    11: 1.15,   # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Reward weights (see long_jump_mdp for semantics).  A 20 cm jump that is
# stuck pays ~80 (progress) + 60 (bonus) + ~150 (stand annuity over ~2 s).
PROGRESS_WEIGHT = float(os.getenv("MICRODUCK_LJ_PROGRESS_W", "400"))
LANDING_BONUS_WEIGHT = float(os.getenv("MICRODUCK_LJ_BONUS_W", "300"))
STAND_WEIGHT = 1.5
STAND_TAX_WEIGHT = 5.0
BODY_CONTACT_WEIGHT = 0.5
LATERAL_WEIGHT = 0.5
SATURATION_WEIGHT = 0.05
UPRIGHT_WEIGHT = float(os.getenv("MICRODUCK_LJ_UPRIGHT_W", "1.0"))
FALL_COST_WEIGHT = float(os.getenv("MICRODUCK_LJ_FALL_COST_W", "20.0"))  # self-negating, one-shot
ENTROPY_COEF = float(os.getenv("MICRODUCK_LJ_ENTROPY", "0.005"))
INIT_STD = float(os.getenv("MICRODUCK_LJ_INIT_STD", "0.8"))
DAWDLE_WEIGHT = float(os.getenv("MICRODUCK_LJ_DAWDLE_W", "0.05"))
LAUNCH_WEIGHT = float(os.getenv("MICRODUCK_LJ_LAUNCH_W", "60.0"))
TAKEOFF_WEIGHT = float(os.getenv("MICRODUCK_LJ_TAKEOFF_W", "30.0"))
FLIGHT_SPAWN_PROB = float(os.getenv("MICRODUCK_LJ_FLIGHT_PROB", "0.30"))


def make_microduck_long_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(
        name=lj.FEET_SENSOR,
        primary=ContactMatch(mode="geom", pattern=r"^(left_foot_collision|right_foot_collision)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Every other collision geom of the robot against the floor: knees,
    # thighs, trunk, head, neck ... a landing on any of them is a failure.
    body_ground_cfg = ContactSensorCfg(
        name=lj.BODY_SENSOR,
        primary=ContactMatch(mode="geom", pattern=r"^(?!(left|right)_foot_collision$)(?!ankle_(left|right)_).*_collision$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    head_ground_cfg = ContactSensorCfg(
        name=lj.HEAD_SENSOR,
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_LONG_JUMP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, body_ground_cfg, head_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions: bounded joint-position commands ─────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    cfg.actions["joint_pos"] = bounded_cfg_from(joint_pos_action, bound_margin=0.0)

    # ── Rewards ──────────────────────────────────────────────────────────────
    for name in ["track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                 "foot_swing_height", "foot_slip", "pose"]:
        cfg.rewards.pop(name, None)
    cfg.rewards.pop("soft_landing", None)

    # Always-on mild upright pressure + a one-shot fall cost: without them the
    # first 500 iterations of j1/j2 ended every episode in a 0.5 s fall
    # (falling was free and nothing paid for balance before the jump).  A
    # crouch tilts the trunk ~30 deg, which the wide std barely prices.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = UPRIGHT_WEIGHT
    cfg.rewards["upright"].params["std"] = math.sqrt(0.1)
    cfg.rewards["fall_cost"] = RewardTermCfg(func=lj.long_jump_fall_cost, weight=FALL_COST_WEIGHT, params={"max_tilt_deg": 70.0})
    cfg.rewards["long_jump_progress"] = RewardTermCfg(func=lj.long_jump_progress, weight=PROGRESS_WEIGHT)
    cfg.rewards["long_jump_landing"] = RewardTermCfg(
        func=lj.long_jump_landing_bonus, weight=LANDING_BONUS_WEIGHT, params={"max_tilt_deg": 30.0},
    )
    cfg.rewards["long_jump_stand"] = RewardTermCfg(
        func=lj.long_jump_stand,
        weight=STAND_WEIGHT,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": float(os.getenv("MICRODUCK_LJ_POSE_STD", "0.40")),   # v8: 0.25 (tighter post-landing stance)
            "joint_indices": _LEG_JOINTS,
            "dist_zero": 0.03,
            "dist_full": float(os.getenv("MICRODUCK_LJ_DIST_FULL", "0.12")),   # v12 knob: the stand pay saturates here
        },
    )
    # SELF-NEGATING terms → POSITIVE weights (mdp sign convention).
    cfg.rewards["long_jump_stand_tax"] = RewardTermCfg(
        func=lj.long_jump_stand_tax, weight=STAND_TAX_WEIGHT, params={"target_height": STAND_Z},
    )
    cfg.rewards["long_jump_body_contact"] = RewardTermCfg(func=lj.long_jump_body_contact_penalty, weight=BODY_CONTACT_WEIGHT)
    cfg.rewards["long_jump_lateral"] = RewardTermCfg(func=lj.long_jump_lateral_penalty, weight=LATERAL_WEIGHT, params={"yaw_weight": 0.3})
    cfg.rewards["long_jump_launch"] = RewardTermCfg(func=lj.long_jump_launch, weight=LAUNCH_WEIGHT, params={"vz_cap": 0.6})
    cfg.rewards["long_jump_takeoff"] = RewardTermCfg(func=lj.long_jump_takeoff_bonus, weight=TAKEOFF_WEIGHT, params={"max_tilt_deg": 45.0})
    cfg.rewards["long_jump_dawdle"] = RewardTermCfg(func=lj.long_jump_dawdle_penalty, weight=DAWDLE_WEIGHT, params={"after_s": 1.0})
    cfg.rewards["command_saturation"] = RewardTermCfg(func=command_saturation_penalty, weight=SATURATION_WEIGHT, params={"action_name": "joint_pos"})
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty, weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Diagnostics at near-zero weight (Episode_Reward logs WEIGHTED values, so a
    # weight-0 term reads 0): long_jump_distance x100 = landed metres per
    # episode, long_jump_landed x100 = fraction of episodes with a valid
    # landing, long_jump_stepped x-1000 = steps flagged as a run-up.
    # (Episode_Reward = episode sum / 200 steps; at 0.01 these logged 0.0000)
    # long_jump_distance x200 = landed metres per episode, long_jump_landed
    # x200 = valid-landing fraction, long_jump_stepped x-20000 = run-up steps.
    cfg.rewards["long_jump_distance"] = RewardTermCfg(func=lj.long_jump_landed_distance, weight=1.0)
    cfg.rewards["long_jump_landed"] = RewardTermCfg(func=lj.long_jump_landed_flag, weight=1.0)
    cfg.rewards["long_jump_sync"] = RewardTermCfg(func=lj.long_jump_sync, weight=1.0)
    cfg.rewards["long_jump_stepped"] = RewardTermCfg(func=lj.long_jump_stepped, weight=-0.01)

    # Regularisers (roulade values: motion blockers ≈ 0 during discovery).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-0.5, params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (61-D contract, command slots zero-padded) ──────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    cfg.observations["critic"].terms.pop("foot_height", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)

    for name in ("projected_gravity", "base_ang_vel"):
        cfg.observations["actor"].terms[name] = deepcopy(cfg.observations["actor"].terms[name])
        cfg.observations["actor"].terms[name].delay_min_lag = 0
        cfg.observations["actor"].terms[name].delay_max_lag = 1
        cfg.observations["actor"].terms[name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms["projected_gravity"]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ─────────────────────────────────────────────────────────
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["fell"] = TerminationTermCfg(func=lj.long_jump_fell, params={"max_tilt_deg": 70.0}, time_out=False)
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)

    # ── Events ───────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(func=microduck_mdp.expand_bam_friction_fields, mode="startup")
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot_collision", "right_foot_collision")
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events.pop("push_robot", None)

    cfg.events["set_long_jump_state"] = EventTermCfg(
        func=lj.reset_long_jump_state,
        mode="reset",
        params={
            "standing_prob": 0.85 - FLIGHT_SPAWN_PROB,
            "crouch_prob": 0.15,
            "flight_prob": FLIGHT_SPAWN_PROB,
            "crouch_overrides": CROUCH_OVERRIDES,
            # v4: flight spawns start trivially easy (a few cm above the
            # floor, upright, slow) so feet landings happen by chance and the
            # landing annuity is discovered; the curriculum widens them.
            "flight_vx_range": (0.2, 0.5),
            "flight_vz_range": (-0.3, 0.0),
            "flight_z_range": (0.12, 0.14),
            "flight_pitch_range": (math.radians(-5.0), math.radians(10.0)),
            "flight_tuck_range": (0.1, 0.4),
        },
    )
    if play:
        cfg.events["set_long_jump_state"].params.update({"standing_prob": 1.0, "crouch_prob": 0.0, "flight_prob": 0.0})

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add",
                    "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)},
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add",
                    "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)},
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale",
                    "ranges": ARMATURE_RANDOMIZATION_RANGE},
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                    "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)},
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE},
        )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ───────────────────────────────────────────────────────────
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum["long_jump_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_long_jump_state",
            "param_stages": [
                {"step": 0, "params": {"standing_prob": 0.85 - FLIGHT_SPAWN_PROB, "crouch_prob": 0.15, "flight_prob": FLIGHT_SPAWN_PROB,
                                       "flight_vx_range": (0.2, 0.5), "flight_vz_range": (-0.3, 0.0), "flight_z_range": (0.12, 0.14)}},
                {"step": 1500 * 24, "params": {"flight_vx_range": (0.2, 0.7), "flight_vz_range": (-0.3, 0.3), "flight_z_range": (0.12, 0.17)}},
                {"step": 3000 * 24, "params": {"standing_prob": 0.90 - 0.67 * FLIGHT_SPAWN_PROB, "crouch_prob": 0.10, "flight_prob": 0.67 * FLIGHT_SPAWN_PROB,
                                               "flight_vx_range": (0.3, 0.9), "flight_vz_range": (0.0, 0.6), "flight_z_range": (0.13, 0.20)}},
                {"step": 5000 * 24, "params": {"standing_prob": 0.95 - 0.5 * FLIGHT_SPAWN_PROB, "crouch_prob": 0.05, "flight_prob": 0.5 * FLIGHT_SPAWN_PROB}},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]},
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_head_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005}, {"step": 1000 * 24, "range": 0.01}]},
        )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "action_rate_l2", "weight_stages": [
            {"step": 0, "weight": -0.1}, {"step": 1500 * 24, "weight": -0.2}, {"step": 3000 * 24, "weight": -0.4}]},
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "joint_torque_rate_l2", "weight_stages": [
            {"step": 0, "weight": 0.0}, {"step": 2500 * 24, "weight": -5e-4}, {"step": 3500 * 24, "weight": -1e-3}]},
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "gentle_landing", "weight_stages": [
            {"step": 0, "weight": 0.002}, {"step": 2500 * 24, "weight": 0.005}]},
    )
    return cfg


MicroduckLongJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": INIT_STD, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=ENTROPY_COEF,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="long_jump",
    run_name="long_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)

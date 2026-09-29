"""Microduck platform flip: backflip (default) or front flip from a raised
platform onto a crash mat.

Built on the long-jump recipe (collision-covered robot, bounded joint
commands, BAM actuators, full DR, 61-D obs) with the stage, sensors, rewards
and spawns swapped for the flip (see ``flip_mdp.py``).  Feasibility probe
2026-09-17 (``scripts/probe_flip_pushoff.py``): open-loop push-offs give
0.055 N m s backward / 0.040 forward of angular momentum, i.e. 1.2 / 0.9
turns from a 0.8 m drop with the tucked inertia of 0.0030 kg m^2, so the
backflip is trained first.

Env knobs: MICRODUCK_FLIP_DIR=back|front, MICRODUCK_FLIP_H_MIN/H_MAX (m),
MICRODUCK_FLIP_MIDFLIP_PROB, MICRODUCK_FLIP_PROGRESS_W.
"""
from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot import flip_stage
from mjlab_microduck.tasks import flip_mdp as fl
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_long_jump_env_cfg import (
    CROUCH_OVERRIDES,
    STAND_Z,
    _LEG_JOINTS,
    make_microduck_long_jump_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

FLIP_DIR = os.getenv("MICRODUCK_FLIP_DIR", "back")
H_MIN = float(os.getenv("MICRODUCK_FLIP_H_MIN", "0.6"))
H_MAX = float(os.getenv("MICRODUCK_FLIP_H_MAX", "0.8"))
MIDFLIP_PROB = float(os.getenv("MICRODUCK_FLIP_MIDFLIP_PROB", "0.4"))
PROGRESS_WEIGHT = float(os.getenv("MICRODUCK_FLIP_PROGRESS_W", "8.0"))   # x rad, roulade value
STAND_WEIGHT = 4.0
STAND_TAX_WEIGHT = 5.0
FALL_COST_WEIGHT = 10.0
BODY_CONTACT_WEIGHT = 0.3
DAWDLE_WEIGHT = float(os.getenv("MICRODUCK_FLIP_DAWDLE_W", "0.3"))
LAUNCH_WEIGHT = float(os.getenv("MICRODUCK_FLIP_LAUNCH_W", "30.0"))
LATERAL_WEIGHT = 0.5
TAKEOFF_WEIGHT = float(os.getenv("MICRODUCK_FLIP_TAKEOFF_W", "20.0"))   # x m/s upward at a jump-like takeoff
TUCK_WEIGHT = float(os.getenv("MICRODUCK_FLIP_TUCK_W", "0.5"))
HOP_WEIGHT = float(os.getenv("MICRODUCK_FLIP_HOP_W", "3.0"))   # preparatory hops: a flip is ONE departure
HOP_FATAL = os.getenv("MICRODUCK_FLIP_HOP_FATAL", "0") != "0"  # ... and with this, the hop ENDS the episode
ENTROPY_COEF = float(os.getenv("MICRODUCK_FLIP_ENTROPY", "0.005"))
INIT_STD = float(os.getenv("MICRODUCK_FLIP_INIT_STD", "0.8"))
# v7: the landing curriculum keys off the PROCESS step counter, so every
# resumed run restarted it at the trivial stage (f12/f13/f16/f17 spent their
# mid-flip spawns on near-upright 0.15x-speed drops).  Shift the stage
# boundaries earlier by this many iterations (4500 = start at stage 3).
CURRICULUM_SHIFT_ITERS = int(os.getenv("MICRODUCK_FLIP_CURRICULUM_SHIFT", "0"))
EPISODE_LENGTH_S = 3.5

# roulade tuck (legs folded + chin tuck), used by the mid-flip spawns
TUCK_OVERRIDES = {**CROUCH_OVERRIDES, 5: -1.0, 6: 1.0}


def _contact(name: str, primary_mode: str, primary_pattern: str, secondary_entity: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode=primary_mode, pattern=primary_pattern, entity="robot"),
        secondary=ContactMatch(mode="body", pattern=secondary_entity, entity=secondary_entity),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


def make_microduck_flip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_long_jump_env_cfg(play=play)
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.scene.entities.update(flip_stage.make_flip_stage_entity_cfgs())
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 3.0)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 1.6

    # "feet" = soles + the foot shells / ankle brackets on the ankle bodies:
    # on the soft mat the sole sinks a few cm and the shells touch, which the
    # v1-v4 sensors counted as a body crash (f8 render: an upright standing
    # landing terminated as `fell` every 0.1 s).
    feet = r"^(left_foot_collision|right_foot_collision|ankle_(left|right)_.*_collision)$"
    body = r"^(?!(left|right)_foot_collision$)(?!ankle_(left|right)_).*_collision$"
    floor = ContactMatch(mode="body", pattern="terrain")
    sensors = [
        _contact(fl.FEET_MAT, "geom", feet, flip_stage.MAT_ENTITY),
        _contact(fl.FEET_PLATFORM, "geom", feet, flip_stage.PLATFORM_ENTITY),
        _contact(fl.BODY_MAT, "geom", body, flip_stage.MAT_ENTITY),
        _contact(fl.BODY_PLATFORM, "geom", body, flip_stage.PLATFORM_ENTITY),
        ContactSensorCfg(name=fl.BODY_FLOOR, primary=ContactMatch(mode="geom", pattern=body, entity="robot"),
                         secondary=floor, fields=("found",), reduce="none", num_slots=1),
        _contact(fl.HEAD_ANY[0], "body", "jaw_soft", flip_stage.MAT_ENTITY),
        _contact(fl.HEAD_ANY[1], "body", "jaw_soft", flip_stage.PLATFORM_ENTITY),
        ContactSensorCfg(name=fl.HEAD_ANY[2], primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
                         secondary=floor, fields=("found",), reduce="none", num_slots=1),
    ]
    keep = [s for s in cfg.scene.sensors if s.name in ("self_collision", "feet_ground_contact")]
    cfg.scene.sensors = tuple(sensors + keep)

    # ── rewards ──────────────────────────────────────────────────────────────
    for name in list(cfg.rewards):
        if name.startswith("long_jump") or name in ("upright", "fall_cost"):
            cfg.rewards.pop(name)
    cfg.rewards["flip_progress"] = RewardTermCfg(func=fl.flip_progress, weight=PROGRESS_WEIGHT)
    cfg.rewards["flip_stand"] = RewardTermCfg(
        func=fl.flip_stand, weight=STAND_WEIGHT,
        params={"target_height": STAND_Z, "height_std": 0.04, "upright_std": 0.40, "pose_std": 0.40, "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["flip_stand_tax"] = RewardTermCfg(func=fl.flip_stand_tax, weight=STAND_TAX_WEIGHT, params={"target_height": STAND_Z})
    cfg.rewards["flip_fall_cost"] = RewardTermCfg(func=fl.flip_fall_cost, weight=FALL_COST_WEIGHT)
    cfg.rewards["flip_body_contact"] = RewardTermCfg(func=fl.flip_body_contact_penalty, weight=BODY_CONTACT_WEIGHT)
    cfg.rewards["flip_dawdle"] = RewardTermCfg(func=fl.flip_dawdle_penalty, weight=DAWDLE_WEIGHT)
    cfg.rewards["flip_lateral"] = RewardTermCfg(func=fl.flip_lateral_penalty, weight=LATERAL_WEIGHT)
    cfg.rewards["flip_launch"] = RewardTermCfg(func=fl.flip_launch, weight=LAUNCH_WEIGHT, params={"vz_cap": 0.6})
    cfg.rewards["flip_takeoff"] = RewardTermCfg(func=fl.flip_takeoff_bonus, weight=TAKEOFF_WEIGHT)
    cfg.rewards["flip_tuck"] = RewardTermCfg(func=fl.flip_tuck_reward, weight=TUCK_WEIGHT)
    # SELF-NEGATING → POSITIVE weight
    cfg.rewards["flip_hop"] = RewardTermCfg(func=fl.flip_hop_penalty, weight=HOP_WEIGHT)
    cfg.rewards["flip_hops"] = RewardTermCfg(func=fl.flip_hops, weight=1.0)   # diagnostic
    # diagnostics at weight 1 (Episode_Reward = episode sum / 175 steps; at
    # 0.01 they logged as 0.0000): flip_jumped x175 = jump-like takeoff
    # fraction, flip_landed x175 = feet-landing fraction, flip_rotation x175 =
    # rad per episode.  +1 one-shots are negligible next to progress (8/rad).
    cfg.rewards["flip_jumped"] = RewardTermCfg(func=fl.flip_takeoff_flag, weight=1.0)
    cfg.rewards["flip_rotation"] = RewardTermCfg(func=fl.flip_max_rotation, weight=1.0)
    cfg.rewards["flip_landed"] = RewardTermCfg(func=fl.flip_landed_flag, weight=1.0)
    # a flip IS a large angular-velocity, large-impact event: keep blockers ~0
    cfg.rewards["body_ang_vel"].weight = -0.001
    cfg.rewards["angular_momentum"].weight = -0.0005
    cfg.rewards["gentle_landing"].weight = 0.002

    # ── observations: drop height + direction in the body-command slot ──────
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=fl.flip_command_obs, params={"dim": 6})

    # ── terminations ─────────────────────────────────────────────────────────
    cfg.terminations["fell"] = TerminationTermCfg(func=fl.flip_fell, time_out=False)
    if HOP_FATAL:
        # the hop is made impossible rather than expensive - see fl.flip_hopped
        cfg.terminations["hopped"] = TerminationTermCfg(func=fl.flip_hopped, time_out=False)

    # ── events: stage + spawn ────────────────────────────────────────────────
    cfg.events.pop("set_long_jump_state", None)
    cfg.events["set_flip_state"] = EventTermCfg(
        func=fl.reset_flip_state, mode="reset",
        params={
            "direction": FLIP_DIR,
            "height_range": (H_MIN, H_MAX),
            "midflip_prob": 0.0 if play else MIDFLIP_PROB,
            "tuck_overrides": TUCK_OVERRIDES,
        },
    )
    if "reset_base" in cfg.events:
        # the flip reset writes the root itself; keep the base reset tiny so it
        # cannot put the robot off the platform before ours runs
        cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}

    # ── curriculum ───────────────────────────────────────────────────────────
    cfg.curriculum.pop("long_jump_spawn_mix", None)
    # In PLAY the spawn mix must not be reinstalled.  The event above sets
    # midflip_prob to 0 for play, but this curriculum's step-0 stage sets it
    # straight back to MIDFLIP_PROB - so every play rollout (renders, probes,
    # batteries) was quietly spawning about a quarter of its environments
    # already in mid-air.  Measured 2026-09-18: a hop-value probe found 27.3 %
    # of "flips" had takeoff vz exactly 0.000 and jump_ok preset true, which is
    # the mid-flip spawn signature, not a flip.
    if not play:
        cfg.curriculum["flip_spawn_mix"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            # v2: mid-flip spawns start as near-upright drops from a few cm above
            # the mat (feet landings by chance -> the landing annuity is found),
            # then widen toward genuine mid-flip states.
            params={"event_name": "set_flip_state", "param_stages": [
                # f6 render (iteration 1000): spawns pitched 30-60 deg nose-up with
                # tucked legs land on the back; the few upright ones stand.  Start
                # UPRIGHT with the legs down, widen slowly.
                {"step": 0, "params": {"midflip_prob": MIDFLIP_PROB, "midflip_angle_range": (math.radians(350.0), math.radians(360.0)),
                                       "midflip_omega_range": (0.0, 2.0), "midflip_u_range": (0.05, 0.12), "tuck_factor_range": (0.0, 0.2),
                                       "midflip_vz_scale": 0.15}},
                {"step": max(0, 1500 - CURRICULUM_SHIFT_ITERS) * 24, "params": {"midflip_angle_range": (math.radians(335.0), math.radians(360.0)),
                                               "midflip_omega_range": (1.0, 5.0), "midflip_u_range": (0.08, 0.2), "tuck_factor_range": (0.1, 0.4),
                                               "midflip_vz_scale": 0.3}},
                {"step": max(0, 3000 - CURRICULUM_SHIFT_ITERS) * 24, "params": {"midflip_angle_range": (math.radians(300.0), math.radians(355.0)),
                                               "midflip_omega_range": (4.0, 9.0), "midflip_u_range": (0.15, 0.5), "tuck_factor_range": (0.3, 0.8),
                                               "midflip_vz_scale": 0.5}},
                {"step": max(0, 4500 - CURRICULUM_SHIFT_ITERS) * 24, "params": {"midflip_angle_range": (math.radians(200.0), math.radians(350.0)),
                                               "midflip_omega_range": (6.0, 12.0), "midflip_u_range": (0.2, 0.7), "tuck_factor_range": (0.5, 1.0),
                                               "midflip_vz_scale": 0.75}},
                {"step": max(0, 6000 - CURRICULUM_SHIFT_ITERS) * 24, "params": {"midflip_prob": 0.6 * MIDFLIP_PROB, "midflip_angle_range": (math.radians(60.0), math.radians(340.0)),
                                               "midflip_omega_range": (8.0, 16.0), "midflip_u_range": (0.35, 0.9), "midflip_vz_scale": 1.0}},
            ]},
        )
    cfg.curriculum["gentle_landing_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.002}, {"step": max(0, 3000 - CURRICULUM_SHIFT_ITERS) * 24, "weight": 0.005}]
    return cfg


MicroduckFlipRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": INIT_STD, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=ENTROPY_COEF,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=SYMMETRY_CFG,
    ),
    wandb_project="mjlab_microduck", experiment_name="flip", run_name="flip",
    save_interval=250, num_steps_per_env=24, max_iterations=10000,
)

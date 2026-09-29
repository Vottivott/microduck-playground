"""Microduck platform jump: a two-footed long jump DOWN from a raised block
onto a lower, separate block across a pit ("kinda parkour like", user sketch
2026-09-18).

Built on the standing long jump recipe (collision-covered robot, bounded
joint commands, BAM actuators, full DR, 61-D obs, v11 graded two-foot
rules) with the floor replaced by two mocap blocks (``platform_stage``):
block A (start, top at 0.10 m + drop) and block B (target, top at 0.10 m)
separated by a gap.  Feet on the floor = fell into the pit (termination).
Gap and drop are per-env curriculum variables and are given to the policy
through the body-command observation slot (no vision).

Env knobs: MICRODUCK_PJ_{PROGRESS_W,BONUS_W,UPRIGHT_W,DAWDLE_W,ENTROPY,
INIT_STD,POSE_STD,DIST_FULL,FLIGHT_PROB,CURRICULUM_SHIFT}.
"""
from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot import platform_stage as ps
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import platform_jump_mdp as pj
from mjlab_microduck.tasks.microduck_long_jump_env_cfg import (
    CROUCH_OVERRIDES,
    STAND_Z,
    _LEG_JOINTS,
    make_microduck_long_jump_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

PROGRESS_WEIGHT = float(os.getenv("MICRODUCK_PJ_PROGRESS_W", "600"))
LANDING_BONUS_WEIGHT = float(os.getenv("MICRODUCK_PJ_BONUS_W", "400"))
STAND_WEIGHT = 1.5
STAND_TAX_WEIGHT = 5.0
BODY_CONTACT_WEIGHT = 0.5
LATERAL_WEIGHT = 0.5
UPRIGHT_WEIGHT = float(os.getenv("MICRODUCK_PJ_UPRIGHT_W", "0.3"))
FALL_COST_WEIGHT = float(os.getenv("MICRODUCK_PJ_FALL_COST_W", "20.0"))
PROGRESS_CAP_MARGIN = float(os.getenv("MICRODUCK_PJ_PROGRESS_CAP", "0.15"))  # pay flight distance only up to gap + this
DAWDLE_WEIGHT = float(os.getenv("MICRODUCK_PJ_DAWDLE_W", "0.3"))
LAUNCH_WEIGHT = 60.0
TAKEOFF_WEIGHT = 30.0
ENTROPY_COEF = float(os.getenv("MICRODUCK_PJ_ENTROPY", "0.005"))
INIT_STD = float(os.getenv("MICRODUCK_PJ_INIT_STD", "0.8"))
POSE_STD = float(os.getenv("MICRODUCK_PJ_POSE_STD", "0.25"))
DIST_FULL = float(os.getenv("MICRODUCK_PJ_DIST_FULL", "0.12"))
FLIGHT_SPAWN_PROB = float(os.getenv("MICRODUCK_PJ_FLIGHT_PROB", "0.30"))
# the curriculum keys off the process step counter (flip v7 lesson): shift
# the stage boundaries earlier by this many iterations on a resume
CURRICULUM_SHIFT_ITERS = int(os.getenv("MICRODUCK_PJ_CURRICULUM_SHIFT", "0"))
# Pin the gap / drop for renders and probe batteries: "0.2" or "0.15,0.25"
# (metres).  When set, the value replaces the spawn event's range AND every
# curriculum stage's, so the curriculum cannot widen it back.
GAP_OVERRIDE = os.getenv("MICRODUCK_PJ_GAP")
DROP_OVERRIDE = os.getenv("MICRODUCK_PJ_DROP")
EPISODE_LENGTH_S = 4.0

# gap / drop stages (m); each stage is reached at its iteration count
GAP_DROP_STAGES = [
    (0, (0.00, 0.05), (0.03, 0.08)),
    (1000, (0.00, 0.12), (0.05, 0.15)),
    (2000, (0.05, 0.20), (0.08, 0.25)),
    (3500, (0.10, 0.30), (0.10, 0.30)),
    (5000, (0.15, 0.40), (0.15, 0.35)),
]

FEET_GEOMS = r"^(left_foot_collision|right_foot_collision)$"
BODY_GEOMS = r"^(?!(left|right)_foot_collision$)(?!ankle_(left|right)_).*_collision$"


def _contact(name: str, primary_mode: str, primary_pattern: str, secondary_entity: str | None,
             reduce: str = "none") -> ContactSensorCfg:
    if secondary_entity is None:
        secondary = ContactMatch(mode="body", pattern="terrain")
    else:
        secondary = ContactMatch(mode="body", pattern=secondary_entity, entity=secondary_entity)
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(mode=primary_mode, pattern=primary_pattern, entity="robot"),
        secondary=secondary,
        fields=("found",),
        reduce=reduce,
        num_slots=1,
    )


def _stage_step(iters: int) -> int:
    return max(0, iters - CURRICULUM_SHIFT_ITERS) * 24


def _range_override(spec: str | None) -> tuple[float, float] | None:
    if not spec:
        return None
    parts = [float(v) for v in spec.split(",")]
    return (parts[0], parts[-1])


def make_microduck_platform_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_long_jump_env_cfg(play=play)
    # Whole-body hulls can exceed the velocity template's 35-contact budget.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
    cfg.scene.entities.update(ps.make_platform_stage_entity_cfgs())
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 3.0)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 1.4

    sensors = [
        _contact(pj.FEET_A, "geom", FEET_GEOMS, ps.PLATFORM_A),
        _contact(pj.FEET_B, "geom", FEET_GEOMS, ps.PLATFORM_B),
        _contact(pj.FEET_FLOOR, "geom", FEET_GEOMS, None),
        _contact(pj.BODY_A, "geom", BODY_GEOMS, ps.PLATFORM_A),
        _contact(pj.BODY_B, "geom", BODY_GEOMS, ps.PLATFORM_B),
        _contact(pj.BODY_FLOOR, "geom", BODY_GEOMS, None),
        _contact(pj.HEAD_ANY[0], "body", "jaw_soft", ps.PLATFORM_A),
        _contact(pj.HEAD_ANY[1], "body", "jaw_soft", ps.PLATFORM_B),
        _contact(pj.HEAD_ANY[2], "body", "jaw_soft", None),
    ]
    # feet_ground_contact stays: the velocity base's critic foot_air_time term reads it
    keep = [s for s in cfg.scene.sensors if s.name in ("self_collision", "feet_ground_contact")]
    cfg.scene.sensors = tuple(sensors + keep)

    # ── rewards ──────────────────────────────────────────────────────────────
    for name in list(cfg.rewards):
        if name.startswith("long_jump") or name in ("fall_cost",):
            cfg.rewards.pop(name)
    cfg.rewards["upright"].weight = UPRIGHT_WEIGHT
    target_height = STAND_Z + ps.B_TOP_M          # trunk above the env origin when standing on B
    cfg.rewards["pj_fall_cost"] = RewardTermCfg(func=pj.pj_fall_cost, weight=FALL_COST_WEIGHT, params={"max_tilt_deg": 70.0})
    cfg.rewards["pj_progress"] = RewardTermCfg(func=pj.pj_progress, weight=PROGRESS_WEIGHT, params={"cap_margin": PROGRESS_CAP_MARGIN})
    cfg.rewards["pj_landing"] = RewardTermCfg(func=pj.pj_landing_bonus, weight=LANDING_BONUS_WEIGHT, params={"max_tilt_deg": 30.0})
    cfg.rewards["pj_stand"] = RewardTermCfg(
        func=pj.pj_stand, weight=STAND_WEIGHT,
        params={"target_height": target_height, "height_std": 0.04, "upright_std": 0.40, "pose_std": POSE_STD,
                "joint_indices": _LEG_JOINTS, "dist_zero": 0.03, "dist_full": DIST_FULL},
    )
    cfg.rewards["pj_stand_tax"] = RewardTermCfg(func=pj.pj_stand_tax, weight=STAND_TAX_WEIGHT, params={"target_height": target_height})
    cfg.rewards["pj_body_contact"] = RewardTermCfg(func=pj.pj_body_contact_penalty, weight=BODY_CONTACT_WEIGHT)
    cfg.rewards["pj_lateral"] = RewardTermCfg(func=pj.pj_lateral_penalty, weight=LATERAL_WEIGHT, params={"yaw_weight": 0.3})
    cfg.rewards["pj_launch"] = RewardTermCfg(func=pj.pj_launch, weight=LAUNCH_WEIGHT, params={"vz_cap": 0.6})
    cfg.rewards["pj_takeoff"] = RewardTermCfg(func=pj.pj_takeoff_bonus, weight=TAKEOFF_WEIGHT, params={"max_tilt_deg": 45.0})
    cfg.rewards["pj_dawdle"] = RewardTermCfg(func=pj.pj_dawdle_penalty, weight=DAWDLE_WEIGHT, params={"after_s": 1.0})
    # diagnostics at weight 1 (Episode_Reward = episode sum / 200 steps)
    cfg.rewards["pj_distance"] = RewardTermCfg(func=pj.pj_landed_distance, weight=1.0)
    cfg.rewards["pj_landed"] = RewardTermCfg(func=pj.pj_landed_flag, weight=1.0)
    cfg.rewards["pj_sync"] = RewardTermCfg(func=pj.pj_sync, weight=1.0)
    cfg.rewards["pj_gap"] = RewardTermCfg(func=pj.pj_gap_obs, weight=1.0)
    cfg.rewards["pj_stepped"] = RewardTermCfg(func=pj.pj_stepped, weight=-0.01)
    cfg.rewards["pj_pit"] = RewardTermCfg(func=pj.pj_pit, weight=-0.01)

    # ── observations: gap + drop in the body-command slot ───────────────────
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=pj.pj_command_obs, params={"dim": 6})

    # ── terminations ─────────────────────────────────────────────────────────
    cfg.terminations["fell"] = TerminationTermCfg(func=pj.pj_fell, params={"max_tilt_deg": 70.0}, time_out=False)

    # ── events: stage + spawn ────────────────────────────────────────────────
    cfg.events.pop("set_long_jump_state", None)
    s0 = GAP_DROP_STAGES[0]
    cfg.events["set_platform_jump_state"] = EventTermCfg(
        func=pj.reset_platform_jump_state, mode="reset",
        params={
            "gap_range": s0[1], "drop_range": s0[2],
            "standing_prob": 0.85 - FLIGHT_SPAWN_PROB, "crouch_prob": 0.15, "flight_prob": FLIGHT_SPAWN_PROB,
            "crouch_overrides": CROUCH_OVERRIDES,
            "flight_vx_range": (0.2, 0.5), "flight_vz_range": (-0.3, 0.0), "flight_z_range": (0.12, 0.16),
        },
    )
    if play:
        cfg.events["set_platform_jump_state"].params.update({"standing_prob": 1.0, "crouch_prob": 0.0, "flight_prob": 0.0})

    # ── curriculum ───────────────────────────────────────────────────────────
    cfg.curriculum.pop("long_jump_spawn_mix", None)
    stages = []
    for i, (iters, gap, drop) in enumerate(GAP_DROP_STAGES):
        p = {"gap_range": gap, "drop_range": drop}
        if i == 1:
            p.update({"flight_vx_range": (0.2, 0.7), "flight_vz_range": (-0.3, 0.3), "flight_z_range": (0.12, 0.18)})
        if i >= 2:
            p.update({"flight_vx_range": (0.3, 0.9), "flight_vz_range": (0.0, 0.6), "flight_z_range": (0.13, 0.22)})
        if i >= 3:
            p.update({"standing_prob": 0.95 - 0.5 * FLIGHT_SPAWN_PROB, "crouch_prob": 0.05, "flight_prob": 0.5 * FLIGHT_SPAWN_PROB})
        stages.append({"step": _stage_step(iters), "params": p})
    cfg.curriculum["platform_jump_stages"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={"event_name": "set_platform_jump_state", "param_stages": stages},
    )
    gap_pin, drop_pin = _range_override(GAP_OVERRIDE), _range_override(DROP_OVERRIDE)
    if gap_pin or drop_pin:
        pin = {}
        if gap_pin:
            pin["gap_range"] = gap_pin
        if drop_pin:
            pin["drop_range"] = drop_pin
        cfg.events["set_platform_jump_state"].params.update(pin)
        for stage in stages:
            stage["params"].update(pin)
    for name in ("action_rate_weight", "torque_rate_weight", "gentle_landing_weight"):
        for stage in cfg.curriculum[name].params["weight_stages"]:
            stage["step"] = max(0, stage["step"] - CURRICULUM_SHIFT_ITERS * 24)
    return cfg


MicroduckPlatformJumpRlCfg = RslRlOnPolicyRunnerCfg(
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
        symmetry_cfg=SYMMETRY_CFG,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="platform_jump",
    run_name="platform_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)

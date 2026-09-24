"""Microduck braced corridor climbing: wedge between two close walls and work
upward (user request 2026-09-18, "wall jump from side to side to get up a
narrow vertical corridor").

Jumping up is ruled out by measurement - the servos give 0.04-0.26 m/s of
vertical launch, so each bounce would have to beat gravity on its own - but
the bracing probe found that the duck WEDGES between walls 12-14 cm apart
with a quarter of its servo torque, holds on surfaces as slick as friction
0.3, and creeps upward while doing it.  So the task is the STEP, not the
hold: alternating press and lift to gain height repeatedly.

Built on the long-jump base (collision-covered robot, bounded joint
commands, BAM actuators, full DR, 61-D obs).  Env knobs:
MICRODUCK_CH_{WIDTH,PROGRESS_W,BRACE_W,SLIP_W,ENTROPY,INIT_STD,SPAWN_Z_MAX}.
"""
from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from dataclasses import replace

from mjlab.entity import EntityCfg

from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import chimney_mdp as ch
from mjlab_microduck.tasks.microduck_long_jump_env_cfg import make_microduck_long_jump_env_cfg
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# the brace the probe found: hips at -90 deg (legs straight across the gap),
# knees neutral, ankles neutral, trunk pitched slightly back into the wall
BRACE_OVERRIDES = {2: -1.57, 3: 0.0, 4: 0.0, 11: 1.57, 12: 0.0, 13: 0.0}

# ... and the same pose as the robot's DEFAULT, which is what the action offset
# is measured from.  With the HOME default (legs down) a zero action drives the
# legs out of the wedge within a few control steps, so run w2 fell out of every
# brace in ~0.2 s and never gained height.  Making the brace the neutral action
# means holding still holds the wedge, and learning is about the STEP.
BRACE_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        **HOME_FRAME.joint_pos,
        r".*left_hip_pitch.*": -1.57,
        r".*right_hip_pitch.*": 1.57,
        r".*left_knee.*": 0.0,
        r".*right_knee.*": 0.0,
        r".*left_ankle.*": 0.0,
        r".*right_ankle.*": 0.0,
    },
)

PROGRESS_WEIGHT = float(os.getenv("MICRODUCK_CH_PROGRESS_W", "800"))   # x metres climbed
# The brace is a foothold, not the goal, and the first version priced it as if
# it were: 0.5 per step over a 300-step episode paid 150 for hanging still,
# while climbing 5 cm paid 40.  Holding beat climbing by a factor of four.
BRACE_WEIGHT = float(os.getenv("MICRODUCK_CH_BRACE_W", "0.05"))
STALL_WEIGHT = float(os.getenv("MICRODUCK_CH_STALL_W", "0.5"))   # per step without a new high point
SLIP_WEIGHT = float(os.getenv("MICRODUCK_CH_SLIP_W", "0.3"))
FALL_COST_WEIGHT = float(os.getenv("MICRODUCK_CH_FALL_COST_W", "10.0"))
HEAD_WEIGHT = float(os.getenv("MICRODUCK_CH_HEAD_W", "0.2"))   # head on a wall: priced, not fatal
ENTROPY_COEF = float(os.getenv("MICRODUCK_CH_ENTROPY", "0.004"))
INIT_STD = float(os.getenv("MICRODUCK_CH_INIT_STD", "0.6"))
SPAWN_Z_MAX = float(os.getenv("MICRODUCK_CH_SPAWN_Z_MAX", "1.30"))
# The bottom of the climb is its own skill.  Every run through w9 spawned the
# robot already wedged 25 cm to 1.3 m up, so entering the corridor from the
# floor has never been trained.
GROUND_PROB = float(os.getenv("MICRODUCK_CH_GROUND_PROB", "0.0"))
# Of the ground starts, the fraction that stand UPRIGHT mid-corridor instead of
# already braced.  That is the state a separate approach policy would hand over
# (user 2026-09-19), and a policy switch only works if the handover state is
# inside the receiver's training distribution.
UPRIGHT_PROB = float(os.getenv("MICRODUCK_CH_UPRIGHT_PROB", "0.0"))
# How far the handover may be from perfect.  An approach policy does not deliver
# a square, centred, motionless duck, and a climb trained only on the perfect
# version falls over when handed the real one.
UPRIGHT_YAW = float(os.getenv("MICRODUCK_CH_UPRIGHT_YAW", "0.0"))
UPRIGHT_X = float(os.getenv("MICRODUCK_CH_UPRIGHT_X", "0.0"))
UPRIGHT_JVEL = float(os.getenv("MICRODUCK_CH_UPRIGHT_JVEL", "0.0"))
# A bank of REAL handover states captured off the approach policy, and how often
# to spawn from it.  Synthetic jitter around the perfect pose did not transfer.
HANDOVER_BANK = os.getenv("MICRODUCK_CH_HANDOVER_BANK", "")
HANDOVER_PROB = float(os.getenv("MICRODUCK_CH_HANDOVER_PROB", "0.0"))
# The STANDING angles for the joints the brace overrides.  These must be given
# explicitly: the robot's default pose IS the brace, so "leave the joints alone"
# yields a braced spawn, not a standing one.
UPRIGHT_OVERRIDES = {2: -0.4579, 3: -0.0049, 4: 0.4529, 11: 0.4579, 12: 0.0049, 13: -0.4529}
# Wall friction.  The bracing measurement found the wedge is GEOMETRIC - a
# 12 cm gap holds at friction 0.3, while 14 cm needs 0.8 - so how slippery the
# wall may be is the sim2real question for this task.  The walls are built at
# 0.9 by default. Historical w9 used 0.4–1.1; the recovered w16 launcher
# used 0.5–1.1. Set the range explicitly for continuation (see resume.json).
WALL_FRICTION = tuple(float(v) for v in os.getenv("MICRODUCK_CH_WALL_FRIC", "0.9,0.9").split(","))
WIDTH_PIN = os.getenv("MICRODUCK_CH_WIDTH")     # pin the corridor for renders and probes
EPISODE_LENGTH_S = float(os.getenv("MICRODUCK_CH_EPISODE_S", "6.0"))


def make_microduck_chimney_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_long_jump_env_cfg(play=play)
    cfg.scene.entities["robot"] = replace(cfg.scene.entities["robot"], init_state=BRACE_FRAME)
    cfg.scene.entities.update(cs.make_corridor_entity_cfgs())
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 3.0)
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.viewer.distance = 1.2

    # contacts come from the raw scan in chimney_mdp; keep only what the base needs
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name in ("self_collision", "feet_ground_contact"))

    # ── rewards ──────────────────────────────────────────────────────────────
    for name in list(cfg.rewards):
        if name.startswith("long_jump") or name in ("fall_cost", "upright"):
            cfg.rewards.pop(name)
    cfg.rewards["ch_progress"] = RewardTermCfg(func=ch.climb_progress, weight=PROGRESS_WEIGHT)
    cfg.rewards["ch_brace"] = RewardTermCfg(func=ch.climb_brace, weight=BRACE_WEIGHT,
                                            params={"max_tilt_deg": ch.BRACE_TILT_MAX_DEG})
    # SELF-NEGATING → POSITIVE weights (mdp sign convention)
    cfg.rewards["ch_slip"] = RewardTermCfg(func=ch.climb_slip_penalty, weight=SLIP_WEIGHT)
    cfg.rewards["ch_fall_cost"] = RewardTermCfg(func=ch.climb_fall_cost, weight=FALL_COST_WEIGHT)
    cfg.rewards["ch_head"] = RewardTermCfg(func=ch.climb_head_penalty, weight=HEAD_WEIGHT)
    cfg.rewards["ch_stall"] = RewardTermCfg(func=ch.climb_stall_penalty, weight=STALL_WEIGHT,
                                            params={"stall_after_s": float(os.getenv("MICRODUCK_CH_STALL_AFTER", "0.5"))})
    # diagnostic at weight 1 (Episode_Reward = episode sum / max steps)
    cfg.rewards["ch_height"] = RewardTermCfg(func=ch.climb_height, weight=1.0)

    # ── observations: the corridor width, one measured constant ─────────────
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=ch.chimney_command_obs, params={"dim": 6},
        )

    # ── terminations ─────────────────────────────────────────────────────────
    cfg.terminations["fell"] = TerminationTermCfg(func=ch.climb_fell, time_out=False)

    # ── events ───────────────────────────────────────────────────────────────
    cfg.events.pop("set_long_jump_state", None)
    width = (float(WIDTH_PIN), float(WIDTH_PIN)) if WIDTH_PIN else (cs.WIDTH_MIN_M, cs.WIDTH_MAX_M)
    cfg.events["set_chimney_state"] = EventTermCfg(
        func=ch.reset_chimney_state, mode="reset",
        params={
            "width_range": width,
            # spawn anywhere up the corridor: without mid-corridor starts only
            # the bottom of the climb would ever be practised
            "height_range": (0.25, 0.45) if play else (0.25, SPAWN_Z_MAX),
            "brace_overrides": BRACE_OVERRIDES,
            "ground_prob": GROUND_PROB,
            "upright_prob": UPRIGHT_PROB,
            "upright_overrides": UPRIGHT_OVERRIDES,
            "upright_yaw_range": (0.0, UPRIGHT_YAW),
            "upright_x_range": (0.0, UPRIGHT_X),
            "upright_jvel_std": UPRIGHT_JVEL,
            "handover_bank": HANDOVER_BANK,
            "handover_prob": HANDOVER_PROB,
        },
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.1)
    for wall in (cs.WALL_LEFT, cs.WALL_RIGHT):
        cfg.events[f"{wall}_friction"] = EventTermCfg(
            mode="startup", func=dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg(wall, geom_names=(f"{wall}_collision",)),
                    "operation": "abs", "ranges": WALL_FRICTION, "shared_random": True},
        )

    # Contact buffer.  The default 35 OVERFLOWS in this task - the logs read
    # "broadphase overflow - please increase nconmax to 46" - and an overflow
    # silently DROPS contacts, which is exactly the wall support a braced duck
    # depends on.  The sitstand recipe hit the same wall ("legs + head all in
    # close ground/self contact") and uses 200; so do the tug and stilt envs.
    cfg.sim.nconmax = 200
    # nconmax was raised here long ago; contact_sensor_maxmatch was NOT, and it
    # defaults to 64.  a12 logged "increase Option.contact_sensor_maxmatch to
    # 81" (2026-09-21), i.e. contact SENSOR matches were being dropped in this
    # task exactly the way broadphase contacts once were - silently, and with
    # the rewards that read contact quietly going wrong.
    cfg.sim.contact_sensor_maxmatch = 500

    cfg.curriculum.pop("long_jump_spawn_mix", None)
    return cfg


MicroduckChimneyRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="chimney",
    run_name="chimney",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8000,
)

"""Basketball circus act: a Microduck balancing on top of a basketball and
walking it around on command (feature/basketball, 2026-09-06).

Built on the velocity recipe (DR / obs noise / delays come for free):

* Scene: the full-collision graphite duck + one "ball" entity (a 0.24 m,
  0.62 kg rubber sphere on a free joint; friction priority over the floor).
* Spawn (``reset_basketball``): ball resting at the env origin, duck placed
  standing on the apex (root at 2R + 0.125 m), random yaw, small noise.
* Obs contract (61-D): twist command as usual, head_command slot zero-padded,
  body_command slot is zero for the released blind actor; the privileged
  critic retains ``basketball_state``. ``blind=False`` enables the old reference.
* Rewards: velocity tracking (rolling the ball = walking on it), upright,
  root over the ball centre, feet resting on the ball, root height above the
  ball, plus the recipe's regularizers; ball speed penalty at small weight.
* Termination: off the ball (low / off-centre / tilted).
* Difficulty axis: ``BallHoldAction`` applies a per-env spring/damper wrench
  on the ball (hold 1 = pinned like a weld, 0 = free); the hold curriculum
  promotes an env when it survives ``promote_len_s`` and demotes quick falls.
  A weld equality cannot be used: mjwarp equalities are shared by all worlds.

Env knobs: MICRODUCK_BB_HOLD_LEVELS="1,0.3,0.1,0.03,0" (curriculum table),
MICRODUCK_BB_PLAY_HOLD (hold level for play/eval, default 0 = free ball),
MICRODUCK_BB_SIM_DT (physics step, control stays 50 Hz).
"""
from __future__ import annotations

import math
import os
from copy import deepcopy

from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.robot.basketball import basketball_robot_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Size-7 basketball: 0.24 m, 0.62 kg.  MICRODUCK_BB_BALL_RADIUS for other balls.
BALL_RADIUS = float(os.getenv("MICRODUCK_BB_BALL_RADIUS", "0.12"))
BALL_MASS = 0.62
BALL_FRICTION = (1.2, 0.01, 0.001)  # sliding, rolling, torsional (rubber on a hall floor)
BALL_RGBA = (0.91, 0.42, 0.12, 1.0)
ROOT_HEIGHT = 0.125  # duck root above the sole plane when standing (velocity spawn z)
EPISODE_LENGTH_S = 10.0
FALL_TILT_DEG = 55.0
HOLD_LEVELS = tuple(float(v) for v in os.getenv("MICRODUCK_BB_HOLD_LEVELS", "1,0.5,0.25,0.1,0.03,0").split(","))
PLAY_HOLD = float(os.getenv("MICRODUCK_BB_PLAY_HOLD", "0"))
BB_SIM_DT = float(os.getenv("MICRODUCK_BB_SIM_DT")) if os.getenv("MICRODUCK_BB_SIM_DT") else None
PUSH_SCALE = float(os.getenv("MICRODUCK_BB_PUSH_SCALE", "0.3"))
CMD_SCALE = float(os.getenv("MICRODUCK_BB_CMD_SCALE", "1"))
BLIND = os.getenv("MICRODUCK_BB_BLIND", "1") == "1"
HISTORY = int(os.getenv("MICRODUCK_BB_HISTORY", "1"))  # b5: frames of proprioception history for the actor  # b3: actor sees no ball state (deployable contract), critic still does  # b2: 2 -> +-0.3 m/s forward, +-1 rad/s yaw (walking on the ball)
# b9: optional fixed smoothness cost for the free-ball PPO continuation.
# None preserves the walking recipe's inherited ramp for existing experiments.
ACTION_RATE_WEIGHT = os.getenv("MICRODUCK_BB_ACTION_RATE_WEIGHT")

ENABLE_VELOCITY_TRACKING = True
TRACK_LIN_WEIGHT = 1.0
TRACK_ANG_WEIGHT = 0.5
CENTERED_WEIGHT = 2.0
FEET_ON_BALL_WEIGHT = 1.0
HEIGHT_WEIGHT = 1.0
UPRIGHT_WEIGHT = 1.0
BALL_SPEED_WEIGHT = -0.05


_ASSETS = Path(__file__).resolve().parents[1] / "robot" / "assets" / "basketball"
BALL_VISUAL_RADIUS = 0.1207  # +0.7 mm over the collision sphere: hides the polygon/sole-hull gap under the feet


def _ball_spec():
    """Collision: an invisible sphere (radius BALL_RADIUS).  Look: a UV-mapped
    sphere mesh with pebble displacement and the real 8-panel seam texture
    (scripts/make_basketball_assets.py; user request 2026-09-06)."""
    spec = mujoco.MjSpec()
    spec.modelname = "ball"
    tex = spec.add_texture()
    tex.name = "basketball_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.file = str(_ASSETS / "basketball.png")
    mat = spec.add_material()
    mat.name = "basketball_mat"
    textures = list(mat.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "basketball_tex"
    mat.textures = textures
    mat.specular = 0.15
    mat.shininess = 0.15
    mat.reflectance = 0.0
    mesh = spec.add_mesh()
    mesh.name = "basketball_mesh"
    mesh.file = str(_ASSETS / "basketball.obj")
    body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, BALL_RADIUS])
    body.add_freejoint()
    g = body.add_geom(name="ball_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE)
    g.size = [BALL_RADIUS, 0.0, 0.0]
    g.mass = BALL_MASS
    g.rgba = [1.0, 1.0, 1.0, 0.0]  # invisible: the mesh below carries the look
    g.group = 3
    g.friction = list(BALL_FRICTION)
    g.priority = 1  # the ball's friction wins over the floor's / the feet's
    g.condim = 4  # torsional friction: a spinning ball under the feet is not free
    g.contype = 1
    g.conaffinity = 1
    v = body.add_geom(name="ball_visual", type=mujoco.mjtGeom.mjGEOM_MESH)
    v.meshname = "basketball_mesh"
    v.material = "basketball_mat"
    v.contype = 0
    v.conaffinity = 0
    v.mass = 0.0
    v.group = 0
    body.add_site(name="ball_center", pos=[0.0, 0.0, 0.0], size=[0.004, 0.004, 0.004], group=3)
    return spec


def make_microduck_basketball_env_cfg(
    play: bool = False, *, blind: bool | None = None, history: int | None = None,
) -> ManagerBasedRlEnvCfg:
    blind = BLIND if blind is None else blind
    history = HISTORY if history is None else history
    if history < 1:
        raise ValueError("Basketball history must contain at least one frame")
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)
    from mjlab_microduck.robot.microduck_constants import get_standup_spec

    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    robot_cfg.spec_fn = get_standup_spec  # full-collision model: body vs ball contacts on a fall
    cfg.scene.entities = {
        "robot": basketball_robot_cfg(robot_cfg),  # user (2026-09-06): cream duck for the basketball act
        "ball": EntityCfg(spec_fn=_ball_spec),
    }
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 2.5)
    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 1.6
    cfg.viewer.max_extra_envs = 0
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.sim.nconmax = max(getattr(cfg.sim, "nconmax", 0) or 0, 200)
    if BB_SIM_DT is not None:
        cfg.sim.mujoco.timestep = BB_SIM_DT
        cfg.decimation = max(1, int(round(0.02 / BB_SIM_DT)))

    # --- ball hold (difficulty axis) --------------------------------------------
    cfg.actions["ball_hold"] = microduck_mdp.BallHoldActionCfg(
        entity_name="ball", levels=(PLAY_HOLD,) if play else HOLD_LEVELS
    )

    # --- commands: rolling speeds start small (balance first) ---------------------
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.15 * CMD_SCALE, 0.15 * CMD_SCALE)
    twist.ranges.lin_vel_y = (-0.1 * CMD_SCALE, 0.1 * CMD_SCALE)
    twist.ranges.ang_vel_z = (-0.5 * CMD_SCALE, 0.5 * CMD_SCALE)
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)

    # --- observations: 61-D contract ---------------------------------------------
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        terms["head_command"] = deepcopy(terms["head_command"])
        terms["head_command"].func = microduck_mdp.basketball_head_pad
        terms["head_command"].params = {}
        terms["body_command"] = deepcopy(terms["body_command"])
        if blind and group == "actor":
            terms["body_command"].func = microduck_mdp.basketball_body_pad
            terms["body_command"].params = {}
        else:
            terms["body_command"].func = microduck_mdp.basketball_state
            terms["body_command"].params = {"pos_scale": 3.0, "vel_scale": 1.0}

    # b5: observation history on the actor's proprioception (the blind actor
    # must estimate the ball's drift under its feet and the rolling speed from
    # how its own state evolves; a single frame carries neither)
    if history > 1:
        for name in ("base_ang_vel", "projected_gravity", "raw_accelerometer", "joint_pos", "joint_vel", "actions"):
            term = cfg.observations["actor"].terms.get(name)
            if term is None:
                continue
            term = deepcopy(term)
            term.history_length = history
            term.flatten_history_dim = True
            cfg.observations["actor"].terms[name] = term

    # --- rewards ---------------------------------------------------------------------
    for name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
        "upright",
    ):
        cfg.rewards.pop(name, None)
    if ENABLE_VELOCITY_TRACKING:
        cfg.rewards["track_linear_velocity"].weight = TRACK_LIN_WEIGHT
        cfg.rewards["track_angular_velocity"].weight = TRACK_ANG_WEIGHT
    else:
        cfg.rewards.pop("track_linear_velocity", None)
        cfg.rewards.pop("track_angular_velocity", None)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["upright"] = RewardTermCfg(func=microduck_mdp.wrestle_upright, weight=UPRIGHT_WEIGHT, params={"std": 0.3})
    cfg.rewards["centered"] = RewardTermCfg(func=microduck_mdp.basketball_centered, weight=CENTERED_WEIGHT, params={"std": 0.04})
    cfg.rewards["feet_on_ball"] = RewardTermCfg(
        func=microduck_mdp.basketball_feet_on_ball,
        weight=FEET_ON_BALL_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"]),
            "ball_radius": BALL_RADIUS,
            "sole_gap": 0.012,
            "std": 0.02,
        },
    )
    cfg.rewards["height"] = RewardTermCfg(
        func=microduck_mdp.basketball_height,
        weight=HEIGHT_WEIGHT,
        params={"ball_radius": BALL_RADIUS, "root_height": ROOT_HEIGHT, "std": 0.03},
    )
    cfg.rewards["ball_speed"] = RewardTermCfg(func=microduck_mdp.basketball_ball_speed_l2, weight=BALL_SPEED_WEIGHT)

    # --- terminations -------------------------------------------------------------------
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["fell"] = TerminationTermCfg(
        func=microduck_mdp.basketball_fell,
        params={"ball_radius": BALL_RADIUS, "fall_tilt_deg": FALL_TILT_DEG, "min_height": 0.05, "max_offset": 0.16},
    )

    # --- events ----------------------------------------------------------------------------
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    spawn = EventTermCfg(
        func=microduck_mdp.reset_basketball,
        mode="reset",
        params={
            "ball_radius": BALL_RADIUS,
            "root_height": ROOT_HEIGHT,
            "xy_noise": 0.01,
            "yaw_range": (-math.pi, math.pi),
            "tilt_noise_deg": 2.0,
            "joint_noise": 0.05,
            "ball_vel_noise": 0.0,
        },
    )
    cfg.events = {"reset_basketball": spawn, **cfg.events}
    push = cfg.events.get("push_robot")
    if push is not None and PUSH_SCALE != 1.0:
        vr = push.params.get("velocity_range", {})
        for key in list(vr):
            lo, hi = vr[key]
            vr[key] = (lo * PUSH_SCALE, hi * PUSH_SCALE)

    # --- curriculum ----------------------------------------------------------------------
    for name in list(cfg.curriculum):
        params = getattr(cfg.curriculum[name], "params", {}) or {}
        cmd = params.get("command_name")
        rew = params.get("reward_name")
        if (cmd is not None and cmd not in cfg.commands) or (rew is not None and rew not in cfg.rewards):
            cfg.curriculum.pop(name)
    if not play:
        cfg.curriculum["basketball_hold"] = CurriculumTermCfg(
            func=microduck_mdp.basketball_hold_curriculum,
            params={"levels": HOLD_LEVELS, "promote_len_s": 6.0, "demote_len_s": 1.5},
        )
    if ACTION_RATE_WEIGHT is not None:
        weight = float(ACTION_RATE_WEIGHT)
        if not math.isfinite(weight) or weight > 0:
            raise ValueError("Basketball action-rate cost must be finite and non-positive")
        cfg.rewards["action_rate_l2"].weight = weight
        cfg.curriculum.pop("action_rate_weight", None)
    return cfg


MicroduckBasketballRlCfg = deepcopy(MicroduckRlCfg)
MicroduckBasketballRlCfg.experiment_name = "basketball"
MicroduckBasketballRlCfg.max_iterations = 6000
# Fresh basketball training uses the same recurrent actor family as the release.
MicroduckBasketballRlCfg.actor.class_name = "RNNModel"
MicroduckBasketballRlCfg.actor.rnn_type = "lstm"
MicroduckBasketballRlCfg.actor.rnn_hidden_dim = 256
MicroduckBasketballRlCfg.actor.rnn_num_layers = 1

"""MDP functions for microduck tasks"""

import math
from dataclasses import dataclass as _dataclass

import numpy as np
import torch
from typing import TYPE_CHECKING, Optional
import mujoco

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.entity import Entity
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers import CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi, quat_apply, quat_from_angle_axis
from rsl_rl.algorithms.ppo import PPO as _PPO

# ---------------------------------------------------------------------------
# Patch 1: RewardManager.compute — sanitize NaN rewards before they enter the
# PPO buffer.  mjlab computes rewards BEFORE resetting environments, so any
# reward term operating on a NaN physics state returns NaN.  That NaN
# propagates: NaN reward → NaN advantage → NaN loss → NaN gradient →
# NaN/negative std → crash in torch.normal on the next mini-batch.
# ---------------------------------------------------------------------------
_orig_reward_compute = _RewardManager.compute

def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums is updated inside compute() before nan_to_num can act.
    # Sanitize in-place so per-term metrics don't show NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)

_RewardManager.compute = _nan_safe_reward_compute

# ---------------------------------------------------------------------------
# Patch 2: PPO.compute_returns — sanitize advantages before normalization.
# At a sudden curriculum step (e.g. reward weight ×2.5) the value function is
# badly wrong: all TD errors shift by the same amount, std(advantages) → tiny,
# and (A − mean) / (std + 1e-8) → huge.  That blows up the gradient for std,
# which the optimizer then pushes below zero.  Zeroing NaN/Inf advantages
# before normalization keeps them in a safe range.
# ---------------------------------------------------------------------------
_orig_compute_returns = _PPO.compute_returns

def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns,    nan=0.0, posinf=0.0, neginf=0.0)

_PPO.compute_returns = _safe_compute_returns

# Patch 3 (ActorCritic._update_distribution std-clamp) was REMOVED in the mjlab
# 1.3.0 migration: rsl_rl 5.0.1 refactored the policy (no ActorCritic class; the
# distribution now lives in rsl_rl.modules.distribution). It was a defensive
# band-aid against std going negative/NaN (microban runs fine without it). If
# std-blowup recurs under 1.3.0, reinstate it against the new GaussianDistribution.

print("[mdp] Patches 1-2 active: NaN-safe reward/advantage")

# ---------------------------------------------------------------------------
# Patch 4: exporter_utils.get_base_metadata — the new microduck model has
# passive joints (jaw linkage closed via equality constraints) that are part
# of the articulation but have no XML actuator.  The upstream exporter
# iterates robot.joint_names (16) and indexes joint_name_to_ctrl_id (14),
# crashing with KeyError on passive_*.  Filter passive joints out of the
# exported metadata so policies stay consistent with the 14-dim action space.
# ---------------------------------------------------------------------------
from mjlab.rl import exporter_utils as _exporter_utils  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg as _JointActionCfg  # noqa: E402

def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }

_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
# Also patch the already-imported reference in the velocity task exporter.
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter  # noqa: E402
    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass

print("[mdp] Patch 4 active: ONNX export filters passive_* joints")

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Name patterns matching the 4 neck/head actuated joints. Used by head_pose
# tracking reward and by UniformPoseCommand asset hookups.
_NECK_JOINT_PATTERNS = [r".*neck_pitch.*", r".*head_pitch.*", r".*head_yaw.*", r".*head_roll.*"]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """Entity-local indices of the servo (non-``passive_``) joints, cached.

    All joint-index-based reward/event params in this module (``joint_indices``,
    ``target_overrides``, qpos-column math) are written against the canonical
    14-servo layout. On models with extra unactuated joints — backlash hinges,
    roller wheels, the jaw linkage, all named ``passive_*`` — the entity joint
    array is wider and interleaved, so raw indices would select the wrong
    joints. Index through this list to recover the servo-only view; on plain
    models it is the identity.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Warm-start a fraction of reset environments with a random forward velocity.

    The robot spawns already moving in its body-forward direction, so it first
    discovers what coasting at speed feels like. The fraction decreases over
    training, forcing it to progressively earn that speed from rest.

    Args:
        velocity_range: (min, max) forward speed in m/s.
        fraction_stages: list of {"step": int, "fraction": float} dicts, sorted by step.
            The fraction active at the current training step is used.
            Example: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity config.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # Determine current fraction from training step
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # Build horizontal forward direction from yaw only — ignoring pitch/roll.
    # IMPORTANT: read quaternion from qpos, NOT from root_link_quat_w.
    # root_link_quat_w reads xquat which requires sim.forward() to be current.
    # After reset_base writes a new yaw to qpos, xquat is still stale (old episode).
    # qpos is updated immediately by write_root_pose, so it's always fresh.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # Spin wheels to match forward velocity — prevents instantaneous no-slip braking.
    # Wheel radius = 0.0175 m (measured).
    # All 4 wheels spin at +ω for forward motion (verified by test_wheel_direction.py).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """
    Reset cached action history for environments that are being reset.
    This is critical for action rate and acceleration penalty terms.

    This function should be called in the post_reset callback or at episode termination.

    Args:
        env: The environment
        env_ids: Indices of environments being reset
        asset_cfg: Asset configuration
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # Reset leg action rate cache
    if hasattr(env, '_prev_leg_actions'):
        # Set to current action (or zero if no action yet)
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # Reset neck action rate cache
    if hasattr(env, '_prev_neck_actions'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # Reset leg action acceleration cache
    if hasattr(env, '_prev_leg_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # Reset neck action acceleration cache
    if hasattr(env, '_prev_neck_actions_for_acc'):
        if hasattr(env, 'action_manager') and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # Reset joint velocity cache for joint accelerations
    if hasattr(asset.data, '_prev_joint_vel'):
        # Get current joint velocities for reset environments
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # Reset contact frequency tracking
    if hasattr(env, '_contact_change_count'):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, '_contact_change_timer'):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, '_prev_contacts_for_freq'):
        if "feet_ground_contact" in env.scene.sensors:
            contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
            env._prev_contacts_for_freq[env_ids] = contacts

    # Reset foot force smoothness tracking
    if hasattr(env, '_prev_foot_forces'):
        if "feet_ground_contact" in env.scene.sensors:
            forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
            env._prev_foot_forces[env_ids] = forces

    # Reset actuator torque rate tracking
    if hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


# -----------------------------------------------------------------------------
# Ladder climb
# -----------------------------------------------------------------------------

# =============================================================================
# Stair-ladder climbing (Mjlab-LadderClimb-MicroDuck)
# =============================================================================
#
# Alternating-tread stair-ladder; see robot/ladder.py for the geometry
# rationale and microduck_ladder_env_cfg.py for the task.  All ladder state
# lives in ``env._stair`` (a SimpleNamespace of per-env tensors) and is written
# by ``reset_stair_ladder``.  Contact classification reads the raw MuJoCo
# contact buffer once per step (``_stair_contacts``) because mjlab contact
# sensors need a single secondary body and the ladder is one entity per tread.

from types import SimpleNamespace as _SimpleNamespace

from mjlab_microduck.robot import ladder as _ladder


def _stair_state(env: ManagerBasedRlEnv) -> _SimpleNamespace | None:
    return getattr(env, "_stair", None)


def _quat_from_yaw_pitch_roll(
    yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor
) -> torch.Tensor:
    """Scalar-first quaternion for ZYX Euler angles (yaw about z applied last)."""
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    return torch.stack(
        (
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ),
        dim=-1,
    )


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _leg_table_tensors(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leg table as (deltas_m[D], forwards_m[F], offsets[D, F, 3]) on device."""
    cached = getattr(env, "_stair_leg_table", None)
    if cached is not None:
        return cached
    table = _ladder.load_leg_table()
    deltas = torch.tensor([d * 1e-3 for d in table["delta_mm"]], device=env.device)
    forwards = torch.tensor([f * 1e-3 for f in table["forward_mm"]], device=env.device)
    offsets = torch.zeros(len(deltas), len(forwards), 3, device=env.device)
    for i, d in enumerate(table["delta_mm"]):
        for j, f in enumerate(table["forward_mm"]):
            entry = table["table"][f"{d}:{f}"]
            offsets[i, j] = torch.tensor(
                (entry["hip_pitch"], entry["knee"], entry["ankle"]), device=env.device
            )
    env._stair_leg_table = (deltas, forwards, offsets)
    return env._stair_leg_table


def _leg_offsets_bilinear(
    env: ManagerBasedRlEnv, delta_m: torch.Tensor, forward_m: torch.Tensor
) -> torch.Tensor:
    """Bilinear (delta, forward) interpolation of the left-leg offset table."""
    deltas, forwards, offsets = _leg_table_tensors(env)

    def _bracket(grid: torch.Tensor, value: torch.Tensor):
        step = grid[1] - grid[0]
        pos = ((value - grid[0]) / step).clamp(0.0, float(len(grid) - 1) - 1e-6)
        lo = pos.floor().long()
        return lo, pos - lo.float()

    di, dt = _bracket(deltas, delta_m)
    fi, ft = _bracket(forwards, forward_m)
    o00 = offsets[di, fi]
    o10 = offsets[di + 1, fi]
    o01 = offsets[di, fi + 1]
    o11 = offsets[di + 1, fi + 1]
    return (
        o00 * ((1 - dt) * (1 - ft))[:, None]
        + o10 * (dt * (1 - ft))[:, None]
        + o01 * ((1 - dt) * ft)[:, None]
        + o11 * (dt * ft)[:, None]
    )


STAIR_SWING_DUTY = 0.4


def _stair_swing_arc(
    u: torch.Tensor,
    riser: torch.Tensor,
    run: torch.Tensor,
    geometry: _ladder.StairLadderGeometry,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swing-foot displacement (dx, dz) from its start tread at swing fraction u.

    The toe extends 31 mm ahead of the ankle while the next same-side tread's
    rear edge is only ~10 mm ahead of it, so a straight lift drives the toe
    into that tread's underside.  The feasible path is retract (u < 0.3),
    lift with the toe behind the rear edge (0.3-0.7), then reach forward and
    land (0.7-1.0) -- how a person climbs a steep ladder.
    """
    spacing = float(geometry.same_side_spacing())
    rear_edge = spacing * run - 0.5 * geometry.tread_depth_m  # ahead of the ankle
    x_back = (_ladder.SOLE_TOE_AHEAD_OF_SITE_M - rear_edge + 0.005).clamp_min(0.0)
    top = spacing * riser
    # Walking-style timing: the swing occupies the first SWING_DUTY of the
    # half cycle, the rest is double support (r19 playback: a slow single-
    # support swing is not something the soft servos hold).
    u = (u / STAIR_SWING_DUTY).clamp(0.0, 1.0)
    p1 = (u / 0.3).clamp(0.0, 1.0)
    p2 = ((u - 0.3) / 0.4).clamp(0.0, 1.0)
    p3 = ((u - 0.7) / 0.3).clamp(0.0, 1.0)
    dx = -x_back * p1 + (spacing * run + x_back) * p3
    dz = 0.006 * p1 + (top + 0.010 - 0.006) * p2 - 0.006 * p3
    return dx, dz


def _stair_ensure_state(env: ManagerBasedRlEnv, geometry: _ladder.StairLadderGeometry) -> _SimpleNamespace:
    state = _stair_state(env)
    if state is not None:
        return state
    n, dev = env.num_envs, env.device
    zeros = lambda *shape: torch.zeros(n, *shape, device=dev)  # noqa: E731
    state = _SimpleNamespace(
        geometry=geometry,
        level=torch.zeros(n, device=dev, dtype=torch.long),
        riser=zeros() + 0.02,
        angle=zeros() + math.radians(55.0),
        x0=zeros() + 0.2,
        y0=zeros(),
        tread_top=zeros(geometry.num_treads),
        tread_centre=zeros(geometry.num_treads, 3),
        tread_nose_x=zeros(geometry.num_treads),
        tread_target_xy=zeros(geometry.num_treads, 2),
        tread_yaw=zeros(geometry.num_treads),
        flat=zeros(),
        lateral=zeros(),
        dyaw=zeros(),
        flight_base=zeros(geometry.num_flights, 2),
        flight_yaw=zeros(geometry.num_flights),
        flight_z=zeros(geometry.num_flights),
        # Walking paths across the landings (world xy), see ladder.walk_paths.
        path_pts=zeros(max(geometry.num_flights - 1, 1), _ladder.WALK_PATH_POINTS, 2),
        path_yaw=zeros(max(geometry.num_flights - 1, 1), _ladder.WALK_PATH_POINTS),
        path_rem=zeros(max(geometry.num_flights - 1, 1), _ladder.WALK_PATH_POINTS),
        tread_side=torch.tensor(
            [geometry.tread_side(i) for i in range(geometry.num_treads)],
            device=dev,
            dtype=torch.float32,
        ),
        spawn_root_z=zeros(),
        spawn_on_floor=torch.zeros(n, device=dev, dtype=torch.bool),
        start_tread=torch.zeros(n, device=dev, dtype=torch.long),
        prev_root_z=zeros(),
        prev_root_vz=zeros(),
        vz_ema=zeros(),
        max_rise=zeros(),
    )
    env._stair = state
    return state


def reset_stair_ladder(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    geometry: _ladder.StairLadderGeometry = _ladder.LADDER_GEOMETRY,
    level_table: tuple[dict, ...] = (
        {"riser": (0.015, 0.018), "angle": (45.0, 52.0)},
    ),
    floor_spawn_prob: float = 0.3,
    max_start_tread: int = 8,
    floor_gap_m: float = 0.04,
    position_noise: float = 0.004,
    yaw_noise_deg: float = 5.0,
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.03,
    fixed_riser_m: float | None = None,
    fixed_angle_deg: float | None = None,
    fixed_level: int | None = None,
    ladder_y_noise: float = 0.01,
    swing_spawn_prob: float = 0.0,
    swing_fraction_range: tuple[float, float] = (0.25, 0.95),
    swing_clearance: float = 0.012,
    swing_lateral_shift: float = 0.02,
    level_mix_prob: float = 0.0,
    landing_spawn_prob: float = 0.0,
    flat_range: tuple[float, float] = (0.06, 0.18),
    landing_approach_prob: float = 0.0,
    top_spawn_prob: float = 0.0,
    approach_max_below: int = 3,
    nose_jitter_m: float = 0.0,
    path_spawn_frac: float = 0.0,
    approach_swing_prob: float = 0.7,
    min_start_tread: int = 0,
) -> None:
    """Place the ladder for each environment and spawn the robot on it.

    ``approach_swing_prob``: fraction of the landing-approach spawns placed
    mid-swing toward the landing (the rest stand on the two treads below).

    ``path_spawn_frac`` (fraction of the landing spawns, staircase mode) puts
    the robot in the HOME pose part-way along the walking path across the
    first landing, facing along the path (reverse curriculum of the walk).

    Staircase mode (``geometry.landing_every > 0``): every flight of mini
    treads ends on a full-width landing and the next flight starts ``flat``
    metres past its nose (``flat`` drawn per env from the level's ``flat``
    range, else ``flat_range``).  ``landing_spawn_prob`` (fraction of
    non-floor spawns) puts the robot in the HOME pose on the first landing
    in front of the next flight: the flat-to-climb transition.

    With ``swing_spawn_prob`` (fraction of on-ladder spawns) the lower foot is
    instead placed mid-step, ``u`` of the way along an arc from its tread to
    the next same-side tread (reverse-curriculum spawn of the maneuver's last
    mile), with the trunk shifted toward the supporting foot.

    Geometry (riser, angle) is sampled from the environment's current
    curriculum level, clamped to the tread-clearance rules, and applied by
    writing every tread/rail mocap pose.  The robot then spawns either on the
    floor in front of the ladder (HOME pose) or standing on treads ``k`` and
    ``k+1`` with the higher leg shortened from the leg table.  Both are
    physically settled poses, not searched contact anchors.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    dev = env.device
    state = _stair_ensure_state(env, geometry)
    asset: Entity = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]

    # --- geometry per environment -------------------------------------------
    if fixed_level is not None:
        state.level[env_ids] = int(fixed_level)
    level = state.level[env_ids].clamp(0, len(level_table) - 1)
    # r21 lesson: once every environment reached the top level the policy
    # forgot the shallow ones (level-0 falls 96%).  A fraction of resets
    # draws a uniformly random level for this episode only.
    if level_mix_prob > 0.0:
        mix = torch.rand(n, device=dev) < level_mix_prob
        random_level = torch.randint(0, len(level_table), (n,), device=dev)
        level = torch.where(mix, random_level, level)
    riser_lo = torch.tensor([s["riser"][0] for s in level_table], device=dev)
    riser_hi = torch.tensor([s["riser"][1] for s in level_table], device=dev)
    angle_lo = torch.tensor([s["angle"][0] for s in level_table], device=dev)
    angle_hi = torch.tensor([s["angle"][1] for s in level_table], device=dev)
    u = torch.rand(n, device=dev)
    riser = riser_lo[level] + u * (riser_hi[level] - riser_lo[level])
    u = torch.rand(n, device=dev)
    angle_deg = angle_lo[level] + u * (angle_hi[level] - angle_lo[level])
    if fixed_riser_m is not None:
        riser = torch.full_like(riser, float(fixed_riser_m))
    if fixed_angle_deg is not None:
        angle_deg = torch.full_like(angle_deg, float(fixed_angle_deg))
    # Clearance rules (see robot/ladder.py): toe under the next same-side
    # tread, ankle shell either above-cleared or behind it.  Clamp the riser
    # up and the angle down where the depth rule binds.
    riser = riser.clamp_min(geometry.min_riser_m())
    spacing = geometry.same_side_spacing()
    ankle_allowance = (
        _ladder.SOLE_TOE_AHEAD_OF_SITE_M + 0.005 - _ladder.ANKLE_SHELL_AHEAD_OF_SITE_M
    )
    min_run = max(geometry.tread_depth_m - ankle_allowance, 1e-4)
    max_angle = torch.rad2deg(torch.atan(spacing * riser / min_run))
    clears = (spacing * riser - geometry.tread_thickness_m) >= (
        _ladder.ANKLE_SHELL_HEIGHT_M + 0.003
    )
    angle_deg = torch.where(clears, angle_deg, torch.minimum(angle_deg, max_angle))
    angle = torch.deg2rad(angle_deg)
    staircase = geometry.landing_every > 0 and geometry.num_flights > 1
    def _level_range(key: str, default: tuple[float, float], scale: float = 1.0) -> torch.Tensor:
        lo = torch.tensor([s.get(key, default)[0] for s in level_table], device=dev) * scale
        hi = torch.tensor([s.get(key, default)[1] for s in level_table], device=dev) * scale
        return lo[level] + torch.rand(n, device=dev) * (hi[level] - lo[level])

    if staircase:
        flat = _level_range("flat", flat_range)
        lateral = _level_range("lateral", (0.0, 0.0))
        dyaw = _level_range("dyaw_deg", (0.0, 0.0), math.pi / 180.0)
    else:
        flat = torch.zeros(n, device=dev)
        lateral = torch.zeros(n, device=dev)
        dyaw = torch.zeros(n, device=dev)

    # --- robot spawn ---------------------------------------------------------
    on_floor = torch.rand(n, device=dev) < floor_spawn_prob
    max_start = max(0, min(max_start_tread, geometry.num_treads - 5))
    # Stance spawns need k+1 to be an ordinary tread (the leg table assumes
    # one riser between the feet).  k may be the first landing: the lower
    # foot then stands at the landing's front edge (heel over the last mini
    # tread, as when stepping onto it) and the higher foot on the first
    # tread of the next flight, or, when the flat is long enough, both feet
    # stand on the landing in the HOME pose (the flat-to-climb transition).
    land_idx = geometry.landing_every - 1 if staircase else -1
    # k+1 may be the first landing (higher foot just landed on it, lower foot
    # on the last mini tread): the state right after the hard step, so its
    # value is seen on-policy.  The forward offset is clamped by the leg
    # table (65 mm forward; the table was extended for this spawn).
    allowed = [
        k for k in range(max(0, min(int(min_start_tread), max_start)), max_start + 1)
        if (not geometry.is_landing(k) or k == land_idx) and not geometry.is_landing(k + 1)
    ] or [0]
    allowed_t = torch.tensor(allowed, device=dev)
    start = allowed_t[torch.randint(0, len(allowed), (n,), device=dev)]
    force_app = torch.zeros(n, dtype=torch.bool, device=dev)
    if staircase:
        # Landing and approach spawns are independent of ``max_start_tread``
        # (which caps the random on-ladder starts): earlier the approach block
        # was silently skipped whenever the cap excluded the landing index.
        force_landing = (~on_floor) & (torch.rand(n, device=dev) < landing_spawn_prob)
        start = torch.where(force_landing, torch.full_like(start, land_idx), start)
        # two to ``approach_max_below`` treads below the first landing: both
        # foot parities lead the landing step (the just-landed spawn, k + 1,
        # collapses: s7).  s21: from further down the duck ARRIVES climbing,
        # with the momentum and the craned head that made every natural
        # arrival nose-dive while the standing-start spawns succeeded.
        # s26: approach spawns below EVERY landing (s25 crossed the first
        # landing in 95 % of floor starts and then dived at the second one,
        # which no approach spawn had ever covered)
        landings = [i for i in range(geometry.num_treads) if geometry.is_landing(i)]
        approaches = [
            k for L in landings for k in range(L)
            if not geometry.is_landing(k) and not geometry.is_landing(k + 1)
            and 2 <= L - k <= max(2, int(approach_max_below))
        ]
        force_app = torch.zeros(n, dtype=torch.bool, device=dev)
        if approaches and landing_approach_prob > 0.0:
            # s28: the top landing's stances were ~4 % of episodes (eight start
            # treads over two landings); give the two treads right below the
            # LAST landing half of the approach spawns.  Its step geometry and
            # foot targets are identical to the first landing's, and the
            # static top stance still failed (0-20 %) - a data-share problem.
            top = landings[-1]
            near_top = [k for k in approaches if 2 <= top - k <= 3]
            app = torch.tensor(approaches, device=dev)[torch.randint(0, len(approaches), (n,), device=dev)]
            if near_top:
                app_top = torch.tensor(near_top, device=dev)[torch.randint(0, len(near_top), (n,), device=dev)]
                app = torch.where(torch.rand(n, device=dev) < 0.5, app_top, app)
            force_app = (~on_floor) & (~force_landing) & (torch.rand(n, device=dev) < landing_approach_prob)
            start = torch.where(force_app, app, start)
    # s34: HOME spawn ON THE TOP landing (both feet on the platform): the
    # still-gated success pays there within 0.2 s, so "stand on the top" gets
    # a value before the hop from 13/14 ever lands it (reverse curriculum).
    top_idx = geometry.num_treads - 1
    has_top = staircase and geometry.is_landing(top_idx)
    if has_top and top_spawn_prob > 0.0:
        force_top = (~on_floor) & (~force_app) & (torch.rand(n, device=dev) < top_spawn_prob)
        start = torch.where(force_top, torch.full_like(start, top_idx), start)
    start = torch.where(on_floor, torch.zeros_like(start), start)
    run = riser / torch.tan(angle)
    start_is_landing = staircase & (start == land_idx) if staircase else torch.zeros(n, dtype=torch.bool, device=dev)
    start_is_top = (start == top_idx) if has_top else torch.zeros(n, dtype=torch.bool, device=dev)
    # Both feet on the landing (HOME) needs the next flight's first tread to
    # start clear of the toe: flat + run - (0.015 + toe + depth) >= ~6 mm, so
    # only for flats >= 75 mm; the site then sits 15 mm past the landing nose
    # (toe 46 mm onto it).  Otherwise the lower foot stands where it arrives
    # from the last mini tread: one run behind the next flight's first tread
    # (heel 35 mm past the landing nose), and the stance is the regular
    # one-run, one-riser stagger.
    both_on_landing = (start_is_landing & (flat >= 0.075)) | start_is_top
    on_landing = both_on_landing
    tops_all = _ladder.tread_top_heights(riser, geometry.num_treads)
    top_k = tops_all.gather(1, start[:, None]).squeeze(1)
    # Flight of the spawn tread and its base height / x offset from x0.  A
    # landing spawn is expressed in the *next* flight's frame.
    flight_of_start = torch.tensor(
        [geometry.flight_of(k) for k in range(geometry.num_treads)], device=dev
    )[start]
    flight_of_start = torch.where(start_is_landing, flight_of_start + 1, flight_of_start)
    # Flight frames with x0 = y0 = 0: base offsets, yaws and base heights.
    zero = torch.zeros(n, device=dev)
    off_all, yaw_all, base_z_all = _ladder.flight_frames(geometry, riser, angle, zero, zero, flat, lateral, dyaw)
    start_off = off_all.gather(1, flight_of_start[:, None, None].expand(-1, 1, 2)).squeeze(1)
    start_yaw = yaw_all.gather(1, flight_of_start[:, None]).squeeze(1)
    start_base_z = base_z_all.gather(1, flight_of_start[:, None]).squeeze(1)
    # Higher foot on the first landing: from the last mini tread's centre to
    # the landing nose (setback + half a tread further than a regular run).
    onto_landing = staircase & (start + 1 == land_idx)
    stance_forward = torch.where(
        onto_landing,
        (run + geometry.landing_setback_m + 0.5 * geometry.tread_depth_m + geometry.landing_target_ahead_m).clamp_max(0.065),
        run,
    )

    root_xy_noise = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * position_noise
    root_x = origins[:, 0] + root_xy_noise[:, 0]
    root_y = origins[:, 1] + root_xy_noise[:, 1]
    site_clearance = 0.002
    root_z_ladder = (
        origins[:, 2]
        + top_k
        + _ladder.SOLE_BOTTOM_ABOVE_SITE_M
        + site_clearance
        + _ladder.HOME_TRUNK_ABOVE_SITE_M
    )
    root_z_floor = origins[:, 2] + 0.12 + site_clearance
    root_z = torch.where(on_floor, root_z_floor, root_z_ladder)

    # Ladder placement relative to the robot.  On the ladder: the lower foot's
    # ankle axis (site) sits at the tread centre, giving the centre of mass
    # half a tread of static margin both ways.  On the floor: tread 0's rear
    # edge is ``floor_gap_m`` ahead of the toe.
    # Spawn site in its flight's frame (u along the axis, v = 0).  Flight-0
    # spawns: at tread k's centre (stance) or with tread 0's rear edge
    # ``floor_gap_m`` ahead of the toe (floor).  Landing spawns: see above.
    u_stance = (top_k - start_base_z) / torch.tan(angle) - 0.5 * geometry.tread_depth_m
    u_floor = -(
        _ladder.SOLE_TOE_AHEAD_OF_SITE_M + floor_gap_m + geometry.tread_depth_m - riser / torch.tan(angle)
    )
    u_landing = torch.where(both_on_landing, 0.015 - flat, torch.full_like(flat, -0.5 * geometry.tread_depth_m))
    # top landing (last flight's frame): its nose sits setback + half a tread
    # past the virtual tread centre; the ankle site 6 cm onto the platform
    u_top = u_stance + geometry.landing_setback_m + 0.5 * geometry.tread_depth_m + 0.06
    u_site = torch.where(on_floor, u_floor, torch.where(start_is_landing, u_landing, torch.where(start_is_top, u_top, u_stance)))
    # Path spawns: part-way along the walk across the first landing, facing
    # along the path (the path is computed with a zero ladder origin, then
    # the origin is solved so the chosen point lands on the robot).
    path_spawn = (
        both_on_landing & ~start_is_top & (torch.rand(n, device=dev) < path_spawn_frac)
        if staircase
        else torch.zeros(n, dtype=torch.bool, device=dev)
    )
    if staircase and bool(path_spawn.any()):
        pts0, pyaw0, _rem0 = _ladder.walk_paths(geometry, riser, angle, off_all, yaw_all)
        frac = 0.15 + 0.7 * torch.rand(n, device=dev)
        k = (frac * (pts0.shape[2] - 1)).round().long()
        p_s = pts0[:, 0].gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)
        t_s = pyaw0[:, 0].gather(1, k[:, None]).squeeze(1)
        start_yaw = torch.where(path_spawn, t_s, start_yaw)
    # Keep the robot at the env origin (+ noise): solve the ladder origin so
    # that base_f + R(yaw_f) (u_site, 0) lands on the robot's foot site.
    c_f, s_f = torch.cos(start_yaw), torch.sin(start_yaw)
    x0 = root_xy_noise[:, 0] - start_off[:, 0] - u_site * c_f
    y0 = root_xy_noise[:, 1] - start_off[:, 1] - u_site * s_f
    y0 = y0 + (torch.rand(n, device=dev) * 2.0 - 1.0) * ladder_y_noise
    if staircase and bool(path_spawn.any()):
        x0 = torch.where(path_spawn, root_xy_noise[:, 0] - p_s[:, 0], x0)
        y0 = torch.where(path_spawn, root_xy_noise[:, 1] - p_s[:, 1], y0)

    state.riser[env_ids] = riser
    state.angle[env_ids] = angle
    state.x0[env_ids] = x0
    state.y0[env_ids] = y0
    state.flat[env_ids] = flat
    state.lateral[env_ids] = lateral
    state.dyaw[env_ids] = dyaw
    fbase, fyaw, fz = _ladder.flight_frames(geometry, riser, angle, x0, y0, flat, lateral, dyaw)
    state.flight_base[env_ids] = fbase + origins[:, None, :2]
    state.flight_yaw[env_ids] = fyaw
    state.flight_z[env_ids] = fz + origins[:, 2:3]
    if staircase:
        ppts, pyaw, prem = _ladder.walk_paths(geometry, riser, angle, fbase, fyaw)
        state.path_pts[env_ids] = ppts + origins[:, None, None, :2]
        state.path_yaw[env_ids] = pyaw
        state.path_rem[env_ids] = prem
    if hasattr(state, "path_fresh"):
        state.path_fresh[env_ids] = True
    # Irregular tread spacing: every tread's nose shifted along the axis by
    # up to nose_jitter_m (landings excluded).  The plain-ladder policy took
    # a fixed two-run stride whatever the target said (staircase s4-s10); a
    # policy must see variable spacing to learn to read the target distance.
    # ``nose_jitter_m`` is the amplitude at a 26 mm two-run stride and scales
    # with the run: a fixed 20 mm exceeded level 0's 15 mm run (r24 lost
    # level 0 entirely: 98 % falls).
    nose_jitter = None
    if nose_jitter_m > 0.0:
        amp = nose_jitter_m * (2.0 * run / 0.026).clamp(0.3, 1.5)
        nose_jitter = (torch.rand(n, geometry.num_treads, device=dev) * 2.0 - 1.0) * amp[:, None]
        is_land = torch.tensor([geometry.is_landing(i) for i in range(geometry.num_treads)], device=dev)
        nose_jitter = torch.where(is_land[None, :], torch.zeros_like(nose_jitter), nose_jitter)
        # keep the spawn tread's nose where the spawn formula assumed it
        nose_jitter.scatter_(1, start[:, None], 0.0)
        nose_jitter.scatter_(1, (start + 1).clamp_max(geometry.num_treads - 1)[:, None], 0.0)
    centres, tops, nose_u, target_xy, tread_yaw = _ladder.tread_layout(
        geometry, riser, angle, x0, y0, flat, lateral, dyaw, nose_jitter
    )
    state.tread_top[env_ids] = tops + origins[:, 2:3]
    state.tread_centre[env_ids] = centres + origins[:, None, :]
    state.tread_nose_x[env_ids] = nose_u  # along-axis nose distance from the flight base
    state.tread_target_xy[env_ids] = target_xy + origins[:, None, :2]
    state.tread_yaw[env_ids] = tread_yaw

    # --- write tread / rail mocap poses --------------------------------------
    for index in range(geometry.num_treads):
        entity: Entity = env.scene[f"{_ladder.TREAD_ENTITY_PREFIX}{index:02d}"]
        pose = torch.cat((state.tread_centre[env_ids, index], _ladder.quat_yaw(tread_yaw[:, index])), dim=-1)
        entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
    if _ladder.TOP_FLOOR_ENTITY in env.scene.entities:
        last = geometry.num_treads - 1
        pose = torch.cat((state.tread_centre[env_ids, last], _ladder.quat_yaw(tread_yaw[:, last])), dim=-1)
        env.scene[_ladder.TOP_FLOOR_ENTITY].write_mocap_pose_to_sim(pose, env_ids=env_ids)
    rail_pos, rail_quat = _ladder.rail_poses(geometry, angle, x0, y0, riser, flat, lateral, dyaw)
    for slot, name in enumerate(_ladder.rail_entity_names(geometry)):
        entity = env.scene[name]
        pose = torch.cat((rail_pos[:, slot] + origins, rail_quat[:, slot]), dim=-1)
        entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    # --- robot root and joints -----------------------------------------------
    yaw = start_yaw + (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(yaw_noise_deg)
    pitch = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(tilt_noise_deg)
    roll = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.radians(0.5 * tilt_noise_deg)
    quat = _quat_from_yaw_pitch_roll(yaw, pitch, roll)
    # Mid-swing spawns lean the trunk over the supporting (higher) foot.
    flat_spawn = on_floor | on_landing
    # No mid-swing spawns from the landing: the arc assumes regular spacing.
    swing_draw = torch.rand(n, device=dev)
    # Approach spawns are mostly mid-swing onto the landing (reverse curriculum
    # for the lift-then-reach step); other stance spawns keep swing_spawn_prob.
    swing_p = torch.where(force_app, torch.full_like(swing_draw, float(approach_swing_prob)), torch.full_like(swing_draw, swing_spawn_prob)) if staircase else torch.full_like(swing_draw, swing_spawn_prob)
    swing = (~flat_spawn) & (~start_is_landing) & (~onto_landing) & (swing_draw < swing_p)
    support_side = (
        torch.where(((start + 1) % 2 == 0), 1.0, -1.0)
        if geometry.alternating
        else torch.zeros(n, device=dev)
    )
    u_pre = swing_fraction_range[0] + torch.rand(n, device=dev) * (
        swing_fraction_range[1] - swing_fraction_range[0]
    )
    # Trunk shift in the flight frame: forward u*run, sideways toward the support foot.
    sw_u = swing.float() * u_pre * run
    sw_v = swing.float() * support_side * swing_lateral_shift
    root_x = root_x + sw_u * c_f - sw_v * s_f
    root_y = root_y + sw_u * s_f + sw_v * c_f
    root_z = root_z + swing.float() * u_pre * 0.5 * riser
    root_pos = torch.stack((root_x, root_y, root_z), dim=-1)
    asset.write_root_link_pose_to_sim(torch.cat((root_pos, quat), dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    servo_ids = _servo_joint_ids(env, asset)
    # Higher foot is the leg standing on tread k+1: left if k+1 is even.
    left_is_higher = ((start + 1) % 2 == 0) if geometry.alternating else torch.zeros_like(on_floor)
    offsets = _leg_offsets_bilinear(env, riser, stance_forward)  # (n, 3): hip_pitch, knee, ankle
    offsets = torch.where(flat_spawn[:, None], torch.zeros_like(offsets), offsets)
    # Mid-swing spawns: the lower foot is lifted u of the way to tread k+2
    # and the trunk has already moved u*run forward and u*riser/2 up, so
    # the landing is the natural continuation rather than a retreat.
    u = u_pre
    spacing = float(geometry.same_side_spacing())
    trunk_dx = swing.float() * u * run
    trunk_dz = swing.float() * u * 0.5 * riser
    support_offsets = _leg_offsets_bilinear(env, riser - trunk_dz, run - trunk_dx)
    offsets = torch.where(swing[:, None], support_offsets, offsets)
    # Clearance tapers from 8 mm at lift-off to 4 mm at u = 1, so the range
    # can include just-landed states (foot resting, trunk not yet forward).
    arc_dx, arc_dz = _stair_swing_arc(u, riser, run, geometry)
    # Swing onto a landing (staircase: lower foot two treads below it): the
    # landing nose is ``setback`` further than a regular tread, and the
    # target sits at the nose rather than a half tread behind it.
    if staircase:
        to_landing = swing & torch.tensor(
            [geometry.is_landing(min(k + 2, geometry.num_treads - 1)) for k in range(geometry.num_treads)],
            device=dev,
        )[start]
        extra = geometry.landing_setback_m + 0.5 * geometry.tread_depth_m + geometry.landing_target_ahead_m
        p3 = ((u / STAIR_SWING_DUTY).clamp(0.0, 1.0) - 0.7).clamp_min(0.0) / 0.3
        arc_dx = arc_dx + to_landing.float() * extra * p3.clamp(0.0, 1.0)
    swing_delta = arc_dz - trunk_dz
    swing_forward = arc_dx - trunk_dx
    swing_offsets = _leg_offsets_bilinear(env, swing_delta, swing_forward)
    swing_offsets = torch.where(swing[:, None], swing_offsets, torch.zeros_like(swing_offsets))
    left_ids = _stair_leg_joint_ids(env, asset, "left")
    right_ids = _stair_leg_joint_ids(env, asset, "right")
    left_mask = left_is_higher.float()[:, None]
    # Right-leg joints mirror the left (HOME signs are negated).  The higher
    # (support) leg gets the stance offsets, the lower leg the swing offsets.
    joint_pos[:, left_ids] += offsets * left_mask + swing_offsets * (1.0 - left_mask)
    joint_pos[:, right_ids] += -offsets * (1.0 - left_mask) - swing_offsets * left_mask
    joint_pos[:, servo_ids] += (
        torch.rand(n, len(servo_ids), device=dev) * 2.0 - 1.0
    ) * joint_noise
    limits = asset.data.joint_pos_limits[env_ids][:, servo_ids]
    joint_pos[:, servo_ids] = torch.maximum(
        torch.minimum(joint_pos[:, servo_ids], limits[..., 1] - 0.02),
        limits[..., 0] + 0.02,
    )
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    # A custom pose reset must also clear the constraint warm-start, otherwise
    # the new contacts inherit whatever collision ended the previous episode.
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim.data, "qacc_warmstart"):
        sim.data.qacc_warmstart[env_ids] = 0.0

    state.spawn_root_z[env_ids] = root_z
    state.spawn_on_floor[env_ids] = on_floor
    if hasattr(state, "landing_paid"):
        state.landing_paid[env_ids] = False
        # a landing the robot is spawned on (or has under a foot) is not a step
        state.landing_paid[env_ids, start.clamp_max(geometry.num_treads - 1)] = True
    state.start_tread[env_ids] = start
    state.prev_root_z[env_ids] = root_z
    if not hasattr(state, "foot_support_z"):
        state.foot_support_z = torch.zeros(env.num_envs, 2, device=dev)
    lower_top = torch.where(on_floor, torch.zeros_like(top_k), top_k)
    higher_top = torch.where(on_floor, torch.zeros_like(top_k), torch.where(on_landing, top_k, top_k + riser))
    # Left foot is the higher one when k+1 is even.
    state.foot_support_z[env_ids, 0] = torch.where(left_is_higher, higher_top, lower_top)
    state.foot_support_z[env_ids, 1] = torch.where(left_is_higher, lower_top, higher_top)
    if not hasattr(state, "foot_last_tread"):
        state.foot_last_tread = torch.full((env.num_envs, 2), -1, dtype=torch.long, device=dev)
        state.overstep = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    none = torch.full_like(start, -1)
    lower_idx = torch.where(on_floor, none, start)
    higher_idx = torch.where(on_floor, none, torch.where(on_landing, start, start + 1))
    state.foot_last_tread[env_ids, 0] = torch.where(left_is_higher, higher_idx, lower_idx)
    state.foot_last_tread[env_ids, 1] = torch.where(left_is_higher, lower_idx, higher_idx)
    state.overstep[env_ids] = False
    if hasattr(state, "prev_potential"):
        state.prev_potential[env_ids] = torch.minimum(
            root_z, state.foot_support_z[env_ids].mean(dim=1) + origins[:, 2] + 0.125
        )
    state.prev_root_vz[env_ids] = 0.0
    state.vz_ema[env_ids] = 0.0
    state.max_rise[env_ids] = 0.0
    if hasattr(state, "target_fresh"):
        state.target_fresh[env_ids] = True
    if hasattr(env, "_stair_airborne_steps"):
        env._stair_airborne_steps[env_ids] = 0
    # Gait phase: the lower foot swings first (left in [0, 0.5), right after).
    if not hasattr(state, "phase"):
        state.phase = torch.zeros(env.num_envs, device=dev)
    swing_u = torch.where(swing, u, torch.zeros_like(u))
    left_lower = ~left_is_higher
    phase = torch.where(left_lower, 0.5 * swing_u, 0.5 + 0.5 * swing_u)
    state.phase[env_ids] = torch.where(flat_spawn, torch.zeros_like(phase), phase)
    if not hasattr(state, "spawn_on_landing"):
        state.spawn_on_landing = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    state.spawn_on_landing[env_ids] = on_landing
    if not hasattr(state, "spawn_on_top"):
        state.spawn_on_top = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    state.spawn_on_top[env_ids] = start_is_top


def _stair_leg_joint_ids(env: ManagerBasedRlEnv, asset: Entity, side: str) -> list[int]:
    cache = env.__dict__.setdefault("_stair_leg_joint_ids", {})
    if side not in cache:
        ids = []
        for joint in ("hip_pitch", "knee", "ankle"):
            found, _ = asset.find_joints(f"^{side}_{joint}$")
            if len(found) != 1:
                raise ValueError(f"expected one {side}_{joint} joint, got {found}")
            ids.append(found[0])
        cache[side] = ids
    return cache[side]


# --- contact classification ---------------------------------------------------


def _stair_geom_ids(env: ManagerBasedRlEnv) -> dict:
    cache = getattr(env, "_stair_geom_ids", None)
    if cache is not None:
        return cache
    model = env.sim.mj_model
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)
    ]
    feet = [names.index("robot/left_foot_collision"), names.index("robot/right_foot_collision")]
    body = [
        g
        for g, name in enumerate(names)
        if name.startswith("robot/")
        and int(model.geom_contype[g]) != 0
        and g not in feet
    ]
    treads, tread_index = [], []
    for g, name in enumerate(names):
        if name.startswith(_ladder.TREAD_ENTITY_PREFIX):
            treads.append(g)
            tread_index.append(int(name.split("/")[0][len(_ladder.TREAD_ENTITY_PREFIX):]))
        elif name.startswith(_ladder.TOP_FLOOR_ENTITY):
            # the upper floor continues the last landing: feet on it count as
            # supported on the top tread (reached_top / fallen / overstep logic)
            treads.append(g)
            tread_index.append(-2)  # resolved to the last tread index below
    rails = [g for g, name in enumerate(names) if name.startswith("rail_")]
    head = [
        g for g, name in enumerate(names)
        if name.startswith("robot/") and int(model.geom_contype[g]) != 0
        and any(k in name for k in ("head", "jaw", "neck", "mouth"))
    ]
    # Trunk / leg geoms (not feet, not head): leaning these on a landing
    # block is the s16 parking pose (trunk shell + upper leg against the
    # block face, head on top) from which no foot can be lifted onto it.
    lean = [
        g for g in body
        if g not in head and "foot" not in names[g] and "ankle" not in names[g]
    ]
    terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    terrain = []
    if terrain_body >= 0:
        start = int(model.body_geomadr[terrain_body])
        terrain = list(range(start, start + int(model.body_geomnum[terrain_body])))
    if not treads:
        raise ValueError("no tread geoms found; is the stair ladder in the scene?")
    last_idx = max(i for i in tread_index if i >= 0)
    tread_index = [last_idx if i == -2 else i for i in tread_index]
    dev = env.device
    lookup = torch.full((model.ngeom,), -1, device=dev, dtype=torch.long)
    lookup[torch.tensor(treads, device=dev)] = torch.tensor(tread_index, device=dev)
    geometry = _stair_state(env).geometry if _stair_state(env) is not None else _ladder.LADDER_GEOMETRY
    landings = [g for g, i in zip(treads, tread_index) if geometry.is_landing(i)]
    cache = {
        "feet": torch.tensor(feet, device=dev, dtype=torch.int32),
        "body": torch.tensor(body, device=dev, dtype=torch.int32),
        "treads": torch.tensor(treads, device=dev, dtype=torch.int32),
        "tread_index": lookup,
        "ladder": torch.tensor(treads + rails, device=dev, dtype=torch.int32),
        "head": torch.tensor(head, device=dev, dtype=torch.int32),
"lean": torch.tensor(lean, device=dev, dtype=torch.int32),
        "landings": torch.tensor(landings, device=dev, dtype=torch.int32),
        "terrain": torch.tensor(terrain, device=dev, dtype=torch.int32),
    }
    env._stair_geom_ids = cache
    return cache


def _stair_contacts(env: ManagerBasedRlEnv) -> dict:
    """Per-step classification of robot/ladder contacts (cached per step)."""
    step = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_stair_contact_cache", None)
    if cache is not None and cache.get("step") == step:
        return cache
    n, dev = env.num_envs, env.device
    ids = _stair_geom_ids(env)
    foot_support = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    foot_on_floor = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    foot_tread = torch.full((n, 2), -1, dtype=torch.long, device=dev)
    body_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    head_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    lean_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    count = int(env.sim.data.nacon[0].item())
    if count > 0:
        pairs = env.sim.data.contact.geom[:count]
        world = env.sim.data.contact.worldid[:count].long()
        dist = env.sim.data.contact.dist[:count]
        normal_z = env.sim.data.contact.frame[:count, 0, 2]
        g0, g1 = pairs[:, 0], pairs[:, 1]
        touching = dist <= 0.0
        for slot in range(2):
            foot = ids["feet"][slot]
            f0 = (g0 == foot) & torch.isin(g1, ids["treads"])
            f1 = (g1 == foot) & torch.isin(g0, ids["treads"])
            # Normal points geom0 -> geom1; flip so it points tread -> foot.
            tread_to_foot = torch.where(f1, normal_z, -normal_z)
            support = touching & (f0 | f1) & (tread_to_foot >= 0.5)
            if bool(support.any()):
                foot_support[world[support], slot] = True
                tread_geom = torch.where(f0, g1, g0).long()
                foot_tread[world[support], slot] = ids["tread_index"][tread_geom[support]]
            floor = touching & (
                ((g0 == foot) & torch.isin(g1, ids["terrain"]))
                | ((g1 == foot) & torch.isin(g0, ids["terrain"]))
            )
            if bool(floor.any()):
                foot_on_floor[world[floor], slot] = True
        body = touching & (
            (torch.isin(g0, ids["body"]) & torch.isin(g1, ids["ladder"]))
            | (torch.isin(g1, ids["body"]) & torch.isin(g0, ids["ladder"]))
        )
        if bool(body.any()):
            body_touch[world[body]] = True
        # Head on a *landing block* only: on a steep ladder the head brushes
        # the mini treads ahead during normal climbing (s5-s7: a global head
        # penalty broke the gait), but resting the head on the solid landing
        # is the tripod stall (s4-s9).
        head = touching & (
            (torch.isin(g0, ids["head"]) & torch.isin(g1, ids["landings"]))
            | (torch.isin(g1, ids["head"]) & torch.isin(g0, ids["landings"]))
        )
        if bool(head.any()):
            head_touch[world[head]] = True
        lean = touching & (
            (torch.isin(g0, ids["lean"]) & torch.isin(g1, ids["landings"]))
            | (torch.isin(g1, ids["lean"]) & torch.isin(g0, ids["landings"]))
        )
        if bool(lean.any()):
            lean_touch[world[lean]] = True
    cache = {
        "step": step,
        "foot_support": foot_support,
        "foot_on_floor": foot_on_floor,
        "foot_tread": foot_tread,
        "body_touch": body_touch,
        "head_touch": head_touch,
        "lean_touch": lean_touch,
    }
    env._stair_contact_cache = cache
    return cache


def _stair_foot_sites(env: ManagerBasedRlEnv, asset: Entity) -> list[int]:
    cache = getattr(env, "_stair_foot_site_ids", None)
    if cache is None:
        cache = [asset.find_sites(name)[0][0] for name in ("left_foot", "right_foot")]
        env._stair_foot_site_ids = cache
    return cache


# --- observations (fill the 4-D head and 6-D body command slots) ---------------


def _stair_foot_target_info(
    env: ManagerBasedRlEnv, asset: Entity, min_rise: float = 0.006
) -> dict:
    """Per foot: index of the next same-side tread above it and the vector to
    its landing point (world frame); cached per step."""
    step = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_stair_target_cache", None)
    if cache is not None and cache.get("step") == step:
        return cache
    state = _stair_state(env)
    g = state.geometry
    sites = _stair_foot_sites(env, asset)
    num = g.num_treads
    arange = torch.arange(num, device=env.device)
    index = torch.full((env.num_envs, 2), num, device=env.device, dtype=torch.long)
    vec = torch.zeros(env.num_envs, 2, 3, device=env.device)
    for slot, side in enumerate((1.0, -1.0)):
        foot = asset.data.site_pos_w[:, sites[slot], :]
        side_ok = (state.tread_side == side) | (state.tread_side == 0.0)
        above = state.tread_top > (foot[:, 2:3] + min_rise)
        candidate = torch.where(above & side_ok[None, :], arange[None, :], num)
        idx = candidate.min(dim=1).values
        clamped = idx.clamp_max(num - 1)
        centre = state.tread_centre.gather(1, clamped[:, None, None].expand(-1, 1, 3)).squeeze(1)
        top = state.tread_top.gather(1, clamped[:, None]).squeeze(1)
        # Landings (staircase mode) are targeted just past their nose rather
        # than at their (far forward) box centre.
        txy = centre[:, :2]
        tz = top + 0.003
        if g.landing_every > 0 and hasattr(state, "tread_target_xy"):
            landing = torch.tensor([g.is_landing(i) for i in range(num)], device=env.device)
            txy_l = state.tread_target_xy.gather(1, clamped[:, None, None].expand(-1, 1, 2)).squeeze(1)
            if g.landing_target_per_side:
                # The landing is full width: aim each foot at its own side (the
                # lateral position of a regular tread of that side) instead of
                # the centreline, so the landing step is a plain forward
                # step and not a converging one (landing study 2026-09-04).
                lat = side * (0.5 * g.center_gap_m + 0.5 * g.side_width_m)
                tyaw = state.tread_yaw.gather(1, clamped[:, None]).squeeze(1)
                txy_l = txy_l + lat * torch.stack((-torch.sin(tyaw), torch.cos(tyaw)), dim=-1)
            is_l = landing[clamped]
            txy = torch.where(is_l[:, None], txy_l, txy)
            tz = torch.where(is_l, top + g.landing_target_up_m, tz)
        if getattr(env, "_floor_desk", False):
            desk_xy=state.tread_target_xy.gather(1,clamped[:,None,None].expand(-1,1,2)).squeeze(1).clone()
            desk_xy[:,1]+=side*.042
            txy=torch.where((clamped>=30)[:,None],desk_xy,txy)
        target = torch.stack((txy[:, 0], txy[:, 1], tz), dim=-1)
        index[:, slot] = idx
        vec[:, slot] = torch.nan_to_num(target - foot, nan=0.0)
    valid = index < num
    if g.landing_every > 0 and getattr(g, "landing_step_gate", False):
        # s18 lesson: from the (4, 5) stance the policy lifted the higher foot
        # (on 5) straight for the landing while standing on 4 - a 7 cm rise
        # from a low support; the toe caught the landing's nose and the duck
        # pitched over.  Offer the landing step only when the other foot
        # stands on (or last stood on) the tread right below the landing, so
        # the only target left from (4, 5) is the plain 4 -> 6 step.
        landing = torch.tensor([g.is_landing(i) for i in range(num)], device=env.device)
        contacts = _stair_contacts(env)
        last = getattr(state, "foot_last_tread", None)
        other_tread = contacts["foot_tread"].clone()
        if last is not None:
            other_tread = torch.where(other_tread >= 0, other_tread, last)
        for slot in range(2):
            idx = index[:, slot].clamp_max(num - 1)
            is_l = landing[idx] & valid[:, slot]
            support_ok = other_tread[:, 1 - slot] >= idx - 1
            gated = is_l & ~support_ok
            valid[:, slot] = valid[:, slot] & ~gated
            vec[:, slot] = torch.where(gated[:, None], torch.zeros_like(vec[:, slot]), vec[:, slot])
            index[:, slot] = torch.where(gated, torch.full_like(idx, num), index[:, slot])
    cache = {"step": step, "index": index, "valid": valid, "vec": vec}
    env._stair_target_cache = cache
    return cache


def ladder_foot_targets(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_rise: float = 0.006,
    scale: float = 0.05,
) -> torch.Tensor:
    """[fwd_L, up_L, fwd_R, up_R]: each foot's vector to its next tread.

    The target is the centre of the lowest same-side tread whose top is at
    least ``min_rise`` above the foot, expressed in the robot's yaw frame and
    scaled by ``scale``.  Zero when no tread is left above the foot.  On the
    real robot this is computed by the runtime from the known ladder geometry
    and the rung count.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    out = torch.zeros(env.num_envs, 4, device=env.device)
    if state is None:
        return out
    info = _stair_foot_target_info(env, asset, min_rise)
    yaw = _yaw_from_quat(asset.data.root_link_quat_w)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    for slot in range(2):
        vec = info["vec"][:, slot]
        valid = info["valid"][:, slot]
        fwd = vec[:, 0] * cy + vec[:, 1] * sy
        out[:, 2 * slot] = torch.where(valid, fwd / scale, 0.0)
        out[:, 2 * slot + 1] = torch.where(valid, vec[:, 2] / scale, 0.0)
    return torch.nan_to_num(out, nan=0.0).clamp(-5.0, 5.0)


def _stair_current_flight(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """Flight index (N,) the robot is climbing: the flight of the lower foot's
    next target tread (the last flight once no target is left)."""
    state = _stair_state(env)
    g = state.geometry
    info = _stair_foot_target_info(env, asset)
    idx = info["index"].min(dim=1).values.clamp_max(g.num_treads - 1)
    flights = getattr(state, "tread_flight", None)
    if flights is None:
        flights = torch.tensor([g.flight_of(i) for i in range(g.num_treads)], device=env.device)
        state.tread_flight = flights
    return flights[idx]


def _stair_path_frame(env: ManagerBasedRlEnv, asset: Entity) -> dict | None:
    """Walking-path frame while the robot crosses a landing: the flight of
    the lower foot's target is f >= 1 and the root is still behind that
    flight's base.  Returns ``on_path`` (N,), signed lateral offset ``v`` to
    the path (left positive), the path tangent yaw and the remaining path
    length to the flight base; cached per step."""
    step = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_stair_path_cache", None)
    if cache is not None and cache.get("step") == step:
        return cache
    state = _stair_state(env)
    g = state.geometry
    if g.landing_every <= 0 or g.num_flights <= 1 or not hasattr(state, "path_pts"):
        return None
    f = _stair_current_flight(env, asset)
    root = asset.data.root_link_pos_w[:, :2]
    base = state.flight_base.gather(1, f[:, None, None].expand(-1, 1, 2)).squeeze(1)
    fyaw = state.flight_yaw.gather(1, f[:, None]).squeeze(1)
    u_r, _v_r = _ladder.to_flight_frame(root, base, fyaw)
    on_path = (f >= 1) & (u_r < 0.0)
    K = state.path_pts.shape[2]
    pidx = (f - 1).clamp(0, state.path_pts.shape[1] - 1)
    pts = state.path_pts.gather(1, pidx[:, None, None, None].expand(-1, 1, K, 2)).squeeze(1)  # (N, K, 2)
    yaws = state.path_yaw.gather(1, pidx[:, None, None].expand(-1, 1, K)).squeeze(1)
    rems = state.path_rem.gather(1, pidx[:, None, None].expand(-1, 1, K)).squeeze(1)
    d = torch.nan_to_num(root[:, None, :] - pts, nan=0.0)
    k = d.norm(dim=-1).argmin(dim=1)
    p = pts.gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)
    ty = yaws.gather(1, k[:, None]).squeeze(1)
    r = rems.gather(1, k[:, None]).squeeze(1)
    dx = torch.nan_to_num(root - p, nan=0.0)
    c, sn = torch.cos(ty), torch.sin(ty)
    along = dx[:, 0] * c + dx[:, 1] * sn
    v = -dx[:, 0] * sn + dx[:, 1] * c
    remaining = (r - along).clamp_min(0.0)
    cache = {"step": step, "on_path": on_path, "v": v, "yaw": ty, "remaining": remaining, "flight": f}
    env._stair_path_cache = cache
    return cache


def ladder_geometry_obs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """[ladder distance, sin angle, cos angle, riser, tread depth, lateral offset].

    Distance is from the trunk to the ladder nose line at trunk height
    (positive = ladder ahead), scaled by 0.1 m; riser by 0.03 m; depth by
    0.05 m; lateral offset from the ladder centre line by 0.05 m.
    """
    state = _stair_state(env)
    out = torch.zeros(env.num_envs, 6, device=env.device)
    if state is None:
        return out
    asset: Entity = env.scene[asset_cfg.name]
    g = state.geometry
    root_w = asset.data.root_link_pos_w
    root = root_w - env.scene.env_origins
    if g.landing_every > 0 and hasattr(state, "flight_base"):
        # Staircase: distance to the nose line of the flight the *feet* are
        # climbing (flight of the lower foot's next target tread), in that
        # flight's frame; slot 4 carries the heading error to the flight axis
        # instead of the (constant) tread depth.  Choosing the flight by
        # trunk height made the reading jump to the next flight as soon as the
        # trunk passed the landing level, and the policy stalled there.
        f = _stair_current_flight(env, asset)
        base = state.flight_base.gather(1, f[:, None, None].expand(-1, 1, 2)).squeeze(1)
        fyaw = state.flight_yaw.gather(1, f[:, None]).squeeze(1)
        fz = state.flight_z.gather(1, f[:, None]).squeeze(1)
        u_r, v_r = _ladder.to_flight_frame(root_w[:, :2], base, fyaw)
        nose_u = (root_w[:, 2] - fz) / torch.tan(state.angle)
        robot_yaw = _yaw_from_quat(asset.data.root_link_quat_w)
        dist = nose_u - u_r
        head_yaw = fyaw
        path = _stair_path_frame(env, asset)
        if path is not None:
            # Crossing a landing: distance along the walking path to the next
            # flight's base (plus the nose offset there), heading error to the
            # path tangent, lateral offset to the path.
            dist = torch.where(path["on_path"], path["remaining"] + nose_u, dist)
            head_yaw = torch.where(path["on_path"], path["yaw"], fyaw)
            v_r = torch.where(path["on_path"], path["v"], v_r)
        out[:, 0] = dist / 0.1
        out[:, 4] = torch.sin(head_yaw - robot_yaw)
        out[:, 5] = v_r / 0.05
    else:
        nose = _ladder.nose_line_x(g, state.riser, state.angle, state.x0, getattr(state, "flat", None), root[:, 2])
        out[:, 0] = (nose - root[:, 0]) / 0.1
        out[:, 4] = g.tread_depth_m / 0.05
        out[:, 5] = (root[:, 1] - state.y0) / 0.05
    out[:, 1] = torch.sin(state.angle)
    out[:, 2] = torch.cos(state.angle)
    out[:, 3] = state.riser / 0.03
    return torch.nan_to_num(out, nan=0.0).clamp(-5.0, 5.0)


# --- rewards --------------------------------------------------------------------


def _stair_support_height(env: ManagerBasedRlEnv, state: _SimpleNamespace) -> torch.Tensor:
    """Mean over feet of the top height of the tread each foot last stood on.

    Updated from the raw contacts every step; a foot in the air keeps the
    height of its last support, a foot on the floor counts as 0.  Also
    remembers each foot's last tread index and latches ``overstep`` when a
    foot lands beyond its next same-side tread (r14 lesson: the policy
    landed the swing foot two treads too high, was paid for the extra rise,
    and hopped from the resulting three-riser stagger).
    """
    contacts = _stair_contacts(env)
    n, dev = env.num_envs, env.device
    if not hasattr(state, "foot_support_z"):
        state.foot_support_z = torch.zeros(n, 2, device=dev)
    if not hasattr(state, "foot_last_tread"):
        state.foot_last_tread = torch.full((n, 2), -1, dtype=torch.long, device=dev)
        state.overstep = torch.zeros(n, dtype=torch.bool, device=dev)
    tread = contacts["foot_tread"]
    supported = contacts["foot_support"] & (tread >= 0)
    spacing = int(state.geometry.same_side_spacing())
    too_far = supported & (tread > state.foot_last_tread + spacing)
    state.overstep |= too_far.any(dim=1)
    new_last = torch.where(supported, tread, state.foot_last_tread)
    landing_every = int(getattr(state.geometry, "landing_every", 0) or 0)
    if landing_every > 0:
        # A landing is one wide platform for both feet: while either foot stands
        # on it, both feet take the landing as their last tread, so stepping off
        # onto the next flight's first treads is not read as skipping.  c1 probe
        # (2026-09-06, real corner staircase): 95 % of floor-start climbs ended
        # as "overstep" on the tread right after a landing (8, 16, 24, 32) - the
        # foot that had not touched the landing was still compared with its
        # tread from the previous flight.
        on_landing = supported & ((tread + 1) % landing_every == 0)
        landing_idx = torch.where(on_landing, tread, torch.full_like(tread, -1)).max(dim=1).values
        has_landing = landing_idx >= 0
        new_last = torch.where(has_landing[:, None], torch.maximum(new_last, landing_idx[:, None]), new_last)
    state.foot_last_tread = new_last
    top = state.tread_top.gather(1, tread.clamp_min(0))
    state.foot_support_z = torch.where(supported, top, state.foot_support_z)
    on_floor = contacts["foot_on_floor"]
    state.foot_support_z = torch.where(on_floor, torch.zeros_like(state.foot_support_z), state.foot_support_z)
    state.foot_last_tread = torch.where(on_floor, torch.full_like(state.foot_last_tread, -1), state.foot_last_tread)
    return state.foot_support_z.mean(dim=1)


def ladder_upward_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    max_delta: float = 0.01,
    max_above_support: float = 0.125,
) -> torch.Tensor:
    """Potential-based climb progress, paid only when climbing is commanded.

    The potential is the trunk height capped at ``max_above_support`` above
    the mean height of the treads the feet are supported on.  r8 lesson: with
    the raw trunk height, the policy rose on tiptoe (ankle 48°, +24 mm) and
    held, collecting the rise without ever stepping; with the cap, only
    putting a foot on a higher tread raises the potential.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    if state is None:
        return torch.zeros_like(z)
    support = _stair_support_height(env, state) + env.scene.env_origins[:, 2]
    potential = torch.minimum(z, support + max_above_support)
    if not hasattr(state, "prev_potential"):
        state.prev_potential = potential.clone()
    delta = torch.nan_to_num(potential - state.prev_potential, nan=0.0).clamp(-max_delta, max_delta)
    state.prev_potential = potential.clone()
    state.max_rise = torch.maximum(state.max_rise, z - state.spawn_root_z)
    climb_cmd = env.command_manager.get_command(command_name)[:, 0]
    return delta * (climb_cmd > 1e-4).float()


def ladder_climb_velocity_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    std: float = 0.02,
    tau_s: float = 0.3,
) -> torch.Tensor:
    """Gaussian tracking of the commanded vertical trunk speed (EMA-filtered).

    The twist ``vx`` slot carries the climb command: exact zero means hold
    still on the ladder.  Requires at least one foot on a tread so a robot
    standing on the floor cannot farm the hold reward.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    if state is None:
        return torch.zeros_like(vz)
    alpha = env.step_dt / max(tau_s, env.step_dt)
    state.vz_ema = state.vz_ema + alpha * (vz - state.vz_ema)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    score = torch.exp(-((state.vz_ema - cmd) / std) ** 2)
    on_tread = _stair_contacts(env)["foot_support"].any(dim=1)
    return score * on_tread.float()


def ladder_stance_composite(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    std: float = 0.035,
    tau_s: float = 0.3,
    upright_std: float = 0.45,
    forward_lean_allow: float = 0.55,
) -> torch.Tensor:
    """Multiplicative stance quality: any-foot support x upright x climb-speed tracking.

    ``upright`` penalizes lateral and backward tilt; forward lean (body-frame
    gravity x > 0, i.e. toward the ladder) is free up to ``forward_lean_allow``
    (sin of the angle, 0.55 ≈ 33°).  r6 lesson: with a symmetric upright
    factor every fall was backward or sideways and none forward; a climber on
    a 45-68° ladder has to lean into it.

    r3 lesson: additive support/upright/alive terms paid the same whether or
    not the climb command was obeyed, so holding still under a climb command
    kept most of the return at zero fall risk.  As a product, a hold command
    (zero) pays in full at rest, while under a climb command standing still
    earns only the tracking factor and the full value requires climbing.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    if state is None:
        return torch.zeros_like(vz)
    alpha = env.step_dt / max(tau_s, env.step_dt)
    state.vz_ema = state.vz_ema + alpha * (vz - state.vz_ema)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    track = torch.exp(-((state.vz_ema - cmd) / std) ** 2)
    support = _stair_contacts(env)["foot_support"].float().max(dim=1).values
    gravity = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.0)
    backward = (-gravity[:, 0]).clamp_min(0.0)
    too_far_forward = (gravity[:, 0] - forward_lean_allow).clamp_min(0.0)
    tilt_sq = gravity[:, 1] ** 2 + backward ** 2 + too_far_forward ** 2
    upright = torch.exp(-tilt_sq / (upright_std * upright_std))
    return track * support * upright


def ladder_swing_overshoot_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    clearance: float = 0.025,
) -> torch.Tensor:
    """Self-negating cost (<= 0) for a foot raised above its next tread top + clearance.

    The ceiling for each foot is the tread it last stood on plus the same-side
    spacing (two risers) plus ``clearance``, so it does not move when the
    foot passes the next tread.  r10 lesson: the first step attempts flung the
    swing foot 10-13 cm up and threw the trunk backward.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None or not hasattr(state, "foot_support_z"):
        return torch.zeros(env.num_envs, device=env.device)
    sites = _stair_foot_sites(env, asset)
    foot_z = asset.data.site_pos_w[:, sites, 2] - env.scene.env_origins[:, 2:3]
    spacing = float(state.geometry.same_side_spacing())
    ceiling = state.foot_support_z + (spacing * state.riser)[:, None] + 0.003 + clearance
    above = (foot_z - ceiling).clamp_min(0.0)
    return -torch.nan_to_num(above, nan=0.0).sum(dim=1)


def ladder_feet_support(env: ManagerBasedRlEnv, mode: str = "any") -> torch.Tensor:
    """Feet supported from above by a tread: ``any`` (1 if at least one foot,
    so single support during a step is not taxed) or ``fraction``."""
    support = _stair_contacts(env)["foot_support"].float()
    if mode == "fraction":
        return support.mean(dim=1)
    return support.max(dim=1).values


def ladder_foot_target_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    max_delta: float = 0.01,
    lower_foot_only: bool = True,
) -> torch.Tensor:
    """Potential-based shaping: per-step decrease of each foot's distance to
    its next-tread landing point, paid only under a climb command.

    Unfarmable: it pays the change in distance, and a step is skipped for a
    foot whose target index changed (landing switches the target to the tread
    above, which must not be a jackpot or a penalty).
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    info = _stair_foot_target_info(env, asset)
    dist = info["vec"].norm(dim=-1)  # (N, 2)
    if not hasattr(state, "prev_target_dist"):
        state.prev_target_dist = dist.clone()
        state.prev_target_index = info["index"].clone()
        state.target_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    same = (info["index"] == state.prev_target_index) & info["valid"]
    delta = (state.prev_target_dist - dist).clamp(-max_delta, max_delta)
    # Only the lower foot should swing (r13 lesson: paying both feet let the
    # policy re-step the foot it had just landed, hopping instead of
    # alternating).  Feet at equal support height (floor) both count.
    if lower_foot_only and hasattr(state, "foot_support_z"):
        support = state.foot_support_z
        lower = support <= support.min(dim=1, keepdim=True).values + 1e-4
    else:
        lower = torch.ones_like(same)
    reward = (delta * (same & lower).float()).sum(dim=1)
    reward = torch.where(state.target_fresh, torch.zeros_like(reward), reward)
    state.prev_target_dist = dist.clone()
    state.prev_target_index = info["index"].clone()
    state.target_fresh[:] = False
    climb_cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.nan_to_num(reward, nan=0.0) * (climb_cmd > 1e-4).float()


def ladder_path_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    max_delta: float = 0.01,
) -> torch.Tensor:
    """Potential-based shaping for the walk across a landing: per-step
    decrease of the remaining walking-path length, paid while on the path
    under a climb command (nothing rises on the flat, so upward progress and
    the foot targets alone leave the crossing unpaid)."""
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    path = _stair_path_frame(env, asset)
    if path is None:
        return torch.zeros(env.num_envs, device=env.device)
    rem = torch.where(path["on_path"], path["remaining"], torch.zeros_like(path["remaining"]))
    if not hasattr(state, "prev_path_rem"):
        state.prev_path_rem = rem.clone()
        state.prev_on_path = path["on_path"].clone()
        state.path_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    delta = (state.prev_path_rem - rem).clamp(-max_delta, max_delta)
    valid = path["on_path"] & state.prev_on_path & ~state.path_fresh
    reward = torch.where(valid, delta, torch.zeros_like(delta))
    state.prev_path_rem = rem
    state.prev_on_path = path["on_path"].clone()
    state.path_fresh[:] = False
    climb_cmd = env.command_manager.get_command(command_name)[:, 0]
    return torch.nan_to_num(reward, nan=0.0) * (climb_cmd > 1e-4).float()


def ladder_head_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Self-negating cost (-1 per step) while the head, jaw or neck touches a
    landing block.  s4 lesson: on the staircase the policy parked its heavy
    head on the landing block and stood on treads 4/5 as a tripod."""
    return -_stair_contacts(env)["head_touch"].float()


def ladder_tread_stall_penalty(env: ManagerBasedRlEnv, stall_s: float = 2.0) -> torch.Tensor:
    """Self-negating cost (-1 per step) while a foot stands on a mini tread
    (not a landing, not the floor) and the highest supported tread index has
    not increased for ``stall_s`` seconds.  s15-s17 lesson: natural arrivals
    parked on treads 4/5 below the landing (58-63 % of floor starts, every
    level, even 31 % of static approach starts) because the stance term paid
    the park and the lean penalty was dodged by hovering; on mini treads a
    climber must step up about once a second, so a park pays negative."""
    state = _stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    g = state.geometry
    contacts = _stair_contacts(env)
    foot_tread = contacts["foot_tread"]
    tread = foot_tread.clamp_min(0)
    if not hasattr(state, "landing_mask"):
        state.landing_mask = torch.tensor([g.is_landing(i) for i in range(g.num_treads)], device=env.device)
    on_mini = (foot_tread >= 0) & ~state.landing_mask[tread]
    best = torch.where(foot_tread >= 0, foot_tread, torch.full_like(foot_tread, -1)).max(dim=1).values
    if not hasattr(state, "stall_best_tread"):
        state.stall_best_tread = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        state.stall_progress_step = env.episode_length_buf.clone()
        state.stall_counter = -1
    counter = int(env.common_step_counter)
    if counter != state.stall_counter:
        state.stall_counter = counter
        fresh = env.episode_length_buf <= 1
        state.stall_best_tread = torch.where(fresh, best, state.stall_best_tread)
        state.stall_progress_step = torch.where(fresh, env.episode_length_buf, state.stall_progress_step)
        progressed = best > state.stall_best_tread
        state.stall_best_tread = torch.where(progressed, best, state.stall_best_tread)
        state.stall_progress_step = torch.where(progressed, env.episode_length_buf, state.stall_progress_step)
    stall_steps = max(1, int(round(stall_s / env.step_dt)))
    stalled = on_mini.any(dim=1) & ((env.episode_length_buf - state.stall_progress_step) > stall_steps)
    return -stalled.float()


def ladder_flight_heading_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, dead_zone_deg: float = 8.0
) -> torch.Tensor:
    """Self-negating cost (-|heading error| in rad beyond ``dead_zone_deg``)
    while a foot stands on a mini tread.  Arrival study (s22, 2026-09-05):
    natural climbs reach the tread below the landing turned 40 deg (std 14)
    from the flight axis while static spawns are within 5 deg; the aligned
    stances take the landing step 78-88 % of the time, the turned arrivals
    dive head first into the block (98 %)."""
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None or not hasattr(state, "flight_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    f = _stair_current_flight(env, asset).clamp(0, state.flight_yaw.shape[1] - 1)
    fyaw = state.flight_yaw.gather(1, f[:, None]).squeeze(1)
    yaw = _yaw_from_quat(asset.data.root_link_quat_w)
    contacts = _stair_contacts(env)
    g = state.geometry
    if not hasattr(state, "landing_mask"):
        state.landing_mask = torch.tensor([g.is_landing(i) for i in range(g.num_treads)], device=env.device)
    ft = contacts["foot_tread"]
    on_mini = ((ft >= 0) & ~state.landing_mask[ft.clamp_min(0)]).any(dim=1)
    on_landing = ((ft >= 0) & state.landing_mask[ft.clamp_min(0)]).any(dim=1) & ~on_mini
    # s24: on a landing the reference is the walking path's tangent (the
    # winder turn), so the duck lines up with the next flight before it
    # steps; s23 arrived, turned and stepped off the block at an angle.
    target_yaw = fyaw
    path = _stair_path_frame(env, asset)
    if path is not None:
        target_yaw = torch.where(path["on_path"] & on_landing, path["yaw"], target_yaw)
    dyaw = torch.atan2(torch.sin(yaw - target_yaw), torch.cos(yaw - target_yaw)).abs()
    err = (dyaw - math.radians(dead_zone_deg)).clamp_min(0.0)
    return -torch.nan_to_num(err, nan=0.0) * (on_mini | on_landing).float()


def ladder_landing_lean_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Self-negating cost (-1 per step) while the trunk or a leg (not feet,
    not head) touches a landing block.  s16 lesson: natural arrivals parked on
    treads 4/5 leaning trunk, upper leg and head on the block (60 % of floor
    starts, every level); with the block as a chest-high wall no foot can be
    lifted onto it, and the stance term paid the park."""
    return -_stair_contacts(env)["lean_touch"].float()


def ladder_body_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Self-negating cost (≤ 0) for non-foot body parts touching the ladder.

    Leaning shins, knees or the trunk on the treads is allowed (it is how a
    handless climber stabilises); this term only prices it lightly.
    """
    return -_stair_contacts(env)["body_touch"].float()


def ladder_vertical_impact_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    clip: float = 0.5,
) -> torch.Tensor:
    """Self-negating cost (≤ 0) on trunk vertical velocity jumps (landing impacts)."""
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    if state is None:
        return torch.zeros_like(vz)
    jump = (vz - state.prev_root_vz).abs().clamp_max(clip)
    state.prev_root_vz = vz.clone()
    return -jump


# --- terminations ----------------------------------------------------------------


def ladder_fallen(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    min_trunk_above_feet: float = 0.055,
    min_trunk_height: float = 0.06,
) -> torch.Tensor:
    """Collapsed: trunk too close to the lower foot, or near the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    sites = _stair_foot_sites(env, asset)
    trunk_z = asset.data.root_link_pos_w[:, 2]
    foot_z = asset.data.site_pos_w[:, sites, 2].min(dim=1).values
    rel = trunk_z - env.scene.env_origins[:, 2]
    return (trunk_z - foot_z < min_trunk_above_feet) | (rel < min_trunk_height)


def ladder_off_side(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_lateral: float = 0.10,
) -> torch.Tensor:
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    root_w = asset.data.root_link_pos_w
    y = root_w[:, 1] - env.scene.env_origins[:, 1]
    if state is None:
        return torch.zeros_like(y, dtype=torch.bool)
    g = state.geometry
    if g.landing_every > 0 and hasattr(state, "flight_base"):
        f = _stair_current_flight(env, asset)
        base = state.flight_base.gather(1, f[:, None, None].expand(-1, 1, 2)).squeeze(1)
        fyaw = state.flight_yaw.gather(1, f[:, None]).squeeze(1)
        _u, v = _ladder.to_flight_frame(root_w[:, :2], base, fyaw)
        path = _stair_path_frame(env, asset)
        if path is not None:
            v = torch.where(path["on_path"], path["v"], v)
        return v.abs() > max_lateral
    return (y - state.y0).abs() > max_lateral


def ladder_foot_fling(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    clearance: float = 0.025,
    margin: float = 0.05,
) -> torch.Tensor:
    """Terminate when a foot is flung ``margin`` above its swing ceiling.

    Hard state gate (AGENTS.md): a step is a foot moving to the next
    same-side tread, not a leg thrown 10 cm into the air; the r12 policy
    still flung the swing foot despite the overshoot cost.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None or not hasattr(state, "foot_support_z"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    sites = _stair_foot_sites(env, asset)
    foot_z = asset.data.site_pos_w[:, sites, 2] - env.scene.env_origins[:, 2:3]
    spacing = float(state.geometry.same_side_spacing())
    ceiling = state.foot_support_z + (spacing * state.riser)[:, None] + 0.003 + clearance
    return ((foot_z - ceiling) > margin).any(dim=1)


def ladder_overstep(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Terminate when a foot landed beyond its next same-side tread (latched)."""
    state = _stair_state(env)
    if state is None or not hasattr(state, "overstep"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    _stair_support_height(env, state)  # keep the latch current even if rewards ran first
    return state.overstep


def ladder_airborne(
    env: ManagerBasedRlEnv,
    grace_steps: int = 3,
) -> torch.Tensor:
    """Terminate hopping: both feet off every tread and the floor for too long."""
    contacts = _stair_contacts(env)
    grounded = (contacts["foot_support"] | contacts["foot_on_floor"]).any(dim=1)
    if not hasattr(env, "_stair_airborne_steps"):
        env._stair_airborne_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._stair_airborne_steps = torch.where(
        grounded, torch.zeros_like(env._stair_airborne_steps), env._stair_airborne_steps + 1
    )
    return env._stair_airborne_steps > grace_steps


def ladder_landing_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """One-shot (per landing, per episode) when a foot stands on a landing
    tread with the robot upright: the step the staircase runs never learned
    (s2-s11 froze or fell two treads below).  Rate-limited by construction."""
    state = _stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    g = state.geometry
    if g.landing_every <= 0:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(state, "landing_paid"):
        state.landing_paid = torch.zeros(env.num_envs, g.num_treads, dtype=torch.bool, device=env.device)
        state.landing_mask = torch.tensor([g.is_landing(i) for i in range(g.num_treads)], device=env.device)
    contacts = _stair_contacts(env)
    tread = contacts["foot_tread"].clamp_min(0)  # (N, 2)
    on = (contacts["foot_tread"] >= 0) & state.landing_mask[tread]
    asset: Entity = env.scene[asset_cfg.name]
    upright = torch.acos((-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)) < math.radians(35.0)
    # s27: pay only when the OTHER foot stands on (or last stood on) the tread
    # right below the landing, consistent with the landing-step gate; s26
    # stepped onto the top landing from three treads below (a 7 cm split)
    # and fell trying to bring the trailing foot up.
    last = getattr(state, "foot_last_tread", None)
    other_tread = contacts["foot_tread"].clone()
    if last is not None:
        other_tread = torch.where(other_tread >= 0, other_tread, last)
    pay = torch.zeros(env.num_envs, device=env.device)
    for slot in range(2):
        idx = tread[:, slot]
        other_ok = other_tread[:, 1 - slot] >= idx - 1
        fresh = on[:, slot] & upright & other_ok & ~state.landing_paid.gather(1, idx[:, None]).squeeze(1)
        state.landing_paid[torch.arange(env.num_envs, device=env.device)[fresh], idx[fresh]] = True
        pay = pay + fresh.float()
    return pay.clamp_max(1.0)


def ladder_reached_top(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.05,
    hold_s: float = 0.5,
) -> torch.Tensor:
    """Success end of episode (registered with time_out=True)."""
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    if state is None:
        return torch.zeros_like(z, dtype=torch.bool)
    top = state.tread_top[:, -1]
    if state.geometry.landing_every > 0:
        # s29: the staircase gait carries the trunk 10-13 cm above the feet,
        # so "HOME height + margin" (17 cm) never fired (0 reached_top in
        # every staircase run).  Success = both feet supported on the top
        # landing, upright, trunk above the landing.
        c = _stair_contacts(env)
        on_top = (c["foot_tread"] == state.geometry.num_treads - 1).all(dim=1)
        tilt = torch.acos((-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        # s32: "standing" also means still (root speed / spin gates) - the
        # s31 0.5 s hold never fired (a hop up then over-run never stands);
        # a shorter hold with a stillness gate rewards the first real stops
        # while a hop-and-drop (fast at touchdown) still pays nothing.
        v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
        w = torch.nan_to_num(asset.data.root_link_ang_vel_w, nan=0.0).norm(dim=-1)
        # s35 lesson: v < 0.15 / w < 2 held 10 steps by the MEAN policy in
        # probes (100 %) but almost never under PPO's action noise in
        # training (reached_top 0.04 per window vs head_touch 27).  A
        # standing duck with noise: ~0.2 m/s, ~3 rad/s; a hop-and-drop
        # touchdown: ~0.9 m/s, ~9 rad/s (s32 probe).
        ok = on_top & (tilt < math.radians(45.0)) & (z > top + margin) & (v < 0.35) & (w < 4.5)
        # s30 lesson: an instant +100 on arrival was a jackpot (episodes 13
        # steps, 165 head touches / 79 falls per window: hop up and drop).
        # Success needs ``hold_s`` of standing there (counter advanced once
        # per step; shared by the termination and the bonus).
        hold_steps = max(1, int(round(hold_s / env.step_dt)))
        step = int(getattr(env, "common_step_counter", -1))
        if not hasattr(state, "top_hold"):
            state.top_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            state.top_hold_step = -1
        if state.top_hold_step != step:
            # s38: tolerant counter (+1 on a good step, -1 on a bad one, floored
            # at 0) - PPO's action noise breaks a run of consecutive good
            # steps; a stand that is good 70 % of the time still gets there.
            state.top_hold = torch.where(ok, state.top_hold + 1, (state.top_hold - 1).clamp_min(0))
            state.top_hold_step = step
        return state.top_hold >= hold_steps
    return z > top + _ladder.HOME_TRUNK_ABOVE_SITE_M + margin


def ladder_top_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, margin: float = 0.05, hold_s: float = 0.5) -> torch.Tensor:
    """s30: one-shot pay on the success end (both feet on the top landing,
    upright, trunk above it - the ``reached_top`` condition, which truncates
    the episode on the same step).  s29 stepped onto the top platform and
    over-ran into a face-plant: nothing paid for stopping upright there."""
    return ladder_reached_top(env, asset_cfg, margin, hold_s).float() / env.step_dt


def ladder_head_touch(env: ManagerBasedRlEnv) -> torch.Tensor:
    """s29 termination: the head, jaw or neck touching a landing block ends
    the episode.  The learned landing crossing was a head-first lunge that
    catches the next flight (s28 probe: head contacts 6 % of the steps at
    the first landing, 0 % on every ordinary tread); at the top landing
    there is nothing to catch and the duck face-plants.  A hard gate, not a
    penalty (the s16-s19 head penalty x10 was dodged)."""
    return _stair_contacts(env)["head_touch"]


# --- curriculum ------------------------------------------------------------------


def ladder_level_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    num_levels: int,
    promote_rise_m: float = 0.05,
    demote_rise_m: float = 0.01,
    early_fraction: float = 0.6,
) -> torch.Tensor:
    """Adaptive per-environment geometry level (terrain-levels style).

    Runs at reset before the state is overwritten: an environment that rose
    ``promote_rise_m`` above its spawn *and* did not end early (a fall after
    mounting a few treads is not a success) moves to a steeper, taller
    ladder; one that fell early without rising ``demote_rise_m`` moves down.
    """
    state = _stair_state(env)
    if state is None or len(env_ids) == 0:
        return torch.tensor(0.0)
    rise = state.max_rise[env_ids]
    early = env.episode_length_buf[env_ids].float() < early_fraction * env.max_episode_length
    level = state.level[env_ids]
    # s34 lesson: a spawn ON the top landing cannot rise and ends early
    # when the stop fails, so it demoted every env it touched (level 2.4 ->
    # 0.8 in 500 iterations).  Those episodes leave the level alone.
    skip = state.spawn_on_top[env_ids] if hasattr(state, "spawn_on_top") else torch.zeros_like(early)
    level = torch.where((rise >= promote_rise_m) & ~early & ~skip, level + 1, level)
    level = torch.where((rise < demote_rise_m) & early & ~skip, level - 1, level)
    state.level[env_ids] = level.clamp(0, num_levels - 1)
    return state.level.float().mean()


def ladder_mean_level(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _stair_state(env)
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    return state.level.float()


# --- phase-based reference gait ------------------------------------------------
#
# r15 lesson: after fifteen rounds the policy learned single steps but not
# alternation, precise landing or the trunk shift.  A kinematic reference
# cycle built from the same leg table as the spawns (support leg loaded,
# swing foot on an arc to the next same-side tread, trunk moving forward/up)
# gives PPO a dense direction for all three.  Phase advances at the
# commanded climb speed and is exposed in the two spare twist slots.


STAIR_REF_ROLL_AMP = 0.0  # rad of hip roll leaning the trunk over the support foot
STAIR_REF_ROLL_SIGN = 1.0


def _stair_reference_roll(phase: torch.Tensor) -> torch.Tensor:
    """Hip-roll offset (same value for both hips) that leans the trunk over the
    support foot: ramps in during the first 20% of each half cycle, holds, and
    reverses sign when the support leg swaps.  Positive = lean toward the
    right foot (support while the left swings) for STAIR_REF_ROLL_SIGN = 1."""
    left_swings = phase < 0.5
    u = torch.where(left_swings, 2.0 * phase, 2.0 * phase - 1.0).clamp(0.0, 1.0)
    ramp = (u / 0.2).clamp(0.0, 1.0)
    side = torch.where(left_swings, 1.0, -1.0)
    return STAIR_REF_ROLL_SIGN * STAIR_REF_ROLL_AMP * side * ramp


def _stair_reference_offsets(
    env: ManagerBasedRlEnv, state: _SimpleNamespace, phase: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right (hip_pitch, knee, ankle) offsets for a phase in [0, 1).

    First half of the cycle: the left foot swings; second half: the right.
    Same parametrization as the mid-swing spawn (``reset_stair_ladder``).
    """
    riser, angle = state.riser, state.angle
    run = riser / torch.tan(angle)
    spacing = float(state.geometry.same_side_spacing())
    left_swings = phase < 0.5
    u = torch.where(left_swings, 2.0 * phase, 2.0 * phase - 1.0).clamp(0.0, 1.0)
    trunk_dx = u * run
    trunk_dz = u * 0.5 * riser
    support = _leg_offsets_bilinear(env, riser - trunk_dz, run - trunk_dx)
    arc_dx, arc_dz = _stair_swing_arc(u, riser, run, state.geometry)
    swing = _leg_offsets_bilinear(env, arc_dz - trunk_dz, arc_dx - trunk_dx)
    # Left-leg table; right leg mirrors (negated) as in the reset.
    left = torch.where(left_swings[:, None], swing, support)
    right = -torch.where(left_swings[:, None], support, swing)
    return left, right


def _stair_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = _stair_state(env)
    if state is None or not hasattr(state, "phase"):
        return torch.zeros(env.num_envs, device=env.device)
    return state.phase


def ladder_command_with_phase(
    env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
    """[climb speed command, sin(2*pi*phase), cos(2*pi*phase)] in the twist slot.

    Runtime contract: vx = climb speed (0 = hold); the two remaining slots
    carry the gait clock, advanced by the runtime at vx / (2 * riser) per
    second while climbing and frozen while holding.
    """
    cmd = env.command_manager.get_command(command_name)[:, 0]
    phase = _stair_phase(env)
    return torch.stack(
        (cmd, torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=-1
    )


def ladder_gait_foot_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    std: float = 0.025,
) -> torch.Tensor:
    """Task-space reference: the swing foot follows an arc from its last tread
    to the next same-side tread; the support foot stays on its tread.

    r16 lesson: joint-space tracking alone was satisfied by bobbing in place
    with both feet down.  Paid only under a climb command; a foot whose last
    support is the floor is not tracked.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None or not hasattr(state, "phase") or not hasattr(state, "foot_last_tread"):
        return torch.zeros(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    phase = state.phase
    left_swings = phase < 0.5
    u = torch.where(left_swings, 2.0 * phase, 2.0 * phase - 1.0).clamp(0.0, 1.0)
    riser, angle = state.riser, state.angle
    run = riser / torch.tan(angle)
    spacing = float(state.geometry.same_side_spacing())
    sites = _stair_foot_sites(env, asset)
    feet = asset.data.site_pos_w[:, sites, :]
    last = state.foot_last_tread
    valid = last >= 0
    idx = last.clamp_min(0)
    start_xy = state.tread_centre.gather(1, idx[:, :, None].expand(-1, -1, 3))[:, :, :2]
    start_z = state.tread_top.gather(1, idx) + 0.003
    swing_mask = torch.stack((left_swings, ~left_swings), dim=1).float()
    arc_dx, arc_dz = _stair_swing_arc(u, riser, run, state.geometry)
    dx = swing_mask * arc_dx[:, None]
    dz = swing_mask * arc_dz[:, None]
    ref = torch.stack((start_xy[:, :, 0] + dx, start_xy[:, :, 1], start_z + dz), dim=-1)
    dist_sq = (feet - ref).square().sum(dim=-1)
    score = torch.exp(-dist_sq / (std * std))
    score = torch.where(valid, score, torch.ones_like(score))
    return torch.nan_to_num(score.mean(dim=1), nan=0.0) * (cmd > 1e-4).float()


def ladder_gait_reference_tracking(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    std: float = 0.3,
) -> torch.Tensor:
    """Advance the gait phase and pay Gaussian tracking of the reference leg pose.

    Phase rate is cmd / (2 * riser) cycles per second (one step per half
    cycle raises the trunk one riser).  Under a hold command the phase is
    frozen and the reference is the current stance.
    """
    state = _stair_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(state, "phase"):
        state.phase = torch.zeros(env.num_envs, device=env.device)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    rate = cmd / (2.0 * state.riser)
    state.phase = torch.remainder(state.phase + rate * env.step_dt, 1.0)
    left_ref, right_ref = _stair_reference_offsets(env, state, state.phase)
    joint_pos = asset.data.joint_pos
    default = asset.data.default_joint_pos
    left_ids = _stair_leg_joint_ids(env, asset, "left")
    right_ids = _stair_leg_joint_ids(env, asset, "right")
    err_left = joint_pos[:, left_ids] - default[:, left_ids] - left_ref
    err_right = joint_pos[:, right_ids] - default[:, right_ids] - right_ref
    mse = torch.cat((err_left, err_right), dim=1).square().mean(dim=1)
    score = torch.exp(-mse / (std * std))
    return torch.nan_to_num(score, nan=0.0) * (cmd > 1e-4).float()


def joint_accelerations_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize joint accelerations using L2 squared norm.
    Joint accelerations are computed using finite differences of joint velocities.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared joint accelerations
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get current joint velocities
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Get previous joint velocities (stored in asset data)
    # Note: This assumes the environment stores previous joint velocities
    if not hasattr(asset.data, '_prev_joint_vel'):
        # Initialize on first call
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Compute joint accelerations using finite differences
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # Store current velocities for next step
    asset.data._prev_joint_vel = joint_vel.clone()

    # Return L2 squared norm
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of leg actions (action_t - action_{t-1}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # Get current and previous actions for leg joints only
    # Actions are stored in env (assuming the action is available)
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    # Get the joint position action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions'):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize the rate of change of neck actions (action_t - action_{t-1}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    # Get current and previous actions for neck joints only
    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions'):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get leg joint indices
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, '_prev_leg_actions_for_acc'):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck action accelerations (action_t - 2*action_{t-1} + action_{t-2}).
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    # Get neck joint indices
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, 'action_manager'):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, '_prev_neck_actions_for_acc'):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """Per-env float mask: 1.0 where the robot counts as FALLEN — trunk height
    below `gate_z_below` OR tilt beyond `gate_tilt_above_deg`. Used to gate the
    recovery rewards so they only steer while actually fallen and contribute
    exactly zero during clean walking (no walk tax / bounce farming)."""
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(
    env: ManagerBasedRlEnv,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **air_time_kwargs,
) -> torch.Tensor:
    """velocity template feet_air_time, zeroed while FALLEN (tilt > gate).

    velstand: a robot lying on its trunk can still tap its feet rhythmically
    through the air-time window — the observed "lies there shaking a leg"
    exploit. Air time is only meaningful upright.
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time
    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Potential-based upright shaping: Δcos(tilt) per step.

    Pays for PROGRESS toward upright, charges for progress toward fallen, and
    pays exactly ZERO for holding any pose — so no state can farm it (the gated
    state-reward it replaces was farmed from sitting, lying flat, and a
    head-tripod lean across three velstand runs). Potential-based shaping is
    policy-invariant (Ng et al.): it accelerates learning of recovery without
    creating new optima. A full prone→stand recovery collects Δ≈+1 total
    (× weight); a fall costs the same on the way down.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(
        1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0
    )
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # Freshly reset envs: no spurious delta from the previous episode's pose.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ceiling: float = 0.115,
) -> torch.Tensor:
    """Potential-based height shaping: Δ min(trunk z, ceiling) per step.

    The z-axis companion to ``upright_progress`` (velstand crouch-endpoint
    lesson): the last mile of a recovery — extending the knees out of a deep
    crouch — is mostly a HEIGHT change at modest tilt, exactly where the
    Gaussian upright/pose rewards are flat and Δcos(tilt) is tiny. Rising pays,
    falling charges, holding pays zero, so gait bobbing nets zero and nothing
    can farm it. Capped at ``ceiling`` (just below full-stand trunk z ≈ 0.117)
    so hopping above stance height pays nothing extra.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def fallen_state_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_tilt_above_deg: float = 40.0,
    release_tilt_below_deg: float | None = None,
    release_z_above: float | None = None,
) -> torch.Tensor:
    """1.0 while FALLEN (weight it negative): a flat per-step tax on staying
    down. Without it, lying still is ~0/step while attempting recovery costs
    action-rate/torque penalties — waiting for the fallen_too_long recycle was
    the rational policy. (Penalties on bad states are safe; it's POSITIVE
    rewards gated on bad states that get farmed.)

    With ``release_*`` set, the tax has HYSTERESIS (velstand crouch-endpoint
    lesson): a fall arms it and it keeps paying until the robot is genuinely
    up (tilt < release_tilt AND z > release_z), not merely under the arming
    gate. Without it, a crouch just below the 40° gate is a zero-cost rest
    state — recoveries learned to park there instead of finishing the stand.
    Arms only on a genuine fall, so gait-cycle tilt wobble is never taxed."""
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    fallen_tilt_deg: float = 40.0,
    min_fallen_s: float = 0.5,
    up_tilt_deg: float = 25.0,
    up_z: float = 0.105,
) -> torch.Tensor:
    """One-shot bounty on a COMPLETED recovery: fires on the frame where an env
    that has been fallen (tilt > fallen_tilt for ≥ min_fallen_s) becomes
    genuinely upright (tilt < up_tilt AND trunk z > up_z). Hysteresis: re-arms
    only by being fallen again, so oscillating around the gate pays nothing.
    Gives the sparse-but-strong endpoint gradient the dense gated terms lack.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen, env._recovery_fallen_s + env.step_dt, torch.zeros_like(env._recovery_fallen_s)
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> torch.Tensor:
    """Linear reward for body uprightness — provides gradient at every tilt angle.

    Returns +1 when fully upright, 0 when horizontal (prone/supine), -1 when inverted.
    Unlike flat_orientation (Gaussian), this has non-zero gradient everywhere, so the
    robot always has a signal to rotate toward upright even when starting from prone.

    Computed as the z-component of the body's local Z-axis expressed in world frame,
    which equals R[2,2] = 1 - 2*(qx² + qy²) for quaternion [w, x, y, z].
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # Recovery-gated variant (velstand): active only while fallen, exactly
        # zero during clean walking so it can't dilute the tracking rewards.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """Gaussian reward on tilt magnitude — sharp pull toward fully vertical.

    Complements ``body_upright_linear`` (which is ``cos(tilt)`` and whose
    gradient ``sin(tilt)`` *vanishes* at the target). This Gaussian's
    gradient is non-zero near vertical and tapers as you move away, so it
    creates a strong differential pull in the regime where the linear
    version is weakest.

    Uses ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` as a tilt-squared
    proxy and applies ``exp(-tilt²/std²)``. Default std=0.1 rad ≈ 5.7°.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # ≈ 1 − cos(tilt); small-angle: tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``body_upright_gaussian`` weighted by smoothstep on trunk z.

    Full Gaussian-upright reward when ``z >= height_high``, zero when
    ``z <= height_low``, smoothstep in between. Use this when the upright
    incentive should only apply at the target standing height — otherwise
    the policy can find a "crouch low and vertical" local optimum that
    collects upright reward without ever rising.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """Trunk ``sum(ω_xy²)`` penalty gated by trunk z (and optionally tilt).

    Height-gated arrival damper: zero below ``height_low`` (ground recovery —
    flips/rolls need large trunk rotation and must stay free), full above
    ``height_high``. Same formula as mjlab's body_angular_velocity_penalty
    (world-frame ω_xy, z-rotation free) but returns the gated POSITIVE cost;
    use a negative weight.

    ``tilt_full_deg`` (optional but STRONGLY recommended): additionally gate
    by tilt — full cost only when tilt ≤ tilt_full_deg, zero when
    ≥ tilt_zero_deg, smoothstep between. LESSON (2026-07 run that broke
    front-recovery): with a height gate alone, the final straighten of a
    bent-over rise (tilt 60°→0 happening INSIDE the z gate) is itself a
    large trunk rotation — taxing it builds a reward wall right before the
    finish, and the policy parks bent-over below the gate instead. With the
    tilt gate, the approach TO vertical is free; only residual wobble
    AROUND vertical (the overshoot→tip→retry oscillation) is damped.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Smooth multiplicative goal-state score (product of three Gaussians).

    Returns ``height_score * upright_score * pose_score``, each ∈ [0, 1].
    Because the factors *multiply*, a deficiency in any one term collapses
    the whole reward — the policy can't claim 80% of this by being perfect
    on 2-of-3. Gradient is non-zero everywhere, so the score works during
    the rise (not just at the goal like a binary bonus would).

    Use to break Nash-equilibrium compromises (e.g., a "lean trunk at the
    right height" basin that satisfies the additive rewards' partial sums).
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_score = torch.exp(-((z - target_height) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def standing_success_bonus(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_tol: float,
    upright_threshold: float,
    pose_tol: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Binary bonus: 1.0 iff height, uprightness AND pose are all within tol.

    Creates a discrete goal-state attractor that gradient-based pose/upright/
    height rewards can't fully match by themselves. Surrounding compromises
    (lean trunk to balance head-forward CoM, park 1cm short of target z,
    etc.) collect partial gradient credit but ZERO bonus — the bonus is
    available only at the true goal state, so it changes the policy's
    relative preference once the rest of the rewards have brought it close.
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values  # tightest joint
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> torch.Tensor:
    """Reward upward CoM velocity to incentivize dynamic standup motion.

    Gated by height: only active while the CoM is below `max_height` (the
    standing target). Once standing, the reward is zero so the robot has no
    incentive to keep squatting to farm upward-velocity reward.

    ``max_vz`` (optional): cap the rewarded velocity. Uncapped, the reward is
    proportional to vz, which pays MORE per step for an explosive launch —
    a violent-rise incentive. With a cap, any rise ≥ max_vz earns the same,
    so the gentlest rise that reaches the cap is optimal (the |a_z| penalty
    then picks the smooth one). The bootstrap property is preserved: any
    upward motion still pays immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    com_z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # Recovery-gated (velstand): without the gate this pays for dip-and-rise
        # during gait whenever the trunk crosses max_height → bounce incentive.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """Terminate envs that have been continuously FALLEN for `max_duration_s`.

    For envs that mix walking with fall recovery (velstand): the fell_over
    termination gets disabled by curriculum so the policy can attempt recovery,
    but without a backstop a failed recovery farms recovery-reward for the whole
    20 s episode, starving the walk of data (audit: ~25% walking share). This
    gives every fall a fair recovery window, then recycles the env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # Freshly reset envs start with a clean timer.
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(
        fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s)
    )
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Terminate environments where MuJoCo produced NaN joint positions.

    MuJoCo's contact solver can overflow to NaN under extreme penetration or
    impulse (e.g. robot landing at high velocity). A NaN simulation state
    propagates into observations, corrupting the policy network weights.

    Terminating immediately resets the environment before the cascade spreads:
    - The observation returned to the runner is from the valid reset state.
    - NaN rewards are avoided on subsequent steps.

    Note: the reward at THIS terminal step may still be NaN from the simulation;
    mjlab computes rewards before resetting (see manager_based_rl_env.py step()).
    Our custom reward functions guard against NaN internally with nan_to_num,
    but standard mjlab rewards can still be NaN here. One NaN reward is
    tolerable because done=True prevents it propagating backward through GAE.

    Couvre TOUT l'état physique, pas seulement joint_pos : la divergence du
    contact fait souvent exploser le FREE-JOINT de base (position/orientation/
    vitesse) ou les ROUES passives, pas les joints actionnés. Ces quantités
    alimentent des termes d'obs critic (base_lin_vel, base_ang_vel,
    projected_gravity, wheel_vel) ; si on ne les surveille pas, l'env ne se
    reset pas et le NaN atteint l'obs → le check_nan de rsl_rl tue tout
    l'entraînement. On teste la non-finitude (NaN ET inf, l'inf devenant NaN en
    aval lors de la normalisation de projected_gravity).
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # Contact FORCES can blow up a step before qpos/qvel do: MuJoCo resolves a
    # degenerate contact into an inf/NaN impulse while the integrated state is
    # still finite. That force feeds the critic-only `foot_contact_forces` obs
    # (sign(F)*log1p(|F|)), which the state checks above do NOT cover — so the
    # env was not reset and the NaN reached the runner's check_nan, killing the
    # whole run (crash 2026-08-21, Velocity2-Rough-Backlash with hfield slopes).
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the trunk drops below ``min_height`` in world z.

    Utilisé par roller_slope comme « tombé dans le vide » : le terrain a un
    plat de sortie au bas de la rampe, donc une descente normale ne passe
    jamais sous le niveau du plat de sortie le plus bas. Choisir min_height
    en dessous de ce niveau => la terminaison ne se déclenche que si le robot
    quitte le solide et chute dans le vide. Indépendant de la géométrie exacte
    de la rampe (longueur/pente).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(
    env: ManagerBasedRlEnv,
    cap: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse d'avance vers le BAS de la pente (monde +x).

    La rampe descend en +x, donc la vitesse linéaire monde en x mesure la
    progression de descente. Plafonnée à ``cap`` m/s : encourage à se laisser
    glisser sans pousser à dévaler de plus en plus vite. Nulle si le robot
    recule/remonte (vx < 0). Sans cette récompense, l'optimum est de rester
    immobile et droit (le robot « freine » au lieu de glisser). NaN-safe.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Départ en ROULEMENT sans glissement (élan aux roues).

    Tire une vitesse d'avance v par env ; met la vitesse LINÉAIRE de base (x
    monde) = v ET la vitesse de ROTATION des 4 roues passives = v / r, donc
    ω·r = v => zéro glissement au contact. Évite l'à-coup de l'ancienne poussée
    base-seule (base qui bouge, roues immobiles = patinage brutal au 1er pas).
    À exécuter APRÈS reset_base (qui pose la base ; ne plus lui donner de
    velocity_range).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo  # (n,) vitesse avant

    # Vitesse de base (monde) : uniquement +x.
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # Rotation des 4 roues passives = v / r (positif = avant, cf. wheel_speed).
    wheel_ids = []
    for name in ("passive_LF_?wheel", "passive_LR_?wheel", "passive_RF_?wheel", "passive_RR_?wheel"):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))  # (n, 4)
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(
    env: ManagerBasedRlEnv,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """Récompense le ROULEMENT des roues vers l'avant (glisse), plafonné.

    Contrairement à descent_speed (vitesse de la BASE, qu'on peut atteindre en
    "courant"/poussant), on récompense la rotation des ROUES passives = vraie
    glisse par roulement. Indépendant de toute commande (la tâche pente a une
    commande nulle : la glisse vient de la gravité). Plafonné à ``cap_speed``
    (m/s de vitesse de roulement) -> AUCUNE incitation à accélérer au-delà ; nul
    si les roues reculent (remontée). NaN-safe.
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # Les 4 roues tournent en positif pour l'avant (cf. wheel_speed_reward).
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """
    Reward for staying alive (not terminated)

    Args:
        env: The environment

    Returns:
        Reward tensor of shape (num_envs,) - ones for all envs
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """
    Reward for keeping the center of mass within a target height range.
    Returns positive reward when in range, negative penalty when outside.

    Args:
        env: The environment
        asset_cfg: Asset configuration
        target_height_min: Minimum target height for CoM (meters)
        target_height_max: Maximum target height for CoM (meters)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Height above terrain spawn origin (world z minus terrain z).
    # env_origins[:, 2] is 0 for flat ground, so this is safe unconditionally.
    # nan_to_num: MuJoCo can produce NaN on contact instability; treat as z=0
    # so the penalty is finite (small, since 0 is near the target range).
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )

    # Reward when in range, penalty when outside
    # Use smooth penalty that increases quadratically with distance from range
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # Compute penalties for being outside range
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # Reward: +1 when in range, -squared_distance when outside
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """Cible de hauteur du tronc « en trapèze » le long de la phase [0,1).

    phase ∈ [0, hold_lo)      : descente   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): palier      height_low   (la glisse accroupie)
    phase ∈ [hold_hi, 1.0)    : remontée    height_low  -> height_high

    Args:
        phase: (B,) phase par env, dans [0, 1).
        height_low: hauteur du tronc accroupi (m).
        height_high: hauteur du tronc debout (m).
        hold_lo, hold_hi: bornes du palier bas en fraction de phase.
    Returns:
        (B,) hauteur-cible en mètres.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """Récompense gaussienne du suivi de la cible de hauteur (fonction pure).

    Décode la phase depuis [cos, sin] puis compare la hauteur mesurée à la
    cible-trapèze. Retourne exp(-((h - cible)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward principale : suit la cible de hauteur du tronc le long de la phase.

    La hauteur du CoM est calculée comme dans `com_height_target` (world z moins
    l'origine du terrain, nan->0). La phase provient de la commande GroundPick.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse avant du tronc (conserver l'élan / ne pas freiner).

    Indépendante de la commande (la commande porte la phase, pas la vitesse).
    tanh(clamp(vx, 0)/vel_ref) → sature à ~1, ne récompense jamais reculer.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def running_forward_progress_from_velocity(
    velocity_x: torch.Tensor,
    speed_cap: float = 1.2,
) -> torch.Tensor:
    """Linear forward-speed objective used by the running task.

    Unlike :func:`forward_speed_reward`, this deliberately does not saturate at
    ordinary walking speed.  Backward motion receives no reward and very large
    velocities are capped so a single physics outlier cannot become a jackpot.
    """
    if speed_cap <= 0.0:
        raise ValueError("speed_cap must be positive")
    velocity_x = torch.nan_to_num(velocity_x, nan=0.0, posinf=speed_cap, neginf=0.0)
    return torch.clamp(velocity_x, min=0.0, max=speed_cap) / speed_cap


def running_forward_progress(
    env: ManagerBasedRlEnv,
    speed_cap: float = 1.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward forward trunk speed with useful gradient above walking speeds."""
    asset: Entity = env.scene[asset_cfg.name]
    return running_forward_progress_from_velocity(
        asset.data.root_link_lin_vel_b[:, 0], speed_cap=speed_cap
    )


def running_flight_event(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    min_forward_speed: float = 0.3,
    max_tilt_deg: float = 50.0,
    min_airborne_steps: int = 3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay once when a stable, forward-moving flight phase begins.

    This is intentionally an *event*, not an airtime reward: extending an
    uncontrolled ballistic phase never increases the return.  Requiring three
    consecutive 50 Hz samples rejects one-frame contact-sensor flicker.  The
    state cache is reset on a fresh episode so spawning in the air cannot
    collect a reward.
    """
    if min_airborne_steps < 1:
        raise ValueError("min_airborne_steps must be at least one")
    sensor = env.scene[sensor_name]
    contacts = sensor.data.found.reshape(env.num_envs, -1).any(dim=-1)
    airborne = ~contacts

    air_steps = getattr(env, "_running_airborne_steps", None)
    if air_steps is None or air_steps.shape != airborne.shape:
        air_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fresh_episode = env.episode_length_buf == 0
    air_steps = torch.where(airborne, air_steps + 1, torch.zeros_like(air_steps))
    air_steps = torch.where(fresh_episode, torch.zeros_like(air_steps), air_steps)
    onset = air_steps == min_airborne_steps
    env._running_airborne_steps = air_steps

    asset: Entity = env.scene[asset_cfg.name]
    forward = torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 0], nan=0.0)
    gravity_z = torch.nan_to_num(asset.data.projected_gravity_b[:, 2], nan=0.0)
    max_tilt_cos = math.cos(math.radians(max_tilt_deg))
    stable = (-gravity_z) >= max_tilt_cos
    return (onset & stable & (forward >= min_forward_speed)).float()


def running_planar_drift_cost_from_values(
    lateral_velocity: torch.Tensor,
    yaw_rate: torch.Tensor,
    lateral_command: torch.Tensor,
    yaw_command: torch.Tensor,
    lateral_weight: float = 4.0,
) -> torch.Tensor:
    """Positive straight-line error cost; use with a negative reward weight."""
    lateral_error = torch.nan_to_num(lateral_velocity - lateral_command, nan=0.0)
    yaw_error = torch.nan_to_num(yaw_rate - yaw_command, nan=0.0)
    return yaw_error.square() + lateral_weight * lateral_error.square()


def running_planar_drift_cost(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    lateral_weight: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize body-frame lateral drift and yaw-rate command error."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return running_planar_drift_cost_from_values(
        asset.data.root_link_lin_vel_b[:, 1],
        asset.data.root_link_ang_vel_b[:, 2],
        command[:, 1],
        command[:, 2],
        lateral_weight=lateral_weight,
    )


def crouch_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose debout, 1 = pose accroupie.

    [0, descent_end)      : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1      (bas / accroupi)
    [hold_end, rise_end)  : 1 -> 0  (se lever)
    [rise_end, 1.0)       : 0       (haut / debout, repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    stand_pose: Optional[dict] = None,
):
    """(cur, target) joint tensors for the phase-interpolated crouch pose.

    Target interpolates per joint STAND <-> crouch_pose by the 4-segment blend
    b(phase) in [0,1] (0 = standing, 1 = crouch). STAND is `stand_pose` where
    given, else the model DEFAULT (HOME). Joints are resolved BY NAME so the
    passive-wheel interspersing on the roller robot never shifts an index.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)   # (B,) 0..1

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                     # (B,k)

    stand = default.clone()                                            # source pose
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor(
        [crouch_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                     # (1,k)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)           # (B,k)
    cur = asset.data.joint_pos[:, ids]                                # (B,k)
    return cur, target


def crouch_glide_pose_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    std: float = 0.4,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian match to a phase-interpolated joint pose (stand <-> crouch).

    Directive reward: tells the robot the exact joint configuration to be in at
    each phase. Standing back up (target = stand_pose) is rewarded exactly like
    crouching (target = crouch_pose) — symmetric by construction.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: Optional[dict] = None,
    stand_pose: Optional[dict] = None,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 bootstrap toward the phase-interpolated crouch pose (negative penalty).

    Constant gradient everywhere — gives the policy a direction to the target
    pose even when the Gaussian above has saturated to ~0 far from it.
    """
    cur, target = _crouch_pose_error(
        env, asset_cfg, command_name, crouch_pose or {},
        descent_end, hold_end, rise_end, stand_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pitch: float = 0.08,
    std: float = 0.1,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Léger penché AVANT du tronc pendant l'accroupi (gaté par le blend crouch).

    Contre la bascule arrière induite par la flexion rapide des hanches. Proxy de
    pitch = projected_gravity_b[:,0] (positif = vers l'avant, vérifié). La porte
    (blend) vaut 1 pendant descente+bas, 0 debout → ne biaise QUE l'accroupi.
    target_pitch petit = "de très peu".
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std ** 2)


def neck_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize neck joint velocities to keep head stable.
    Neck joints are indices 5-8 (4 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get neck joint indices (neck_pitch, head_pitch, head_yaw, head_roll).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    neck_joint_indices = list(range(5, 9))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # Return L2 squared norm of neck joint velocities
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize leg joint velocities to encourage smoother, less dynamic motion.
    Leg joints are indices 0-4 and 9-13 (10 joints total).

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get leg joint indices (left hip-ankle: 0-4, right hip-ankle: 9-13).
    # Servo view: passive_* joints (backlash, wheels) don't shift the indices.
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # Return L2 squared norm of leg joint velocities
    return torch.sum(torch.square(leg_joint_vel), dim=1)

_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """Penalize foot sites not being parallel to the ground.

    The foot site frame has Z+ pointing up when flat. We project a unit gravity
    vector (pointing down) into each foot site's local frame. When flat, gravity
    maps to [0,0,-1] in site frame (xy=0, penalty=0). Any tilt rotates Z away
    from world-up, giving nonzero xy components.

    Max value ≈ 2.0 per foot (foot fully sideways), total ≈ 4.0.

    When ``sensor_name`` is given, each foot's penalty is GATED by that foot's own
    ground contact: the airborne (swing) foot is free to tilt, only the stance
    blade is asked to stay flat (so its wheels keep gripping). Without this gate
    the penalty punishes the recovery-foot lift a stride needs — it is minimised
    by keeping BOTH blades flat on the ground, i.e. the swizzle. Assumes the site
    order (left, right) matches the sensor slot order (ankle_l_v1,
    ankle_r_v1) — both left-first in this model.

    Bug note: must normalize gravity PER ENV with dim=-1. Using torch.norm()
    without dim computes a scalar over all envs × 3 dims, making the vector
    ~1/sqrt(num_envs) in magnitude → penalty ~num_envs times too small.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    import torch.nn.functional as F

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor
        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time  # (B, N_feet)
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def feet_tiptoe_alignment(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward each foot site's local x-axis pointing downward — tiptoe stance.

    When flat, foot site x points roughly forward (horizontal). Pitching the
    foot forward (heel up, toe down) rotates x toward world -Z. We reward the
    z-component of the foot x-axis being -1 (perfectly downward).

    Per foot: alignment ∈ [-1, 1], summed over both feet ∈ [-2, 2].

    Gated on |vel_cmd_xy| > command_threshold so the policy isn't required to
    stand on tiptoes at rest — only while walking. The companion
    feet_flat_penalty is NOT used in this task; the two would fight.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N, 4) [w, x, y, z]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)  # (B, N) — z-component of local x-axis in world
    alignment = (-x_axis_z).sum(dim=-1)  # +1 per foot when pointing straight down

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """Penalize hip_pitch and knee joint velocities (L2 squared).

    Walking requires rapid oscillation of these sagittal-plane joints.
    Skating uses hip_roll laterally and glides with minimal sagittal movement.
    This penalizes the oscillation without preventing static balance adjustments.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
    pattern: str = r".*(neck|head).*",
) -> torch.Tensor:
    """Penalize neck/head joint position deviation from default (L2 squared).

    Uses find_joints() every call to avoid stale cached indices when the same
    SceneEntityCfg singleton is reused across robots with different joint layouts
    (e.g. walk robot vs rollers robot where passive wheels shift neck indices).

    ``pattern`` sélectionne les joints comptés (défaut : toute la nuque + la tête).
    La tâche spin passe un motif qui EXCLUT `head_yaw`, pour laisser la tête servir
    de volant d'inertie au lancement de la rotation.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # Exclude passive_* joints (backlash hinges also contain "neck"/"head").
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """
    Penalize actuator forces (torques) to encourage energy-efficient motion.

    Args:
        env: The environment
        asset_cfg: Asset configuration

    Returns:
        Penalty tensor of shape (num_envs,) - sum of squared actuator forces
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get actuator forces (scalar actuation in actuation space)
    actuator_forces = asset.data.actuator_force

    # Return L2 squared norm
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize rate of change in actuator torques (proxy for gearbox shock).

    Sudden torque spikes occur when the robot impacts the ground and actuators
    resist the impulse. Penalising this rate encourages soft landings and smooth
    force transitions that protect gearboxes.

    Returns the sum of squared torque differences from the previous step.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force  # (num_envs, num_actuators)

    if not hasattr(env, '_prev_actuator_forces'):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Positive reward for feet contacting the ground (0, +0.5, or +1.0).

    Uses the contact sensor's `found` field. For the feet_ground_contact sensor
    which has 2 primary foot geoms, `found` has shape (num_envs, 2) with per-foot
    binary contact. We sum and normalize to [0, 1].
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (num_envs, num_feet) or (num_envs, 1)
    if found.dim() > 1:
        found = found.sum(dim=-1)  # collapse foot dimension
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize terrain contact forces above a threshold on protected body parts.

    Used to discourage slamming the trunk shell or head into the ground during
    falls. The sensor should cover the relevant body or subtree with
    reduce='netforce'. Forces below threshold are free; above that the penalty
    grows linearly.

    Args:
        sensor_name: Name of a ContactSensorCfg with fields=("force",),
            reduce="netforce".
        threshold: Contact force (N) below which no penalty is applied.

    Returns:
        Penalty tensor (num_envs,) — N above threshold per step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force  # (num_envs, N_bodies, 3)
    total_force = forces.sum(dim=1)  # sum over bodies in the subtree
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
    bidirectional: bool = False,
) -> torch.Tensor:
    """Reward wheel spin proportional to commanded push.

    All 4 wheels spin positive for forward motion (verified visually).
    tanh saturation at vel_scale m/s equivalent prevents runaway.

    - ``bidirectional=False`` (default): forward only — reward forward spin for
      cmd_x > 0, silent otherwise (cmd_x < 0 handled by the braking reward).
    - ``bidirectional=True``: reward wheel spin in the COMMANDED direction —
      forward for cmd_x > 0, backward for cmd_x < 0 — with magnitude |cmd_x|.
      Lets cmd_x < 0 mean "go backward" instead of "brake".
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    # All 4 wheels spin positive for forward motion (verified by test_wheel_direction.py)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        # spin aligned with the command sign (fwd for +, back for -)
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",)),
) -> torch.Tensor:
    """Reward coasting: low leg-joint velocity while at target speed.

    Returns exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²).
    Both factors must be high simultaneously — robot is rewarded for being at
    target speed AND keeping its legs still (gliding), not for either alone.

    Typical values when coasting well: ~0.7–1.0.  When actively stomping at
    speed the joint_vel term suppresses the reward toward 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std ** 2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """Reward coming to a stop when cmd_x < 0 (brake commanded).

    Returns clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²).
    - Silent when cmd_x ≥ 0 (coast or push).
    - At cmd_x = -1 and vel = 0: reward = 1.0 (full stop achieved).
    - At cmd_x = -1 and vel = vel_std: reward ≈ 0.37 (strong gradient).
    vel_std=0.3 m/s gives meaningful gradient down to walking-pace speeds.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std ** 2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """
    Penalize high frequency of contact changes to encourage slower stepping.
    Tracks the number of contact state changes per second and penalizes when above threshold.

    Args:
        env: The environment
        sensor_name: Name of the contact sensor
        max_contact_changes_per_sec: Maximum allowed contact changes per second
        command_threshold: Minimum command magnitude to apply penalty

    Returns:
        Penalty tensor of shape (num_envs,) - negative when exceeding threshold
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # Check if command is above threshold
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # Initialize tracking if needed
    if not hasattr(env, '_contact_change_count'):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # Detect any contact changes (either foot)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # Increment change counter
    env._contact_change_count += contact_changed.float()

    # Update timer
    env._contact_change_timer += env.step_dt

    # Calculate current frequency (changes per second)
    # Avoid division by zero
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # Reset counter and timer every 1 second
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # Penalize when frequency exceeds maximum
    # Use quadratic penalty for frequencies above threshold
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # Update previous contacts
    env._prev_contacts_for_freq = contacts.clone()

    # Apply command threshold mask
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Ground Pick Rewards
# ==============================================================================

def mouth_ground_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.03,
    target_height: float = 0.0,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward for mouth tip approaching the ground, weighted by the approach phase.

    The command for the ground pick task is [cos(2π*phase), sin(2π*phase), 0].
    The approach phase is the first half-cycle (sin > 0, phase ∈ [0, 0.5]),
    smoothly weighted by max(0, sin(2π*phase)).

    Args:
        std: Gaussian std on mouth_tip height (m). 0.03 m gives strong gradient.
        target_height: Target z-height for the mouth tip (m). 0 = ground level.
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)

    # Approach weight: max(0, sin(2π*phase)) — peaks at 1 at phase=0.25, zero at 0 and 0.5
    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward the mouth tip x-axis being vertical (pointing down) during the approach phase.

    A perfectly perpendicular contact gives alignment=1; horizontal gives 0; pointing up gives -1.
    Weighted by max(0, sin(2π*phase)) so it only applies during the descent.
    """
    asset = env.scene[asset_cfg.name]
    # site_quat_w: (num_envs, num_sites, 4) as [w, x, y, z]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # z-component of the site x-axis in world frame (first column of rotation matrix)
    x_axis_z = 2.0 * (qx * qz - w * qy)
    # dot with [0, 0, -1]: 1 = perfectly downward, -1 = upward
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def sit_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: Optional[str] = None,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_cos_threshold: float = 0.5,
) -> torch.Tensor:
    """Positive reward for trunk-ground contact WHILE upright.

    Gated additionally on the trunk's body-frame +Z axis pointing in roughly the
    world-up direction (cosine >= ``upright_cos_threshold``, default 0.5 → up to
    60° tilt accepted). Without this gate, the policy can earn the contact
    bonus by tipping sideways or face-forward — the trunk hits the ground in
    those weird poses, sit_grounded fires, and the policy converges to a
    "fallen" mode that competes with the actual sit pose.

    When ``command_name`` is provided, the reward is gated to the sit window of
    a phase command. Otherwise it's always-on, optionally gated to the late
    part of the episode via ``min_progress_frac``.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    # Upright check: trunk body's +Z (world frame, third column of rotation matrix
    # derived from the trunk quaternion) dot world-up = trunk's body-up · world-up.
    # Equivalently: 1 - 2*(qx² + qy²) for a unit quaternion (w, x, y, z).
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) = (w, x, y, z)
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: Optional[str] = None,
    ang_vel_std: float = 0.5,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
) -> torch.Tensor:
    """Bonus for low body angular velocity.

    Phase-gated when ``command_name`` is set (sit window of a phase command).
    Always-on otherwise, optionally restricted to the late part of the episode
    via ``min_progress_frac``. Encourages a stable rest pose.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty for joint positions deviating from their default (HOME).

    Returns sum of |joint_pos - default| over the selected joints. Unlike the
    Gaussian `pose` reward (which saturates near 1.0 for any small deviation),
    this gives a *linear* gradient at all deviation magnitudes — useful as a
    focused penalty on a subset of joints (e.g. hip_yaw / hip_roll) to prevent
    them drifting to wide-base stances even when other joints are near HOME.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.15,
) -> torch.Tensor:
    """L1 penalty for joint positions entering a ``margin`` (rad) band next to
    their *hard* range limits.

    The base ``joint_pos_limits`` reward only fires past the *soft* limit
    (global ``soft_joint_pos_limit_factor`` = 0.9 → roughly the last 7.5% of
    range) and only by the radians-overshoot magnitude, so it's near-useless
    against a joint parked on its stop. This term instead reads the *hard*
    limits directly and lets each reward set its own wide margin, scoped to
    specific joints.

    Motivating case: with a low-kp position servo and wide ctrlrange the policy
    can command far past a joint's limit "for free" (no command-side cost) and
    park the joint on its hard stop — e.g. hip_yaw slammed to ±limit so the foot
    slides/pivots. The overshoot is *intended* (it's how a low-kp servo reaches
    its target), so the deterrent must live on the qpos side and bite well
    before the stop.

    For each selected joint with hard limits ``[lo, hi]``::

        soft_lo = lo + margin,  soft_hi = hi - margin
        penalty = relu(soft_lo - q) + relu(q - soft_hi)

    summed over joints: zero in the interior, ramping linearly toward each stop.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]  # (num_envs, num_sel_joints, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_z: float,
    sit_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk_z tracking a sin-interpolated target between stand and sit heights.

    Used for the sitstand task instead of joint-angle matching for the sit pose —
    rewards the END STATE (low trunk) without prescribing HOW the robot gets there.
    The policy is free to find any motion strategy (deep squat, head-supported
    descent, etc.).

    Command (from GroundPickPhaseCommand): cmd[:, 1] = sin(2π·phase).
    sin = +1 at phase 0.25 (sit peak) → target = sit_z.
    sin = -1 at phase 0.75 (stand peak) → target = stand_z.
    sin = 0 at transitions → target = midpoint.
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
) -> torch.Tensor:
    """Always-on Gaussian on joint positions vs a target pose.

    Non-phase analog of ``phase_pose_match``: useful for episodic tasks (e.g.
    the sit env) where there's no cyclic command to weight the reward by, and
    the target pose is constant for the whole episode.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate.
        target_overrides: ``{joint_index: angle_rad}``. Joints not listed default
            to ``asset.data.default_joint_pos`` (the home/standing pose).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def interpolated_pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on joint positions vs a time-interpolated target pose.

    Tracks a target that linearly interpolates from a source pose to a target
    pose over the episode, between progress fractions ``ramp_start_frac`` and
    ``ramp_end_frac``. Before/after the ramp the target is clamped to source /
    final target respectively.

    The point is to enforce smooth descent: snapping to the final target early
    leaves the robot *off-target* relative to where the interpolated target
    currently is, costing pose reward for the duration of the mismatch.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate.
        source_overrides: ``{joint_index: angle_rad}`` defining the source pose
            (start of the ramp). ``None`` = default/HOME pose.
        target_overrides: same, for the target pose (end of the ramp).
        ramp_start_frac, ramp_end_frac: episode-progress window in [0, 1] over
            which the target moves from source to target.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-((joint_pos - interp) / std) ** 2).mean(dim=-1)


def interpolated_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
    source_overrides: Optional[dict] = None,
    target_overrides: Optional[dict] = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target pose (negative — used as penalty).

    Same interpolation schedule as ``interpolated_pose_target_match`` but
    returns ``-mean(|joint_pos - interp|)`` instead of a Gaussian. The L1
    gradient is constant everywhere — useful as a bootstrap signal when the
    Gaussian variant saturates to zero far from target and leaves the policy
    no gradient to discover the target direction.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """L1 distance from a time-interpolated target height (negative — penalty).

    Same role as ``interpolated_pose_l1_penalty`` but on trunk z. Provides a
    constant gradient toward the target height regardless of how far off the
    current z is, complementing the Gaussian ``interpolated_height_target``.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def interpolated_height_target(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """Gaussian on trunk z vs a time-interpolated target height.

    Companion to ``interpolated_pose_target_match`` — same time-interpolation
    logic applied to the trunk height.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def bilateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    left_indices: list,
    right_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 penalty on left/right leg asymmetry.

    For a bilaterally-symmetric robot the leg HOME and any symmetric target
    (FOLD, SIT) satisfy ``q_left + q_right == 0`` on each matched joint pair
    (because the left/right joints use mirrored sign conventions). This term
    penalises departures from that constraint.

    Useful when ``mean()`` of pose-target rewards lets the policy get away
    with one-leg-correct solutions (you collect ~half the reward for free
    and the gradient toward fixing the second leg is too weak to escape that
    local minimum). The penalty here has constant L1 gradient regardless of
    magnitude, so any asymmetry pays a cost and the unique zero is the
    fully-symmetric configuration.

    Returns ``-sum_i |q[left_i] + q[right_i]|`` averaged over the N pairs.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """Compute the time-interpolated joint target across N waypoints.

    waypoints: ordered list of dicts {"frac": float in [0,1],
                                       "overrides": dict[int,float] | None}.
    First waypoint should have frac=0.0 (typically HOME, overrides=None).
    Subsequent waypoints define milestones. Between two waypoints the target
    linearly interpolates. Before the first / after the last it clamps.

    Returns a (num_envs, num_joints) tensor of target joint angles.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # Find which segment we're in (broadcast over envs).
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # Take this segment's value when progress is in [f0, f1] or past it.
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """Same logic as _multistage_target_pose but for trunk z height.

    waypoints: [{"frac": float, "height": float}, ...].
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Multi-waypoint variant of interpolated_pose_target_match.

    waypoints: [{"frac": 0.0, "overrides": None},
                {"frac": 0.4, "overrides": FOLD_OVERRIDES},
                {"frac": 0.7, "overrides": SIT_OVERRIDES}]

    Use this to enforce a curriculum-style trajectory through one or more
    intermediate poses (e.g. stand → fold → sit). Same per-joint Gaussian
    semantics as the single-stage version.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def multistage_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to multistage_pose_target_match."""
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.03,
) -> torch.Tensor:
    """Multi-waypoint Gaussian on trunk z."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_z) / std) ** 2)


def multistage_height_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to multistage_height_target."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_z)


def pose_target_match(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Gaussian pose-match against a single fixed target.

    target = ``default_joint_pos`` with the per-index overrides applied. No
    waypoints, no episode-progress interpolation — the same target is rewarded
    from t=0 to the end of the episode.
    """
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def pose_l1_penalty(
    env: ManagerBasedRlEnv,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """L1 companion to ``pose_target_match`` (constant gradient toward target)."""
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> torch.Tensor:
    """Gaussian on trunk z against a single fixed target."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return torch.exp(-((z - target_height) / std) ** 2)


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``height_target_gaussian``."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty proportional to ``|a_z|`` of the trunk (finite-diff of v_z).

    Captures hard impacts (large deceleration spike on landing) AND incentivises
    a smooth quasi-static descent (constant velocity → a_z ≈ 0). At rest a_z is
    zero so the seated robot pays no cost.

    State is kept on the env in ``_prev_trunk_vz``; at episode reset the
    accel is zeroed to avoid a transient from the previous episode's final
    state leaking into the new one.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # Zero out a_z at reset steps to suppress the cross-episode transient.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on downward trunk velocity beyond ``max_down_vel``.

    Caps descent SPEED, which ``trunk_vertical_accel_penalty`` alone cannot:
    a fast constant-velocity drop has a_z ≈ 0 the whole way down and pays only
    one impact spike at the bottom — cheap relative to arriving at the target
    pose sooner. This term makes every step of a too-fast descent cost reward,
    so the gentlest descent that stays under the cap is optimal. Zero at rest
    and for any motion slower than the cap (including all upward motion).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(
    env: ManagerBasedRlEnv,
    height_full: float = 0.06,
    height_zero: float = 0.08,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while seated UPRIGHT: |v| Gaussian, z- and tilt-gated.

    exp(-(|v|/vel_std)²) · smoothstep(z) · smoothstep(tilt). The z gate is full
    below ``height_full`` and zero above ``height_zero`` (inactive during the
    descent). The tilt gate is full below ``tilt_full_deg`` and zero above
    ``tilt_zero_deg`` — WITHOUT it, "lie still on your back" scores as well as
    "sit still upright" (the trunk on its back is inside the seated z band and
    perfectly motionless), which is exactly the exploit run 2 converged to.
    Makes "rest quietly, upright, at the seated height" the only rewarded rest.
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear upright reward weighted by a smoothstep on trunk z.

    Returns ``body_upright_linear * smoothstep((z - low)/(high - low))`` so the
    upright incentive is full while the robot is still standing tall, and
    fades to zero once it has committed to the lower sit configuration (where
    butt-on-ground orientation is fine). Prevents the policy from learning to
    tip backward while still high (which would otherwise farm the descent
    reward via a controlled fall).
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose STAND, 1 = pose DOWN.

    [0, descent_end)       : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1       (bas)
    [hold_end, rise_end)   : 1 -> 0  (se lever)
    [rise_end, 1.0)        : 0       (haut / repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """Cible articulaire interpolée d'un geste de shoot à 4 keyframes.

    phase (B,) ∈ [0,1). stand/back/forward (k,) ou (1,k). Retour (B,k).

    [0, windup_end)        STAND   -> BACK     (armement)
    [windup_end, kick_end) BACK    -> FORWARD  (frappe sèche)
    [kick_end, return_end) FORWARD -> STAND    (retour)
    [return_end, 1.0)      STAND             (repos)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # à s3=1 (phase>=return_end) => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out


def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
    joint_names: Optional[list] = None,
):
    """(cur, target) pour le geste de shoot, joints résolus PAR NOM.

    Les 3 poses partagent les mêmes clés (14 joints). L'ordre des noms est
    donné par `stand_pose` (ou par `joint_names` si fourni — un sous-ensemble
    des clés, ex. jambe droite + cou d'un côté, jambe gauche de l'autre, pour
    appliquer des std différents au geste vs à la jambe d'appui).
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device,
                            dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v,
                              windup_end, kick_end, return_end)          # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée du shoot.

    Reward directif et symétrique : chaque phase impose la config articulaire
    exacte. Résolution PAR NOM. `joint_names` restreint l'évaluation à un
    sous-ensemble (ex. jambe droite + cou tracés serré, jambe gauche d'appui
    tracée lâche pour la laisser équilibrer).
    """
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: Optional[dict] = None,
    back_pose: Optional[dict] = None,
    forward_pose: Optional[dict] = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: Optional[list] = None,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (gradient constant, pénalité<=0)."""
    cur, target = _kick_pose_error(
        env, asset_cfg, command_name, stand_pose or {}, back_pose or {},
        forward_pose or {}, windup_end, kick_end, return_end, joint_names,
    )
    return -(cur - target).abs().mean(dim=-1)


def kick_engagement(
    phase: torch.Tensor,
    windup_end: float,
    return_end: float,
) -> torch.Tensor:
    """Gate d'engagement du geste ∈ [0,1] (pur) — pour pondérer les rewards
    d'équilibre unipède qui ne doivent s'appliquer que hors du repos STAND.

    [0, windup_end)        : 0 -> 1  (montée pendant l'armement)
    [windup_end, return_end): 1       (phase de frappe = appui unipède attendu)
    [return_end, 1.0)      : 0        (repos STAND, appui bipède, CoM centré OK)
    """
    g = torch.zeros_like(phase)
    ramp = phase < windup_end
    g = torch.where(ramp, phase / windup_end, g)
    hold = (phase >= windup_end) & (phase < return_end)
    g = torch.where(hold, torch.ones_like(phase), g)
    return g


def com_over_support_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.04,
    windup_end: float = 0.35,
    return_end: float = 0.75,
) -> torch.Tensor:
    """Reward gaussien : projection horizontale du CoM proche du pied d'appui,
    gaté sur la phase de frappe (kick_engagement).

    Apprend le transfert latéral du poids sur le pied d'appui (support). Sans
    ça, un geste à un pied issu de poses relevées en appui bipède garde le CoM
    centré entre les deux pieds → bascule et chute dès que l'autre pied se lève.
    Au repos STAND le gate est 0 (appui bipède, CoM centré autorisé).

    `asset_cfg` doit cibler le site du pied d'appui (ex. site_names=["left_foot"]).
    `std` en mètres (rayon de tolérance CoM↔pied, ~taille du pied).
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std ** 2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """(cur, target) pour la pose interpolée par la phase, résolue PAR NOM.

    Cible = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` si fourni, sinon le DEFAULT/HOME du modèle). blend ∈ [0,1]
    (0 = STAND, 1 = target_pose) via `phase_pose_blend`.
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée STAND<->DOWN.

    Reward directif : indique la config articulaire exacte à chaque phase. Se
    relever (cible → STAND) est récompensé exactement comme se baisser (cible →
    DOWN) — symétrique par construction. Résolution PAR NOM.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (pénalité négative).

    Gradient constant partout — donne une direction vers la cible même quand la
    gaussienne ci-dessus a saturé à ~0 loin de la cible.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    target_overrides: Optional[dict] = None,
    phase: str = "approach",
) -> torch.Tensor:
    """Reward matching a target pose, weighted by phase-cycle command.

    Generic helper for phase-conditioned tasks (e.g. sit/stand). The command
    encodes phase as [cos(2π·phase), sin(2π·phase), 0]:
      - "approach" weight = max(0, sin(2π·phase)) — peaks at phase 0.25.
      - "return"   weight = max(0,-sin(2π·phase)) — peaks at phase 0.75.

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Optional subset of joints to evaluate (rest ignored).
        target_overrides: {joint_index: angle_rad}. Joints not listed default
            to asset.data.default_joint_pos (the home/standing pose).
        phase: "approach" or "return".
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    if phase == "approach":
        weight = torch.clamp(cmd[:, 1], min=0.0)
    else:
        weight = torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
) -> torch.Tensor:
    """Reward for returning to the standing pose after ground pick, weighted by the return phase.

    The return phase is the second half-cycle (sin < 0, phase ∈ [0.5, 1.0]),
    smoothly weighted by max(0, -sin(2π*phase)).

    Args:
        std: Gaussian std per joint (rad).
        joint_indices: Subset of joints to evaluate. Use to apply different stds
            to leg joints vs neck/head joints (call this reward twice).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos  = _servo_joint_pos(env, asset)        # (num_envs, n_servo_joints)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos   = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)

    # Return weight: max(0, -sin(2π*phase)) — peaks at 1 at phase=0.75, zero at 0.5 and 1
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward trunk verticality, weighted by the RETURN phase (stand-up aid).

    Same return weighting as ``ground_pick_return_pose`` (``max(0, -sin(2π·phase))``)
    so it only rewards being upright during the stand-up, never fighting the
    forward lean of the approach. Verticality = ``exp(-tilt²/std²)`` with the same
    tilt proxy as ``body_upright_gaussian`` (``2*(qx²+qy²) ≈ 1-cos(tilt)``). A broad
    std (0.4 rad ≈ 23°) gives gradient even from a fairly tilted crouch.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # qx² + qy²
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# --------------------------------------------------------------------------- #
# Ground-pick : gating de phase SEGMENTÉ (durées descente/palier/remontée/repos #
# indépendantes, au lieu de la pondération sinusoïdale max(0,±sin)).            #
#   down-gate  = phase_pose_blend(phase, descent_end, hold_end, rise_end)       #
#               0 (haut) -> 1 (descente) -> 1 (palier bas) -> 0 (remontée/repos) #
#   up-gate    = phase_rise_gate(phase, hold_end, rise_end)                      #
#               0 avant la remontée -> 0..1 (remontée) -> 1 (repos debout)       #
# --------------------------------------------------------------------------- #
def phase_rise_gate(
    phase: torch.Tensor, hold_end: float, rise_end: float
) -> torch.Tensor:
    """Gate montante pour le RETOUR : 0 avant hold_end, 0->1 sur [hold_end,
    rise_end), 1 après (repos debout)."""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    std: float = 0.10,
    target_height: float = 0.0,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_ground_proximity gaté par la down-gate segmentée (descente+palier)."""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-((mouth_z - target_height) / std) ** 2)
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_perpendicular_to_ground gaté par la down-gate segmentée."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = bouche pointe droit vers le bas
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_pose gaté par la up-gate segmentée (remontée+repos)."""
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-((joint_pos - default_pos) / std) ** 2).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_upright gaté par la up-gate segmentée."""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    joint_indices: Optional[list] = None,
    hold_end: float = 0.35,
) -> torch.Tensor:
    """Pénalise la vitesse des joints du cou pendant la DESCENTE+palier (freine le
    piqué de la tête).

    Coût = mean(joint_vel²) sur les joints donnés, gaté à 1 pour phase < hold_end
    (descente + palier bas) et 0 ensuite (remontée + repos) -> ne gêne PAS le
    relever du cou. Retourne un coût positif ; à utiliser avec un poids négatif.
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel ** 2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # descente + palier bas uniquement
    return gate * cost


def sample_mouth_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_kg: float = 0.01,
    max_kg: float = 0.04,
) -> None:
    """Event de reset : tire une masse d'objet 'tenu dans la bouche' par env (kg),
    stockée sur env._mouth_payload_kg. Utilisée par apply_mouth_payload_force."""
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
    ),
    command_name: str = "twist",
    hold_end: float = 0.35,
    ramp: float = 0.05,
    gravity: float = 9.81,
) -> torch.Tensor:
    """Hook par-step (utilisé comme reward de poids 0) : applique le POIDS de
    l'objet tenu dans la bouche comme force externe verticale au mouth_tip, gaté
    sur la remontée (phase >= hold_end, rampe rapide au moment du 'grab').

    Émule une masse ponctuelle au bout de la bouche pendant le relever : la force
    m·g est appliquée au CoM du corps + le couple (p_mouth - p_com) × F, ce qui
    équivaut à l'appliquer au mouth_tip (bon bras de levier pour le cou). Retourne
    0 (ce n'est pas une vraie récompense — juste le hook d'application)."""
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # 0 avant grab -> 1 après
    fz = -(gate * payload) * gravity                     # (N,) force verticale (bas)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]           # (N,3)
    p_com = asset.data.body_com_pos_w[:, bid, :]         # (N,3)
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)        # applique F au mouth_tip
    asset.write_external_wrench_to_sim(
        forces=F.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid],
    )
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================================
# Domain Randomization Events
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """Randomize firmware PD gains per episode (NON-accumulating).

    Under the canonical BAM actuator (``bam.mjlab.BamActuator``) gains are scaled
    per-env via ``set_gains``/``reset_gains`` (the actuator owns ``kp_scale``/
    ``kd_scale``), so we never touch the MuJoCo model — no accumulation risk. The
    sampled per-joint factors are averaged into a single scalar per env (the
    actuator applies one scale across its joints), matching the previous behavior.
    Non-BAM actuators are skipped (e.g. the roller XmlActuator, which doesn't
    expose set_gains).

    Args:
        env: The environment
        env_ids: Environment IDs to randomize (None = all envs)
        kp_range: (min, max) for kp randomization
        kd_range: (min, max) for kd randomization
        asset_cfg: Asset configuration
        operation: unused (kept for cfg compatibility; scaling is always applied)
    """
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # Restore nominal first (prevents accumulation), then apply fresh scale.
        actuator.reset_gains(env_ids)
        actuator.set_gains(
            env_ids,
            kp_scale=kp_samples.mean(dim=1, keepdim=True),
            kd_scale=kd_samples.mean(dim=1, keepdim=True),
        )


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """No-op startup event whose only purpose is the decorator above.

    bam's BamActuator (mjlab_frictionloss branch) writes a per-env friction
    budget into MuJoCo's dof_frictionloss/dof_damping every step, which
    requires those model fields to be expanded per world. mjlab expands
    exactly the fields declared by event functions via requires_model_fields,
    so every env using the BAM actuator must register this event.
    """


def randomize_bam_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Per-episode joint-friction randomization for the BAM actuator (NON-accumulating).

    Under BAM, MuJoCo's dof_frictionloss is zeroed (BAM computes friction in
    compute()), so stock dr.dof_frictionloss is a no-op. Instead this samples a
    per-env scalar in ``scale_range`` and applies it to the FrictionDRBamActuator's
    ``friction_scale``, which multiplies BAM's velocity-independent friction budget
    (Coulomb + Stribeck + load). Restores nominal (1.0) first to avoid accumulation.
    No-op on actuators without a friction_scale hook.
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize body mass and inertia together with the same scaling factor.

    This maintains physical consistency - mass and inertia must scale together
    to avoid creating invalid inertia tensors that cause simulation instability.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        scale_range: (min, max) scaling factor applied to both mass and inertia
        asset_cfg: Asset configuration specifying which bodies to randomize
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # Get body indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # Sample ONE random scale per environment (applied to both mass and inertia)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # Store original values on first call
    if not hasattr(env, '_original_mass_inertia'):
        env._original_mass_inertia = {
            'mass': env.sim.model.body_mass[0, body_indices].clone(),
            'inertia': env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # Reset to original first (to prevent accumulation)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original['mass'].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = original['inertia'].unsqueeze(0).expand(num_envs, -1, -1)

    # Apply same scale to both mass and inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # Scale all 3 inertia components


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """Update the relative number of standing environments based on training progress.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term
        standing_stages: List of dicts with 'step' and 'rel_standing_envs' keys
            Example: [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        Current rel_standing_envs value as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update rel_standing_envs based on current step
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """Update velocity tracking std parameter based on training progress.

    Starts with loose std (easy rewards) to learn basic walking, then gradually
    tightens to improve velocity tracking accuracy.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        reward_name: Name of the reward term (e.g., "track_linear_velocity")
        std_stages: List of dicts with 'step' and 'std' keys
            Example: [
                {"step": 0, "std": 0.5},      # Start loose - learn to walk
                {"step": 250, "std": 0.3},     # Moderate - refine gait
                {"step": 500, "std": 0.2},     # Strict - accurate tracking
            ]

    Returns:
        Current std value as a tensor
    """
    del env_ids  # Unused

    # Get reward term configuration
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # Update std based on current step
    current_std = std_stages[0]["std"]  # Default to first stage

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # Update the reward term's std parameter
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """Update push velocity range based on training progress.

    Starts with no/small pushes to learn clean walking, then gradually increases
    to build robustness without disrupting early learning.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        event_name: Name of the push event term (e.g., "push_robot")
        push_stages: List of dicts with 'step' and 'velocity_range' keys
            Example: [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        Current max push magnitude as a tensor
    """
    del env_ids  # Unused

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    # Update velocity_range based on current step
    current_range = push_stages[0]["velocity_range"]  # Default to first stage

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # Update the event configuration's velocity_range parameter
    event_cfg.params["velocity_range"] = current_range

    # Return max magnitude for logging
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    ranges_stages: list[dict],
) -> torch.Tensor:
    """Update wheel friction based on training step stages."""
    del env_ids  # Unused

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[dict],
) -> torch.Tensor:
    """Step-staged reward weight curriculum.

    mjlab 1.3.0 dropped the built-in ``mdp.reward_weight`` helper, so microduck
    provides its own. ``weight_stages`` is a list of ``{"step": int, "weight":
    float}`` dicts; the weight of the latest stage whose step has elapsed is
    applied. Mutates the live RewardManager term cfg (not env.cfg, which is a
    deepcopy at manager init).
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Update CoM randomization range based on training progress.

    Gradually increases the CoM offset range so the robot first learns to walk
    with a small CoM uncertainty, then progressively larger.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused)
        event_name: Name of the CoM randomization event (e.g., "randomize_com")
        range_stages: List of dicts with 'step' and 'range' keys (range in meters)
            Example: [
                {"step": 0,          "range": 0.003},
                {"step": 1000 * 24,  "range": 0.005},
                {"step": 2000 * 24,  "range": 0.008},
            ]

    Returns:
        Current range value as a tensor (for logging)
    """
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """Masques de promotion/rétrogradation du curriculum de pente.

    move_up   : a parcouru plus de 40% de la tuile → il a dévalé la rampe,
                on la rend plus raide. Aligné sur la termination
                terrain_edge_reached (~3.8 m, threshold_fraction=0.95 par
                défaut sur size_x=8.0), qui termine l'épisode avant le seuil
                de moitié (4.0 m) — sans cet alignement un traverseur réussi
                n'est jamais promu.
    move_down : a à peine avancé (< 20% de la tuile) → chute/blocage précoce,
                on adoucit la rampe.
    """
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Curriculum de raideur pour roller_slope (pas de vitesse commandée).

    Progression basée sur la distance en x parcourue depuis l'origine de spawn.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = (
        asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    update_lin_vel_y: bool = True,
    update_ang_vel_z: bool = True,
    forward_only: bool = False,
) -> torch.Tensor:
    """Update velocity command ranges based on training progress.

    Gradually increases the commanded velocity ranges to allow the robot to learn
    higher speeds progressively. Starts with smaller ranges for stable learning,
    then expands to more challenging velocities.

    Args:
        env: The RL environment
        env_ids: Environment IDs (unused, but required by curriculum interface)
        command_name: Name of the velocity command term (e.g., "twist")
        velocity_stages: List of dicts with 'step', 'lin_vel_range', and 'ang_vel_range' keys
            Example: [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]

    Returns:
        Current max linear velocity as a tensor
    """
    del env_ids  # Unused

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from typing import cast

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # Update velocity ranges based on current step
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # Update command ranges
    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def running_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    speed_stages: list[dict],
) -> torch.Tensor:
    """Advance a forward-only running speed band over training.

    A band avoids spending most samples near zero while an explicit standing
    bucket in the command cfg still trains the deployment idle state.
    """
    del env_ids

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    current_min = float(speed_stages[0]["min_speed"])
    current_max = float(speed_stages[0]["max_speed"])
    for stage in speed_stages:
        if env.common_step_counter >= stage["step"]:
            current_min = float(stage["min_speed"])
            current_max = float(stage["max_speed"])
    if not (0.0 <= current_min <= current_max):
        raise ValueError(f"invalid running speed band: {(current_min, current_max)}")

    cfg.ranges.lin_vel_x = (current_min, current_max)
    return torch.tensor([current_max], device=env.device)


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Projected gravity vector in body frame.

    Returns the gravity vector projected into the robot's body frame,
    representing pure orientation without linear acceleration.
    This is simpler than raw accelerometer and only depends on orientation.

    Returns:
        torch.Tensor: Projected gravity in body frame (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """Per-env constant IMU mounting-misalignment rotation (sampled once).

    Models a fixed small mounting/calibration error of the IMU on each robot.
    Sampled lazily on first use and cached — constant per env for the whole run
    (like a startup randomization), so it's a *systematic per-robot bias*, not
    per-step noise. Replaces the old randomize_imu_orientation event, which wrote
    site_quat (not per-env expanded under mjlab 1.3.0, and not read by the
    projected_gravity / base_ang_vel observations anyway).

    Returns a (num_envs, 4) unit quaternion (w, x, y, z).
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """projected_gravity with a per-env constant IMU mounting misalignment."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """base angular velocity with the SAME per-env IMU misalignment as gravity."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Raw accelerometer reading (includes gravity + linear acceleration).

    Returns normalized raw accelerometer which mimics what a real IMU measures.
    This is different from pure projected_gravity which only reflects orientation.
    Reads from the MuJoCo accelerometer sensor "imu_accel".

    Returns:
        torch.Tensor: Normalized raw accelerometer reading (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Access the model to find the sensor address
    # The accelerometer sensor is the 5th sensor (index 4) in robot.xml
    # Sensors: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # Get sensor address from model arrays (sensor_adr is torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # This is a TorchArray/tensor
    sensor_id = 4  # imu_accel is the 5th sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # Convert to Python int

    # Read accelerometer data (specific force measured by sensor)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr:sensor_adr+3]

    # MuJoCo accelerometer measures specific force (like real sensor)
    # Negate to match convention: when at rest upright, should point down
    accel_negated = -accel_raw

    # Normalize to unit vector
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b  # Fallback to projected gravity
    )

    return accel_normalized

def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize IMU sensor mounting orientation by small angles.
    
    Simulates slight mounting errors or calibration offsets in the real robot.
    The IMU orientation is randomized by rotating around random axes by up to max_angle_deg.
    
    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_angle_deg: Maximum rotation angle in degrees (default 2.0°)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    
    asset: Entity = env.scene[asset_cfg.name]

    # IMU site is the first site (index 0) in robot.xml
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0
    
    # Store original orientation on first call
    if not hasattr(env, '_original_imu_quat'):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()
    
    # Generate random rotations for each environment
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0
    
    # Random rotation angles [-max_angle, +max_angle] for each axis
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad
    
    # Convert Euler angles to quaternions (small angle approximation for efficiency)
    # For small angles: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w component
    quats_delta[:, 1:] = half_angles  # x, y, z components
    
    # Normalize the quaternion
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)
    
    # Get original quaternion and apply delta rotation
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)
    
    # Quaternion multiplication: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = quats_delta[:, 0], quats_delta[:, 1], quats_delta[:, 2], quats_delta[:, 3]
    w2, x2, y2, z2 = original_quat[:, 0], original_quat[:, 1], original_quat[:, 2], original_quat[:, 3]
    
    new_quat = torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
    ], dim=1)
    
    # Apply to the selected environments
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Simple time-based phase for standing task.

    Returns a scalar phase value that cycles from 0 to 1 based on time.
    This allows the policy to have a sense of time progression even when standing.

    Args:
        env: The RL environment
        asset_cfg: Not used, but kept for API consistency

    Returns:
        Phase value [0, 1] as tensor of shape (num_envs, 1)
    """
    # Simple time-based phase that cycles every 2 seconds
    # This gives the policy a time-varying signal
    phase_period = 2.0  # seconds
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,    # below this: no reward (standing)
    running_threshold: float = 0.5,     # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """Air-time reward with separate swing-time windows for walking vs running.

    - command < command_threshold  → 0 (standing, no reward)
    - command_threshold–running_threshold → walk window [walk_min, walk_max]
    - command > running_threshold  → run  window [run_min,  run_max]

    This lets the walking gait keep its deliberate 100–250 ms swing while
    running can use a faster 50–250 ms cadence.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # Per-env thresholds broadcast over feet
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # sum over feet

    # Zero reward when standing
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """Reward staying still when command is near zero.

    Returns exp(-body_vel² / vel_std²) when command < threshold, else 0.
    This is monotonically decreasing with body speed — moving faster is always
    less rewarding. There is no threshold the robot can cross to 'escape' it,
    unlike gate-based stepping penalties.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-body_vel ** 2 / vel_std ** 2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalise leg joint velocities only when command is near zero.

    Targets the standing-shake problem: the policy makes rapid oscillating
    corrections around the home pose when standing. Gated on command so it
    does not affect the walking gait at all.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel ** 2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise stepping when at zero command and the body is not being pushed.

    Symmetric counterpart to the air_time reward:
    - air_time gives  +reward for stepping when command > threshold  (walk)
    - this gives      -reward for stepping when command < threshold  (stand)

    The body-velocity gate prevents penalising recovery steps after a push:
    if the robot is already moving fast (pushed), no penalty is applied so it
    can still take steps to catch itself.

    Returns a value in [0, 1] (use a negative weight in the config).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # Was either foot recently lifted? (last completed air phase > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # Are we in standing mode? (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # Is the body still? (not being pushed)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot air time only when at zero command AND robot has high velocity (recovering from push).

    This encourages the robot to take steps to recover balance when pushed,
    but does NOT fire during normal walking (command > threshold).

    Args:
        env: The RL environment
        asset_cfg: Asset configuration (unused but kept for API consistency)
        command_name: Name of the velocity command in the command manager
        command_threshold: Speed below which the robot is considered to be in standing mode
        velocity_threshold: Linear velocity threshold to activate stepping reward (m/s)
        air_time_threshold: Minimum air time to count as a step (seconds)

    Returns:
        Reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Only fire for standing envs (command near zero)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Only reward stepping when velocity is high (being pushed)
    should_step = vel_magnitude > velocity_threshold

    # Get foot air time from contact sensor
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - left and right foot

    # Reward if either foot has been in air recently
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # Only give reward when: standing command AND high body velocity AND foot stepped
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Reduce pose tracking weight when robot has high velocity (recovering from push).

    This gives the robot freedom to deviate from the standing pose when taking
    recovery steps, while maintaining strict pose tracking when standing still.

    Args:
        env: The RL environment
        base_pose_reward: The original pose reward (before weighting)
        asset_cfg: Asset configuration (unused but kept for API consistency)
        velocity_threshold: Linear velocity threshold to start reducing weight (m/s)
        min_weight: Minimum weight multiplier (0-1) at high velocities

    Returns:
        Weighted reward tensor of shape (num_envs,)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Get base linear velocity magnitude
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # Only XY plane

    # Compute weight: 1.0 when stationary, min_weight at high velocity
    # Use smooth transition via sigmoid-like function
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Randomize base orientation at episode start to force reactive behavior.

    Adds random pitch and roll to the robot's base orientation at the start of
    each episode. This prevents the policy from memorizing a single initial state
    and forces it to use feedback to adapt to different orientations.

    Args:
        env: The environment
        env_ids: Environment IDs to randomize
        max_pitch_deg: Maximum pitch angle in degrees (forward/backward tilt)
        max_roll_deg: Maximum roll angle in degrees (side-to-side tilt)
        asset_cfg: Asset configuration
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # Generate random pitch and roll angles
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # Keep yaw at 0

    # Convert Euler angles (roll, pitch, yaw) to quaternion
    # Using the standard aerospace sequence (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # Normalize quaternion
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # Get root position index (freejoint starts at qpos index 0)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # Quaternion starts at index 3

    # Apply the randomized orientation to selected environments
    env.sim.data.qpos[env_ids, root_quat_idx:root_quat_idx+4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Set the robot to a prone (belly-down) orientation for stand-up training.

    Rotates the robot 90° forward around the pitch axis (Y) so the front/belly
    faces the ground and legs point upward. Combined with a random yaw.

    Quaternion derivation:
        quat_pitch90 = [s, 0, s, 0]   where s = sqrt(2)/2  (90° around Y)
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,   # w
            -s * sy,  # x
            s * cy,   # y
            s * sy,   # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
):
    """Randomly initialize each env as face-down (belly) or face-up (back), with random yaw.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]

    Args:
        face_down_prob: probability of sampling face-down (vs face-up). A curriculum
            can ramp this from a high initial value (easier task) toward 0.5.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: Optional[dict] = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """Reset to a random ground state: face-down, face-up, sitting, or standing.

    Broader than ``set_random_prone_orientation`` — used by the stand-up env so
    the policy learns to recover from any plausible pose, including the sitting
    keyframe (rest state of the sit policy) and an already-standing pose (so it
    also learns to *hold* a stand, not only to rise).

    Modes (probabilities are normalized; they need not sum to 1.0):
      - face-down (belly to floor): +90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - face-up   (back to floor):  -90° pitch, random yaw, z in [prone_z_min, prone_z_max].
      - sitting:                    upright (±sitting_tilt_max), random yaw, z low,
                                    joints set to ``sitting_joint_overrides``.
      - standing:                   upright (±sitting_tilt_max), random yaw, z in
                                    [standing_z_min, standing_z_max], joints left at
                                    HOME (whatever ``reset_robot_joints`` set).

    Args:
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}`` to write into
            ``qpos[7+idx]`` for envs sampled into the sitting bucket. ``None``
            keeps joints at whatever ``reset_robot_joints`` already set.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd  = face_down_prob / total
    p_fu  = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0 ** -0.5  # sqrt(2)/2

    face_down = torch.stack([ s * cy, -s * sy,  s * cy,  s * sy], dim=1)
    face_up   = torch.stack([ s * cy,  s * sy, -s * cy,  s * sy], dim=1)
    # Upright sitting: yaw-only by default, with optional ±sitting_tilt_max
    # pitch/roll noise so the policy doesn't overfit to perfectly-upright starts.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll  = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll  * 0.5); sr = torch.sin(roll  * 0.5)
        # ZYX intrinsic Euler → quaternion (yaw * pitch * roll).
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fd    = u < p_fd
    is_fu    = (u >= p_fd) & (u < p_fu)
    is_sit   = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # Face-up partial-roll noise: rotate the supine pose about the body's long
    # axis by uniform ±face_up_roll_max. WHY (2026-07, back-recovery was
    # seed-lucky): the reward landscape between supine and prone is FLAT —
    # upright_linear (cos tilt) is ≈0 through the whole roll, height doesn't
    # change — so rolling off the back only pays via the front-rise path that
    # follows, a long-horizon dependency that noisy exploration rarely finds
    # from a perfectly flat supine start. With roll noise, a fraction of
    # face-up spawns start near-on-side (partway along the roll): the policy
    # learns roll-completion from easy starts and generalizes back to flat
    # supine — a built-in reverse curriculum. Uniform sampling keeps every
    # difficulty represented (flat back |roll|<15° ≈ 17% at ±90°), so no
    # annealing schedule is needed, and varied post-fall poses are realistic
    # DR for deployment anyway.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll = rotation about the body's LONG axis, which is body z (the
        # spine: trunk z is up when standing → horizontal when lying). NOT body
        # x — supine leaves body x pointing skyward, so an x-roll would only
        # spin the robot in place like the yaw noise already does.
        # Body-frame rotation → right-multiply: q_fu ⊗ [ct, 0, 0, st].
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack(
            [
                w * ct - z * st,
                x * ct + y * st,
                y * ct - x * st,
                w * st + z * ct,
            ],
            dim=1,
        )

    # Sitting and standing share the same upright orientation (identity + optional
    # ±sitting_tilt_max); they differ only in trunk height and joint pose.
    new_quat = face_down.clone()
    new_quat[is_fu]    = face_up[is_fu]
    new_quat[is_sit]   = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    # Random z per env: prone heights for face-down/up, low for sit, ~standing for stand.
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit   = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2]   = new_z
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6]  = 0.0

    # Sitting-bucket joint overrides (e.g. knee/ankle bent to keyframe).
    # Override keys are SERVO indices (14-joint layout); translate to entity
    # joint indices so models with interleaved passive_* joints (backlash)
    # write the intended joints. qpos column = 7 + entity joint index
    # (robot free joint first, all hinges 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # Joint noise for sitting envs: Gaussian noise on every actuated joint
    # so the policy sees a distribution of plausible "sit" starts rather than
    # a single canonical pose. Captures real-world transfer where the robot's
    # joint angles won't match the SIT keyframe exactly when the standup
    # policy takes over from the sit policy.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # Servo joints only: passive_* joints (backlash hinges) have tiny
            # ranges and must stay at 0 on reset.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# Deep-crouch anchor pose (velstand run-5): the "stuck" mid-recovery basin —
# knees folded under the body, trunk pitched forward, feet flat. Values chosen
# by extending the HOME zig-zag (hip fwd / knee back / ankle fwd, sign
# conventions per the SIT keyframe fold directions) to deep flexion, inside
# the ±1.57 joint limits. hip_yaw/hip_roll/neck stay at HOME.
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """Reset selected envs into a random mid-recovery crouch.

    Reverse curriculum for the recovery last mile (velstand run-5 lesson):
    prone-init episodes spend most of their fallen budget getting TO the deep
    crouch and are recycled shortly after reaching it, so the crouch→stand
    mile gets almost no on-policy data — the policy converged to parking
    there. Seeding resets ACROSS that mile (depth λ ∈ [depth_min, depth_max]
    between standing and the deep-crouch anchor, trunk pitch and z scaled
    with λ) makes the frontier dense from step 0 of the episode.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # Joints: lerp HOME → anchor on the leg pitch chain, uniform noise on the
    # servo joints only (passive_* backlash hinges have ±1° ranges — noise
    # there would spawn them pinned outside their limits).
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # Base orientation: forward pitch scaled with depth (the stuck basin is a
    # forward crouch from both fall directions), random yaw, small roll noise.
    pitch = lam * math.radians(pitch_max_deg) \
        + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state's sitting branch.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # Trunk height scaled with depth, small upward margin to settle cleanly.
    z = z_stand + lam * (z_deep - z_stand) \
        + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
):
    """Reset event that overrides orientation to prone with probability `prone_prob`.

    With prob `prone_prob`, replaces the upright orientation (already set by
    reset_base) with a prone orientation; otherwise leaves it upright. Among the
    overridden envs, `face_down_prob` picks face-down (belly) vs face-up (back).

    Also lifts z to [prone_z_min, prone_z_max] for the overridden envs so the
    head/neck clearance is sufficient — the vel-env reset z (~0.125) would
    clip the head through the ground at 90° pitch.

    At prone_prob=2/3 and face_down_prob=0.5 you get a balanced 33/33/33 split
    of upright/face-down/face-up resets, which is the standard mixture for
    learning fall recovery alongside normal upright start.

    With ``crouch_prob`` > 0, an additional exclusive slice of envs is reset
    into a random mid-recovery crouch via ``set_random_crouch_state`` (reverse
    curriculum for the recovery last mile — see its docstring).
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None means "all envs" (the initial global reset passes None —
    # the old early-return silently skipped prone init there).
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = env_ids.to(env.device, dtype=torch.long) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    # One draw partitions envs into exclusive prone / crouch / untouched slices.
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(
            env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob
        )
        # Override z so the prone body has head/neck clearance when settling.
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


def event_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """Mutate an event term's params at scheduled steps.

    Mirror of termination_param_curriculum but for events. Uses the live
    EventManager term cfg via get_term_cfg, since env.cfg.events is a deepcopy.
    param_stages: list of {step: int, params: dict}. Shallow-merged into the
    live event term's params at the latest matching stage.
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def face_down_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    prob_stages: list[dict],
) -> torch.Tensor:
    """Ramp face_down_prob on a reset event over training.

    Args:
        event_name: name of the event term using set_random_prone_orientation
        prob_stages: list of {step: int, prob: float}. Higher prob = more
            face-down resets (easier task); ramp toward 0.5 as training proceeds.
    """
    del env_ids

    # NOTE: must update the live EventManager term_cfg, not env.cfg.events —
    # EventManager.__init__ does deepcopy(cfg), so mutating env.cfg.events is a no-op.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_prob = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter > stage["step"]:
            current_prob = stage["prob"]

    event_cfg.params["face_down_prob"] = current_prob
    return torch.tensor([current_prob])


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """Like UniformVelocityCommand but only draws the command arrows (no actual velocity arrows)."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Turn-in-place practice: for a fraction of envs, zero the linear velocity
        # and force a meaningful (away-from-zero) yaw command. Independent uniform
        # sampling almost never produces "lin≈0, |ang| large" (~2% of samples), so
        # spinning on the spot was effectively untrained → slow/unstable real-robot
        # turning. Mirrors the base rel_forward_envs mechanism.
        p = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if p <= 0.0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        rr = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(rr.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        mag = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * maxr, maxr)
        self.vel_command_b[turn_ids, 2] = sign * mag
        # These envs must actually turn — un-mark them as standing (which would
        # zero the command) and refresh the world-frame reference copy.
        self.is_standing_env[turn_ids] = False
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        # Command linear velocity arrow (blue).
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world(
            (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
        )
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


@_dataclass(kw_only=True)
class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    # Fraction of envs commanded to turn in place (lin=0, |ang| forced to
    # [0.4·max, max]) each resample. 0 = disabled (base uniform sampling only).
    rel_turn_in_place_envs: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """Velocity command where cmd[2] is the heading error in the robot's body frame.

    cmd[0] = lin_vel_x  (throttle: 0=coast, +push, -brake)
    cmd[1] = lin_vel_y  (unused, 0)
    cmd[2] = heading_error  (+ = target is to the right/CW, - = to the left/CCW)
             0 → go straight, ±max = target is max_angle rad to the right/left

    During training: a random world-frame heading is sampled at each episode reset.
    At every step, cmd[2] = clamp(wrap(current_yaw - target_yaw), ±max_angle).
    Positive when the robot is pointing CCW (left) of the target → needs to turn right.

    At inference: the user feeds cmd[2] directly.  Holding cmd[2] = constant gives
    a proportional heading correction = approximately constant turn rate.

    Set heading_command=False and rel_heading_envs=0.0 in the cfg (we handle
    heading internally).  ang_vel_z range in cfg is used as the clip limit for cmd[2].
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Sampled target heading per env, world frame (rad)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        # Clip limit for cmd[2]: use ang_vel_z[1] from cfg (the positive bound)
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        # Sample random world-frame target heading uniformly in [-π, π]
        self._target_heading_w[env_ids] = (
            torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        )
        # Zero ang_vel slot; _update_command will fill it each step
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # Do NOT call super()._update_command() — it would run the heading
        # proportional controller and overwrite cmd[2] with a yaw rate.
        # Instead recompute heading error from scratch each step.
        quat = self.robot.data.root_link_quat_w  # (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # Positive = target is CCW (left) of robot → turn left. Standard convention.
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for heading command


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        return RelativeHeadingVelocityCommand(self, env)


class SpawnHeadingVelocityCommand(RelativeHeadingVelocityCommand):
    """Expose error from the episode's spawn heading in command slot 2.

    Unlike :class:`RelativeHeadingVelocityCommand`, this does not ask the robot
    to turn toward a random world heading.  Every resample captures the current
    heading, then subsequent drift appears as a signed, closed-loop correction
    command.  This gives a running actor the missing information required to
    steer back without changing the shared 61D observation layout.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._heading_max = cfg.heading_error_clip

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self._target_heading_w[env_ids] = self.robot.data.heading_w[env_ids]
        self.vel_command_b[env_ids, 2] = 0.0


@_dataclass(kw_only=True)
class SpawnHeadingVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    heading_error_clip: float = 1.0

    def build(self, env: ManagerBasedRlEnv) -> "SpawnHeadingVelocityCommand":
        return SpawnHeadingVelocityCommand(self, env)


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for reducing heading error when cmd[2] encodes heading error.

    Returns exp(-cmd[2]² / std²).
    - At error = 0 (on heading): reward = 1.0.
    - At error = std: reward ≈ 0.37 (strong gradient).
    - At error = 1.0 rad with std=0.5: reward ≈ 0.018 (nearly zero).

    std=0.5 rad (≈28°) gives a meaningful gradient across the expected range.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error ** 2) / (std ** 2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """Reward feet air time only when pushing (cmd_x > 0).

    Encourages the robot to lift each foot during the recovery phase of the
    skating stroke rather than dragging it on the ground.
    Scaled by cmd_x so the incentive grows with push intensity.

    When ``vel_gate_ref`` > 0 the reward is also multiplied by a forward-speed
    gate so lifting feet without propelling the body (tap-dancing on the spot)
    earns nothing. ``threshold_min`` sets the shortest swing that counts — raise
    it to forbid a frantic high-cadence flutter.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """0→1 ramp in body forward speed: 0 when standing still, 1 at/above v_ref.

    Used to gate stride-shaping rewards so that stepping which does NOT propel
    the body (e.g. tap-dancing on the spot) earns nothing — the reward for the
    FORM of a stride is only paid when the stride actually does its JOB (moving
    forward). Returns None when disabled (v_ref <= 0)."""
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """Reward single-support (a skating stride), mildly discourage the swizzle.

    Real skating is a STRIDE: push off one blade while the other swings, i.e.
    single support that alternates left/right. A symmetric swizzle keeps BOTH
    blades grounded the whole time and still spins the wheels, so wheel_speed
    alone converges to it.

    Per step, counting blades in contact:
      - exactly 1 blade down (stride)  → + clamp(cmd_x,0) · gate
      - 2 blades down    (double supp) → − double_penalty · clamp(cmd_x,0)
      - 0 blades down    (flight/hop)  →  0

    The POSITIVE single-support reward is gated by forward speed (``vel_gate_ref``)
    so stepping in place (no propulsion) earns nothing — kills the tap-dance hack.
    The double-support penalty is small and UNGATED: brief double support during
    weight transfer / push-off is NORMAL skating, so we only lightly discourage
    PERMANENT double support (the swizzle) rather than forbid it. The real
    anti-swizzle signal is skating_air_time — the swizzle never lifts a foot.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=(r".*(hip|knee|ankle).*",)
    ),
) -> torch.Tensor:
    """Reward the GLIDE phase of a stride: coast on ONE blade with quiet legs.

    Nothing else rewards gliding — skating_air_time pays each swing, so the policy
    maximises swing FREQUENCY (frantic kicking). This term pays staying on one
    foot and coasting, giving the policy a reason to slow down and commit to each
    stroke:

        reward = single_support · forward_gate · stillness · (cmd_x >= 0)

    - single_support: exactly ONE blade in contact. REQUIRED — this is the fix vs
      the earlier broken glide, which omitted it and let a two-blade swizzle-coast
      farm the reward and regress the gait.
    - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 0 when not moving forward.
    - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → high only when legs
      are quiet; a kick (fast joint motion) gets ~0, so only a real glide pays.
    - active on push/coast only (cmd_x >= 0); silent on brake.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(
        torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    stillness = torch.exp(-joint_vel_sq / stillness_std ** 2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"),
) -> torch.Tensor:
    """Reward left/right legs mirroring — the swizzle's defining symmetry.

    The robot uses mirrored L/R sign conventions, so a bilaterally-symmetric config
    satisfies q_left + q_right ≈ 0 per matched joint pair. Returns
    ``-mean_pairs |q_left + q_right|`` (L1, constant gradient); use with a POSITIVE
    weight so asymmetry is penalised and the symmetric swizzle is favoured. L/R index
    pairs are resolved once by name and cached on env.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """Reward BOTH blades in contact — a classic swizzle stays grounded (no lifting).

    Mirror of single_support_reward but rewarding double support (n_contact >= 2),
    scaled by |cmd_x| so it shapes the push phase in EITHER direction (forward or
    backward — the swizzle env drives cmd_x < 0 as "go backward").
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize lopsided left/right foot usage (one blade doing most of the work).

    With symmetry augmentation OFF, nothing stops the policy learning an asymmetric
    stride that pushes mostly with one leg — which veers and destabilises (esp. at
    launch). Accumulates per-foot swing time over the episode and penalises the
    normalised imbalance |L - R| / (L + R):
      - balanced alternating stride  -> ~0 (no penalty)
      - one foot swinging much more   -> ~1 (max penalty)
    Only the CUMULATIVE imbalance is penalised — the instantaneous single-support
    asymmetry of a real stride (one foot swinging now) is fine.
    """
    from mjlab.sensor import ContactSensor
    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time  # (N, num_feet)
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward holding the SPAWN heading (go straight) — corrective, angle-based.

    Rewards the yaw ANGLE staying near the heading captured at reset:
        reward = exp(-wrap(yaw - yaw_spawn)² / std²)

    This is the RIGHT way to go straight (vs penalising yaw-RATE, which just tells
    the policy 'never turn' → it can't steer back and drifts open-loop). Here a
    drift lowers the reward, and the policy is free to yaw back to recover it.

    The spawn heading is captured per-env on the first step(s) after reset
    (episode_length_buf <= 1), when the robot is still ~at its spawn pose. Reads
    root_link_quat_w, which is fresh at reward time (post physics step). Heading-
    invariant: the reference is each env's own random spawn yaw, so it works with
    the full-circle yaw randomisation at reset.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-π, π]
    return torch.exp(-(err ** 2) / std ** 2)


def action_over_limit_penalty(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    overshoot: float = 0.3,
) -> torch.Tensor:
    """Penalise commanding a joint target beyond its hard limit (+ overshoot).

    Policy-side deterrent against over-driving a joint onto its mechanical stop:
    e.g. hip_roll has a ±0.38 rad limit but a ±10 rad ctrlrange, so the low-kp
    servo can be commanded far past the stop to slam it with max torque — a
    fragile sim-only trick that will not transfer.

    Reads the commanded target (raw_action · scale + offset) and penalises only
    the part BEYOND (hard_limit + overshoot):

        penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)

    Unlike a qpos-limit penalty, this fires on the COMMAND, not the joint
    position — so the joint may still reach its full range (command ≈ limit) and
    no usable amplitude is stolen. Because it constrains the policy's OUTPUT, the
    learned behaviour is baked into the network and transfers to deployment
    WITHOUT any env-side action clip (which would only exist in sim → mismatch).
    ``overshoot`` gives the low-kp servo the headroom to reach near-limit targets
    under load; only the wild over-drive past that is penalised.
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset  # (B, action_dim) abs targets
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]  # (B, action_dim, 2)
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",)),
) -> torch.Tensor:
    """Reward leaning slightly forward when pushing, to counteract the backward
    torque from skating strokes.

    Uses projected_gravity_b x-component as a pitch proxy:
      forward_lean = -gravity_b[:, 0]  (positive when leaning forward)

    Only fires when cmd_x > 0. Peaks at target_pitch radians of forward lean.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std ** 2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """Phase-encoding command for the ground pick / sit-stand tasks.

    Replaces the velocity command with a cyclic phase signal:
        command = [cos(2π*phase), sin(2π*phase), 0]

    Phase ∈ [0, 0.5]: approach (go down).
    Phase ∈ [0.5, 1.0]: return (come back up).

    Phase is randomized per environment on episode reset to decorrelate envs.
    Period defaults to 4s; override via the cfg.period field (sitstand uses 8s
    for a slower, gentler sit-down).
    """

    PERIOD: float = 4.0  # default; cfg.period overrides

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # When False, each episode starts at phase 0 (standing) instead of a
        # random phase. Matches the runtime, where the button starts the cycle
        # at phase 0 from standing. Default True keeps the historical ground_pick
        # behavior (random phase to decorrelate envs).
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))
        self._wrap_phase = bool(getattr(cfg, "wrap_phase", True))

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        next_phase = self._gp_phase + dt / self._period
        self._gp_phase = (
            next_phase % 1.0
            if self._wrap_phase
            else torch.clamp(next_phase, max=1.0 - 1.0e-6)
        )
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            reset_phase = getattr(self._env, "_ladder_reset_phase", None)
            if reset_phase is not None:
                self._gp_phase[env_ids] = reset_phase[env_ids]
            elif self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # Phase is continuous; no resampling needed

    def _update_command(self) -> None:
        pass  # Updated in compute()

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for ground pick


from dataclasses import dataclass as _dataclass

@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False -> each episode starts at phase 0 (standing)
    wrap_phase: bool = True  # False -> clamp at the terminal phase for reverse curricula

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)


# --------------------------------------------------------------------------- #
# Unified pose command machinery                                               #
# --------------------------------------------------------------------------- #
#
# Background: we deprecated the old NeckOffsetJointPositionAction +
# disturbance-randomization approach (where head/body movement was an external
# perturbation the policy was supposed to be robust to). That trained a weak,
# indirect signal — see `project_neck_offset_decoupling.md` for the
# post-mortem.
#
# Replacement: head and body pose are now *commands* — direct, dense policy
# inputs with tracking rewards. At deployment, the runtime feeds those slots
# with whatever pose the user requests; at training, they're sampled uniformly
# from per-dim ranges (kept non-zero from step 0 so input neurons stay alive)
# and ramped via curriculum.
#
# Layout, unified across all microduck policies for runtime obs compatibility:
#   command vector (13D) = [vx, vy, vtheta,           ← "twist" (velocity)
#                           neck_pitch, head_pitch,   ← "head_pose" (deltas)
#                           head_yaw, head_roll,
#                           body_x, body_y, body_z,   ← "body_pose" (deltas)
#                           body_roll, body_pitch, body_yaw]
# Total policy obs becomes 61D (51 - 3 + 13).
# --------------------------------------------------------------------------- #


from dataclasses import dataclass, field


class UniformPoseCommand(CommandTerm):
    """Generic N-dim uniform pose command.

    Samples each dim independently uniform in cfg.ranges[i] = (lo, hi) and holds
    the value between resamples. No metrics, no debug viz — keep it lightweight
    since we have many of these.
    """

    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(lo, hi)
        # Explicit zero-command bucket. Uniform sampling essentially never
        # produces the all-zero command, so the deployment idle case ("hold the
        # nominal pose") would otherwise be absent from training (velocity
        # body-control run-1 lesson: the policy only stood still when a command
        # was present).
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Per-dim uniform ranges; builds a UniformPoseCommand."""
    # Tuple of (lo, hi) per dim. Length defines the command dim.
    ranges: tuple[tuple[float, float], ...] = ()
    # Probability that a resample yields the exact all-zero command.
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        return UniformPoseCommand(self, env)


def zero_command_padding(
    env: ManagerBasedRlEnv,
    dim: int,
) -> torch.Tensor:
    """Constant-zero obs term of width `dim`.

    Used by envs that don't actively track head/body commands (e.g. sitstand,
    ground_pick) but still need the unified 61D obs shape so the runtime can
    feed all policies with the same buffer layout.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    std: float = 0.5,
    fine_std: float | None = None,
    fine_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-joint Gaussian reward for matching commanded neck/head deltas.

    Mean over the 4 neck/head joints of exp(-(err/std)^2). Result is (N,) in
    [0, 1]. Mean form (vs sum-of-squares) keeps gradient alive when only one
    joint is off — vs SOS where a single big error kills the whole reward.

    `std` is the per-joint tolerance: at err=std the per-joint reward is 1/e
    (~0.37). Pick std on the order of the command range so the gradient
    doesn't die as the curriculum widens.

    `fine_std` (optional) blends in a second, narrow Gaussian:
    (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²).
    Rationale: a single wide std (0.5 rad ≈ 29°) makes small errors nearly
    free — a 10° gravity sag on the heavy head costs ~0.03 reward, so the
    policy lets it droop. The narrow component (~0.1 rad) prices those small
    errors while the wide one keeps gradient alive at far commands during
    curriculum widening.

    cmd has shape (N, 4) = deltas from default joint positions in the order
    [neck_pitch, head_pitch, head_yaw, head_roll].

    On backlash models the measured angle is qpos[servo] + qpos[backlash] —
    the OUTPUT link, which is also what the encoder obs
    (joint_pos_rel_backlash) reports. Measuring the servo alone would let the
    head droop the backlash play reward-free AND penalize the policy for
    compensating it (servo biased up = servo-side "error"). On models without
    passive_*_backlash joints the mask is 0 and this reduces to the servo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor(
            [0 if b is None else b for b in bl], device=env.device, dtype=torch.long
        )
        env._head_pose_bl_mask = torch.tensor(
            [0.0 if b is None else 1.0 for b in bl], device=env.device
        )

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-(err / std) ** 2)
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(
            -(err / fine_std) ** 2
        )
    return per_joint.mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NaN-safe wrappers for the sensor-derived critic observations.
#
# `robot_state_is_nan` covers joint + root state, so every obs derived from
# those is protected by the reset it triggers. The three terms below are NOT:
# they read sensor data (raycast heights, contact air-time, contact forces),
# which MuJoCo can return non-finite for while the integrated robot state is
# still clean. They are critic-only, so a single sanitized step costs the
# policy nothing, whereas letting the value through kills the entire run via
# rsl_rl's check_nan. Sanitizing here does not hide real physics blowups —
# those still terminate through nan_state and show up as
# Episode_Termination/nan_state in wandb.
# ─────────────────────────────────────────────────────────────────────────────


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_contact_forces` (see note above)."""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_height` (see note above)."""
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_air_time` (see note above)."""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    gate_height_low: float | None = None,
    gate_height_high: float = 0.11,
    gate_tilt_full_deg: float = 20.0,
    gate_tilt_zero_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the time-averaged (DC) neck/head tracking error: -mean(|EMA(err)|).

    Companion to ``head_pose_tracking``, which scores the INSTANTANEOUS error.
    Why a separate DC term instead of just tightening that Gaussian's std:
    walking unavoidably shakes a head that is 38% of the robot's mass, so an
    instantaneous tight-tolerance term is a permanent tax on walking that no
    policy can escape — measured at ~0.77/step against an air_time reward of
    ~1.01/step, which is exactly what made velocity run 2026-08-20 abandon
    stepping altogether (wandb 5yay13u4). The steady-state droop IS escapable:
    the policy can bias its neck command up to cancel gravity sag. Averaging
    over ``tau_s`` lets the oscillation cancel and prices only the bias.

    L1 (not Gaussian) on purpose: the gradient stays constant at large bias,
    where a tight Gaussian would be flat and dead.

    On backlash models the measured angle reads through the play, matching
    head_pose_tracking and the encoder obs.

    ``gate_height_low`` (optional): upright gate for recovery envs (standup /
    velstand), same smoothstep shape and semantics as body_ang_vel_at_height —
    zero below gate_height_low or above gate_tilt_zero_deg tilt, full above
    gate_height_high and below gate_tilt_full_deg. The gate multiplies the
    ERROR feeding the EMA (not just the output): while fallen/rising the EMA
    sees zero and decays, so arriving upright starts the bias clock from ~0
    instead of charging the whole ground phase's accumulated error at the
    finish line — that would be a reward wall right before recovery completes,
    the exact failure mode of the retired head_impact_penalty. The output is
    gated too, so a fresh fall stops the charge immediately.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        # Share the id cache with head_pose_tracking (either may run first).
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = (
        joint_pos[:, neck_ids]
        + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(
            asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            nan=0.0,
        )
        t = torch.clamp(
            (z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
            0.0, 1.0,
        )
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp(
            (gate_tilt_zero_deg - tilt_deg)
            / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6),
            0.0, 1.0,
        )
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # Freshly reset envs: drop the previous episode's accumulated bias.
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.02,
    z_std: float = 0.01,
    angle_std: float = math.radians(8),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean of 6 per-axis Gaussian rewards for tracking commanded body pose.

    cmd has shape (N, 6) = [x, y, z, roll, pitch, yaw] all as deltas from the
    nominal standing pose (xy delta from spawn origin, z delta from
    nominal_height, angles delta from upright = 0).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # Position relative to env spawn origin. nan_to_num because MuJoCo can
    # produce NaN on contact instability and we don't want to taint the reward.
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    # ZYX Euler from quat.
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def termination_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """Mutate a termination term's params at scheduled steps.

    TerminationManager keeps its own deepcopy of the cfg dict, so the live
    term_cfgs list must be edited directly — env.cfg.terminations is a no-op.
    Useful for disabling a termination later in training (e.g. set
    bad_orientation's limit_angle to pi at iter N so the robot can fall over
    without ending the episode and learn to recover).

    param_stages: list of {step: int, params: dict}. The dict is shallow-merged
    into the live term_cfg.params at the latest matching stage.
    """
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # Term was removed (e.g. play mode disables fell_over entirely).
        return torch.tensor(0.0)
    idx = tm._term_names.index(term_name)
    term_cfg = tm._term_cfgs[idx]

    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)

    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
) -> torch.Tensor:
    """Locomotion-aware 6D body pose tracking.

    Same shape as body_pose_tracking_6d (6D cmd, mean of 6 Gaussians), but
    x/y/yaw are measured *relative to the feet support polygon*, not the spawn
    origin. This makes the reward meaningful while the robot walks (or stands):

      x, y  : trunk position − feet-centroid, rotated into trunk body frame.
              dx = +0.02 means "lean trunk 2 cm forward of foot centroid."
      z     : trunk world height (− nominal_height) — locomotion-neutral.
      roll  : trunk world roll                     — locomotion-neutral.
      pitch : trunk world pitch                    — locomotion-neutral.
      yaw   : trunk world yaw − circular-mean(feet site yaws). dyaw = +0.3 rad
              means "twist the trunk 17° relative to where the feet point."

    The body_pose_tracking_6d reward measures x/y/yaw vs spawn origin / world
    yaw, which kills the gradient as soon as the robot translates or turns. This
    version stays meaningful regardless of where in the world the robot is.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll  = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # Feet centroid in world frame.
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]   # (N, 2, 3)
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids] # (N, 2, 4)
    feet_centroid = foot_pos.mean(dim=1)                     # (N, 3)

    # Trunk xy in body frame relative to feet centroid (rotate world Δxy by −yaw).
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body =  cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # Z relative to spawn-origin terrain height (still in world).
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Feet yaws → circular mean. NOTE: this depends on the site orientation
    # matching the foot pointing direction; if the site frame is rotated, this
    # yaw reference may have an offset (constant per-env, so dyaw=0 still maps
    # to "feet-aligned").
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))  # (N, 2)
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err     = x_body - dx
    y_err     = y_body - dy
    z_err     = z_world - (nominal_height + dz)
    roll_err  = roll  - droll
    pitch_err = pitch - dpitch
    yaw_err   = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-(x_err / xy_std) ** 2)
    r_y = torch.exp(-(y_err / xy_std) ** 2)
    r_z = torch.exp(-(z_err / z_std) ** 2)
    r_r = torch.exp(-(roll_err  / angle_std) ** 2)
    r_p = torch.exp(-(pitch_err / angle_std) ** 2)
    r_w = torch.exp(-(yaw_err   / angle_std) ** 2)

    # Per-axis weighted mean. Pass axis_weights=(0,0,1,1,1,1) to disable xy
    # tracking — useful when xy lean is mechanically coupled to pitch/roll on
    # the robot, making independent xy commands a noise source rather than a
    # learnable objective.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx*r_x + wy*r_y + wz*r_z + wr*r_r + wp*r_p + wyaw*r_w) / max(total_w, 1e-6)

    # Optional gate: when vel_gate_command_name is set, scale the reward by a
    # Gaussian on the velocity command's magnitude. With vel_gate_std ≈ 0.1,
    # the gate is ~1 when commanded velocity is 0 and decays to ~exp(-9)≈0
    # by |vel_cmd|≥0.3 — body tracking only meaningfully contributes when the
    # robot is supposed to be standing still. Avoids the tracking vs walking
    # conflict that prevented the previous run from learning either well.
    if vel_gate_command_name is not None:
        # Gate on commanded LINEAR velocity only (xy) — turning in place still
        # leaves body pose meaningful, but walking forward/sideways doesn't.
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)  # (N, 3)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-(vel_mag / vel_gate_std) ** 2)
        reward = reward * gate

    return reward


def pose_command_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """Ramp a UniformPoseCommand's per-dim ranges over training.

    range_stages: list of {step: int, ranges: tuple[(lo, hi), ...]}.
    The first stage applies before its step; latest passed stage wins.
    Always uses the live CommandManager term cfg (NOT env.cfg.commands) so
    updates take effect — CommandManager keeps its own term refs and reads
    `term.cfg.ranges` each resample.
    """
    del env_ids

    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"
    cfg = term.cfg  # type: ignore[assignment]

    current = range_stages[0]["ranges"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["ranges"]

    cfg.ranges = tuple(current)
    # Return the max abs range as a scalar for wandb visibility.
    max_abs = max((max(abs(lo), abs(hi)) for lo, hi in current), default=0.0)
    return torch.tensor(max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# Gait-shaping penalties ported from mjlab_microban (microban velocity recipe).
# ─────────────────────────────────────────────────────────────────────────────
def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize feet in the air when the commanded speed is below threshold.

    Discourages marching in place when the robot should stand still. Returns the
    count of airborne feet per environment (use with a negative weight).
    Ported from mjlab_microban.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the feet getting too close to each other in the horizontal plane.

    Returns ``clamp(min_dist - d, min=0)`` per env (use with a negative weight),
    where ``d`` is the horizontal (xy) distance between the two foot sites.
    Ported from mjlab_microban. Not wired into velocity yet — pinned for later.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # (N,)
    return torch.clamp(min_dist - dist, min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Non-accumulating domain randomization (restore-nominal-then-apply).
#
# The stock mdp.randomize_field with operation="add"/"scale" + mode="reset"
# reads the CURRENT model value and applies the op to it, with no restore to
# nominal — so on every episode reset the perturbation STACKS on the previous
# one and the parameter random-walks away from nominal over training. For
# body_ipos (CoM) this was the long-standing microduck instability: the CoM
# drifted centimeters off-center over hundreds of resets → progressively
# unbalanced robot → falls more → reward/episode-length collapse after the early
# peak. These functions mirror randomize_mass_and_inertia: cache the nominal
# once, restore it before each draw, then apply a freshly-sampled perturbation —
# so it is re-sampled per episode but never accumulates.
# ─────────────────────────────────────────────────────────────────────────────
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    field: str = "body_ipos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Randomize body CoM (body_ipos) per episode WITHOUT accumulating.

    Drop-in replacement for the buggy mdp.randomize_field(add, body_ipos, reset).
    ``ranges`` is (lo, hi) applied to all 3 CoM axes; the com_range curriculum
    updates this same ``ranges`` param. ``field`` is declared so the event can run
    with ``domain_randomization=True`` (mjlab reads params["field"] to expand that
    model field per-env).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # Key the cache by (field, body set): multiple randomize_com events can share
    # the same field (e.g. trunk + head both randomize body_ipos) and must NOT
    # collide on a single _original_body_ipos attr — their body counts differ.
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # Cache nominal on first call (model[0] is still nominal at that point).
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    # Restore nominal first (prevents accumulation), then add a fresh offset.
    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    field: str,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Scale a per-dof model field (e.g. dof_frictionloss/dof_damping) per episode
    WITHOUT accumulating: restore nominal, then apply a fresh scale.

    ``field`` doubles as the domain_randomization field name. NOTE: under the BAM
    actuator, dof_frictionloss and dof_damping are zeroed in edit_spec (BAM models
    friction itself), so scaling them is a no-op — these only matter with the XML
    position actuator. Kept correct to avoid the accumulation footgun if re-enabled.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))


# =============================================================================
# BallKick task — ball reset event, kick rewards, critic-only ball observations
# =============================================================================


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env world-frame kick direction (XY unit vector), lazily allocated.

    Set by ``reset_ball_in_front_of_foot`` to the robot's forward direction at
    episode reset. Frozen for the episode so the policy can't redefine "forward"
    by turning after the kick.
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    offset: tuple = (0.09, -0.042),
    noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    asset_name: str = "ball",
):
    """Place the ball in front of the (right) foot; store the kick direction.

    ``offset`` is the nominal ball-center position in the robot's yaw frame:
    at HOME the right foot is centered at (0, -0.042) with the toe tip at
    x≈0.034, so (0.08, -0.042) puts a 35mm-radius ball ~1cm in front of the
    toe. ``noise_xy`` (uniform ± per axis) is the placement DR: the policy is
    BLIND to the ball, so this is what forces a swing that works across the
    real-world placement error.

    Reads the robot root from qpos directly (root_link_pos_w lags until the
    next forward()); must be registered AFTER reset_base / set_ground_state
    (events run in dict insertion order) so the robot pose is final.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0  # identity quat
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids
    )

    kick_dir = _ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def ball_forward_velocity(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    max_speed: float = 5.0,
) -> torch.Tensor:
    """Ball XY velocity along the per-env kick direction, clamped to [0, max].

    Dense and linear-in-speed up to ``max_speed``: every extra bit of forward
    ball speed pays more every step the ball keeps rolling, so exploration
    nudges bootstrap the kick with no peak-detection machinery. Backward /
    lateral ball motion earns 0 rather than a penalty — a mis-hit shouldn't
    scare the policy away from contacting the ball at all.

    With ``max_speed`` set to a TARGET speed (rather than a large cap), pair
    with ``ball_speed_overshoot_penalty``: the reward saturating at the target
    alone does NOT remove "harder is better" — a harder kick keeps the ball
    at/above the cap for more steps, so the rolling-time integral still grows
    with strike speed. The overshoot penalty is what makes the target the
    actual optimum.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    target_speed: float = 1.0,
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """Ball forward speed in excess of ``target_speed`` (linear, ≥ 0).

    Companion to ``ball_forward_velocity`` for a target-speed kick: below the
    target this is 0 (the capped linear reward provides the upward gradient);
    above it, each m/s of overshoot costs linearly every step it persists.
    Keep this term's |weight| BELOW the capped reward's weight so the combined
    landscape peaks at the target with a gentler slope on the overshoot side —
    erring slightly hard must stay cheaper than not kicking at all.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Binary reward: 1 while the sensed foot touches the terrain.

    Single-foot variant of ``feet_grounded_reward`` — used to pin the SUPPORT
    foot during the kick (anti-hop): swinging the right leg is free, lifting
    the left foot costs this reward every step.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball position relative to the robot root, in the robot's base frame.

    CRITIC-ONLY observation (asymmetric actor-critic): the deployed policy has
    no ball sensing, so the actor must stay blind to the ball — the critic can
    still use it to predict the kick payoff.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """Ball linear velocity in the robot's base frame. CRITIC-ONLY (see above)."""
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Tâche SPIN — rotation rapide sur place sur rollers                            #
# --------------------------------------------------------------------------- #
# Enveloppe de phase : la commande du slot bouton porte une phase, qui pilote
# une VITESSE DE LACET cible en trapèze (et non une pose comme le crouch).
#   [0, accel_end)        0.5 s   0 -> rate_max    (lancement)
#   [accel_end, hold_end) 1.6 s   rate_max         (régime)
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    (freinage)
#   [brake_end, 1.0)      1.4 s   0                (repos debout)
# Aire sous l'enveloppe sur un cycle = 2.1 * SPIN_RATE_MAX rad. À 3.0 rad/s :
# 2.1 * 3.0 = 6.3 rad ~ 1 tour (et non ~2, comme avec l'ancienne cible 6.0).
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Vitesse de lacet cible (rad/s, positive = anti-horaire) le long de la phase."""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(
        brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w
    )
    return w


def spin_gate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Porte de shaping dans [0,1] = enveloppe normalisée.

    Vaut 0 sur tout le segment de repos : les amorces (ciseau des jambes,
    différentiel des roues) ne s'appliquent que pendant lancement + régime, donc
    le robot revient en station neutre avant de rendre la main à la policy roller.
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """Récupère la phase [0,1) depuis la commande [cos(2πφ), sin(2πφ), 0] du slot."""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(
    omega_z: torch.Tensor, omega_target: torch.Tensor, std: float
) -> torch.Tensor:
    """Gaussienne sur l'erreur de vitesse de lacet (fonction pure, testable)."""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 1.5,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Objectif principal du spin : suivre la vitesse de lacet cible ω*(φ).

    ω_z est pris en repère corps (c'est ce que voit le gyro de l'IMU, donc ce que
    la policy observe). Une rotation dans le mauvais sens est plus punie que
    l'immobilité, la gaussienne étant centrée sur une cible positive.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 : gradient constant vers la cible même quand la gaussienne
    de `spin_rate_track` sature loin de la cible. À utiliser avec un poids
    POSITIF (la valeur retournée est déjà négative)."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # atténuation du coût de dérive pendant le lancement


def spin_stay_in_place(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE,
    accel_end: float = SPIN_ACCEL_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Coût ‖v_xy‖² du tronc : tourner SUR PLACE, et tuer l'élan d'entrée.

    Pas d'état de référence (contrairement à une dérive mesurée depuis le reset),
    donc reste valide sur les 5 cycles d'un épisode. À utiliser avec un poids
    NÉGATIF.

    ATTÉNUÉ PENDANT LE LANCEMENT : sur `[0, accel_end)` le robot doit pousser au
    sol pour s'injecter du moment angulaire, et l'état d'entrée lui donne jusqu'à
    0.3 m/s qu'il est censé CONVERTIR en rotation. Facturer la translation à plein
    tarif à cet instant s'oppose donc directement à l'objectif. Le coût est
    multiplié par `launch_scale` sur ce seul segment, et vaut plein tarif ensuite
    (régime, freinage, repos) où « sur place » est le vrai critère.

    Contrairement aux autres amorces du spin, ce terme n'est PAS éteint par
    `spin_gate_by_phase` : pendant le repos on veut justement qu'il reste plein,
    puisque c'est là que le robot doit être immobile.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(
        phase < accel_end,
        torch.full_like(cost, launch_scale),
        torch.ones_like(cost),
    )
    return cost * scale


# Demi-voie mesurée sur le modèle rollers (pose HOME, sites left_foot/right_foot) :
# 0.0499 m, contre 0.03 m estimé au spec. Conséquence mécanique de SPIN_RATE_MAX
# (A1) : différentiel attendu = 2*SPIN_RATE_MAX*demi_voie/r, r = 0.0175 m.
# À l'ancienne cible 6.0 rad/s : 2*6.0*0.0499/0.0175 = 34.2 rad/s (retenu comme
# 34.0, soit +71% par rapport aux 20.0 estimés au spec -> seuil de 30% dépassé).
# À la nouvelle cible 3.0 rad/s : 2*3.0*0.0499/0.0175 = 17.1 rad/s. Laisser 34.0
# ici plafonnerait le terme à tanh(17.1/34) = 0.47 de son propre maximum, ce qui
# affaiblirait exactement le shaping qu'on veut renforcer (cf. spin_stay_in_place).
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s ; recalibré sur la demi-voie mesurée et SPIN_RATE_MAX = 3.0


def spin_wheel_differential_from_values(
    diff: torch.Tensor, gate: torch.Tensor, omega_scale: float
) -> torch.Tensor:
    """Fonction pure : tanh du différentiel de roues, portée par gate, clampée ≥ 0."""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    omega_scale: float = SPIN_WHEEL_OMEGA_SCALE,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Récompense la rotation EN ROULEMENT (et non en patinage).

    Pour un spin anti-horaire, le patin gauche recule et le droit avance ; les 4
    roues tournant positif en marche avant, cela donne ω_D − ω_G > 0. Le tanh
    sature à `omega_scale` pour éviter la course à la vitesse de roue.
    """
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(
        omega_right - omega_left, gate, omega_scale
    )


def spin_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Les deux lames au sol pendant le spin — empêche « je saute et je vrille ».

    Variante de `grounded_reward` du swizzle, qui n'est pas réutilisable ici :
    elle se pondère par cmd_x, qui vaut cos(2πφ) sur la commande de phase.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_pitch", "knee"),
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """Amorce le CISEAU des jambes (une avant / une arrière) pendant le spin.

    Le robot a des conventions de signe MIROIR gauche/droite : une pose
    symétrique satisfait q_G + q_D ≈ 0 (cf. `leg_symmetry_reward`), donc le
    ciseau satisfait q_G ≈ q_D. On retourne `gate(φ) · (−mean|q_G − q_D|)` — à
    utiliser avec un poids POSITIF, décroissant par curriculum : l'amorce
    s'efface pour laisser la policy affiner son propre geste.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor


# =============================================================================
# Backlash model — encoder-through-backlash joint observations
# =============================================================================
# The backlash model (robot_allcollisions_backlash.xml) puts an unactuated
# ``passive_<joint>_backlash`` hinge in series with each servo joint. The link
# angle is qpos[servo] + qpos[backlash], and the real encoder sits on the
# OUTPUT side of the play — it reads the sum. These obs replace joint_pos_rel /
# joint_vel_rel in backlash tasks (see tasks/backlash.py) so the policy sees
# exactly what the runtime will feed it. The asset_cfg regex is expected to
# select only the servo joints (the usual ``^(?!passive_).*``).


def _backlash_encoder_ids(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) — cached per (entity, joint selection).

    mask is 1.0 where a matching passive_<name>_backlash joint exists, so the
    same obs functions run unchanged on models without backlash joints.
    """
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    if isinstance(jnt_ids, slice):
        main_ids = list(range(len(names)))[jnt_ids]
    else:
        main_ids = [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (
        torch.tensor(main_ids, dtype=torch.long, device=device),
        torch.tensor(bl_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )
    cache[key] = out
    return out


def joint_pos_rel_backlash(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_pos_rel where the encoder reads through the backlash hinge.

    Returns (qpos[servo] + qpos[backlash]) - default[servo]. With biased=True
    the per-env encoder-calibration bias is applied to the servo reading (one
    encoder per servo → one bias per joint; the backlash summand stays raw).
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_vel_rel where the encoder reads through the backlash hinge.

    The firmware derives present_velocity from encoder positions, so it also
    sees the backlash motion: qvel[servo] + qvel[backlash].
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Sit↔Stand posture command + posture-conditioned rewards (sitstand env).
#
# One policy, both directions: the command is a single sit/stand flag carried
# in the twist slot (cmd = [sit_flag, 0, 0], so "stand" is the all-zero
# command — same deployment idle as every other policy). All task rewards
# below select their target (SIT keyframe + SIT_Z vs HOME + STAND_Z) from the
# live command, per env, so the same reward stack drives the descent, the
# seated rest, the rise and the standing rest. Uses the _servo_* helpers →
# backlash-model compatible.
# ─────────────────────────────────────────────────────────────────────────────


class SitStandCommand(UniformVelocityCommand):
    """Posture command: cmd = [sit_flag, 0, 0] with dwell-time resampling and a
    SLEWED internal target blend.

    sit_flag ∈ {0.0, 1.0}. Resampled by the command manager on the cfg's
    resampling_time_range (the dwell time in each posture) and on episode
    reset. cfg.sit_prob is the probability a resample commands SIT; with the
    reset-state mix this trains all four (start-state × command) combinations,
    including "hold what you're already doing".

    ``alpha`` (0 = STAND target, 1 = SIT target) slews toward the flag at a
    constant rate (full transition in cfg.ramp_s seconds) and is what the
    posture_* rewards track. THE anti-crash mechanism: with a binary target,
    arriving early pays the full goal-state jackpot for every step saved,
    while the linear speed-cap penalties integrate to a bounded excess-
    distance cost — an instant drop beat a 1 s descent by ~7×. With the
    slewed target, being AHEAD of the ramp scores ~0 on the height/composite
    stack (z far from the commanded height), so tracking the slow setpoint IS
    the argmax; the caps remain as backstops for overshoot/bounce. The OBS
    stays the raw binary flag (deployment: runtime writes 0/1; the trained
    response to a flip is the ~ramp_s glide).

    On episode reset, alpha is initialised from the robot's ACTUAL trunk
    height, not the flag — a seated spawn must not be dragged upward by a
    stand-initialised ramp (and vice versa).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed target blend: 0 = STAND target, 1 = SIT target."""
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, 2]
            - self._env_ref.scene.terrain.env_origins[:, 2],
            nan=self._stand_z,
        )
        return torch.clamp(
            (self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0
        )

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Episode-start re-init of the blend from the ACTUAL trunk height.
        # Done here (not in reset()) because the command manager resets BEFORE
        # the set_ground_state event teleports the robot, so reset() would read
        # the pre-teleport height. On the first compute of an episode the spawn
        # state is in place.
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        # Constant-rate slew of the target blend toward the commanded flag.
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass  # No heading controller / standing-env machinery.

    def _update_metrics(self) -> None:
        pass  # No velocity-tracking metrics for a posture flag.


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    class_type: type = SitStandCommand
    # Probability that a resample commands SIT (vs STAND).
    sit_prob: float = 0.5
    # Seconds for the internal target blend to traverse STAND↔SIT in full.
    ramp_s: float = 2.0
    # Rest heights, used to initialise the blend from the spawn state.
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Target blend ∈ [0, 1] (0 = STAND, 1 = SIT) for the posture rewards.

    Uses the SitStandCommand's slewed ``alpha`` (the moving setpoint) when the
    term exposes it; falls back to the raw binary flag otherwise.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(target blend, per-env joint target) for the commanded posture.

    STAND target = default_joint_pos (HOME); SIT target = HOME with the
    keyframe overrides applied; the SLEWED blend interpolates between them,
    so mid-ramp the rewarded pose folds in sync with the descending height.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(slewed target trunk z, actual trunk z) per env."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    return target_z, z


def posture_pose_match(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian pose-match against the commanded posture's target pose."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-((joint_pos - target) / std) ** 2).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_pose_match`` (constant gradient to target)."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussian on trunk z against the commanded posture's target height."""
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-((z - target_z) / std) ** 2)


def posture_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 companion to ``posture_height_gaussian`` — the transition driver.

    While the robot rests in the *wrong* posture this charges a constant
    per-step cost (~|Δz| = 55 mm), which is what makes "ignore the command"
    a net-negative strategy in both directions.
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    sit_z: float,
    stand_z: float,
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Multiplicative goal score vs the commanded posture (height·upright·pose
    [·head]).

    The posture-conditioned version of ``standing_composite_score``: a
    deficiency in any factor collapses the whole term, so partial-sum
    compromises (plank, flop, lean) never pay. Both rest states demand an
    upright trunk, so the upright factor is posture-independent.

    ``head_std`` (optional): adds a fourth factor on the neck/head joints vs
    the ``head_pose`` command (same error convention as head_pose_tracking).
    Without it the goal state is head-blind: the trained policy rested with
    the head dangling to the floor — trunk upright, legs in pose, z on target
    all held while the head hung, costing only the light tracking term. With
    the factor, "arrived" REQUIRES the head at its commanded pose, so head
    assist stays free mid-transition (composite is ≈0 there anyway) but must
    be retracted to collect the goal reward.
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-((z - target_z) / height_std) ** 2)

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward trunk stillness while AT the commanded posture, upright.

    Generalizes ``seated_stillness`` to both rest states: exp(-(|v|/std)²)
    gated by a smoothstep on |z − commanded z| (full inside ``band_full``,
    zero beyond ``band_zero`` → inactive during transitions) and by trunk
    tilt (a tilted rest — back/face/side — earns nothing). Additionally gated
    on the target ramp being COMPLETE (|flag − alpha| small), so stillness
    never pays mid-transition. Makes "rest quietly, upright, at the commanded
    height" the peak of the stack.
    """
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_height: float,
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Upward-vz reward, active only when STAND is commanded and z < max_height.

    The standup-env lesson: destination-only rewards have zero gradient at
    zero motion, so "stay seated and eat the L1" is a local optimum — paying
    for the rise *motion* itself makes any attempt immediately positive.
    Gated off above ``max_height`` (set just ABOVE the stand target so the
    final cm still pays; gating at exactly STAND_Z parks the policy short).
    Zero whenever SIT is commanded, so it can never fight the descent.
    ``max_vz`` caps the rewarded speed (any rise ≥ the cap earns the same, so
    an explosive launch can't out-earn a gentle one).
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalty on upward trunk velocity beyond ``max_up_vel``.

    Mirror of ``trunk_downward_velocity_penalty`` for the rise: charges every
    step of a too-fast (violent) stand-up, so the explosive rise can't be
    amortised against arriving-standing reward. Zero at rest, for any rise
    slower than the cap, and for all downward motion. Introduce via
    curriculum AFTER the rise is discovered (attempt-tax lesson).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)
# ==============================================================================
# Roulade (forward roll) task — episodic dynamic maneuver
# ==============================================================================
#
# Third attempt at the roulade. What the first two taught us:
#   • origin/roulade (phase-clock + time-windowed reward stages): plateaued
#     face-down at ~90° — time windows are keyframes-in-time, campable local
#     optima (the sit/standup lesson exactly). Also integrated -ω_y as forward
#     progress, which by this codebase's own convention (face-down = +90° pitch
#     = rotation about +y, see set_random_ground_state) is the WRONG SIGN — the
#     progress reward paid for backward rotation.
#   • origin/roulade later commits (keyframe imitation): same waypoint-camping
#     family, dropped per feedback-episodic-pose-landing.
#
# This design uses the proven episodic recipe instead:
#   • ONE dense progress signal: paid INCREMENTS of the max-so-far cumulative
#     forward rotation (potential-based — a camping policy earns zero/step, a
#     full roll earns exactly 2π worth no matter the path or speed).
#   • Landing rewards (composite product, upright, height, rise velocity) are
#     gated on ROLL COMPLETION (max rotation ≥ threshold) — state-based gates,
#     not clock-based. "Do nothing" earns nothing; standing at spawn earns
#     nothing; only rolling opens the standing-attractor annuity.
#   • Reverse curriculum via mid-roll spawns (the face-up partial-roll trick
#     that fixed back-recovery): a slice of episodes starts pitched 50°–185°
#     into the roll, tucked, optionally with forward angular momentum, and the
#     rotation accumulator is initialized to the spawn angle so the progress
#     accounting stays consistent.
#
# RUN-1 LESSON (2026-08): with unsupported rotation counting and uncapped
# paid rate, the optimal policy is a violent ballistic whip ("breakdance") —
# same 2π, finishes sooner, more discounted annuity. Doesn't transfer. Fixes:
#   • SUPPORT GATE: the accumulator only integrates while some robot geom
#     touches the terrain (robot_ground_contact sensor) — a real roulade never
#     leaves the ground; airborne rotation now earns nothing and cannot open
#     the completion gate.
#   • HEAD LATCH: the landing annuity additionally requires head-ground
#     contact to have occurred while accum was in the first-quadrant window —
#     "went over the head" is a requirement, not a 0.5-weight suggestion.
#   • PAID-RATE CAP: progress increments are capped at max_paid_rate; rotation
#     faster than the cap FORFEITS the excess (not deferred), so speed no
#     longer pays. An explicit overspeed penalty backs this up.
#
# Per-env state on the env object (created lazily, reset by
# reset_roulade_state):
#   env._roulade_accum      — supported-only integral of forward pitch rate (rad)
#   env._roulade_max        — max(accum) so far this episode (progress frontier)
#   env._roulade_paid       — frontier already paid out by roulade_progress
#   env._roulade_head_latch — True once the head touched ground mid-first-quadrant

# Forward-roll sign: face-down is +90° pitch = rotation about body +y
# (set_random_ground_state convention), so forward roll = POSITIVE body-frame
# ω_y. Verified empirically (see claude_experiments smoke test): a positive
# qvel about +y pitches the robot nose-down/forward and drives accum upward.
_ROULADE_FWD_SIGN = 1.0

# Sensor names read by the accumulator update (must match the env cfg).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# Head-latch window: head-ground contact while accum is inside this window
# marks the episode as a genuine over-the-head roll. In a real roulade the
# head plants at ~60–120° of body rotation; the window is generous around it.
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# Head-top axis in jaw_soft's LOCAL frame (measured empirically 2026-08-13:
# world-up expressed in jaw_soft's frame with the robot settled at HOME).
# The latch requires this axis to point DOWN at contact — "the flat top of
# the head on the floor", not the face or the side of the shell (run-5 fix:
# the run-4 policy rolled over the shoulder, which still touched jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) threshold. Measured landmarks (trunk pitched 110°):
# passive face-plant (neck at HOME) reads +0.6, full chin-tuck (neck_pitch −1,
# head_pitch +1) reads −0.99 — 0.3 accepts partial tucks while staying far
# from any face/side contact.
_HEAD_TOP_DOWN_MIN = 0.3

# Sagittal flatness gate on the accumulator (run-5): in a clean forward roll
# the body's LATERAL axis stays horizontal the whole way — its world-z
# component is 2(q_y·q_z + q_w·q_x) ≈ 0 for ANY amount of pure pitch, and
# grows toward ±1 as the roll goes over the shoulder instead. Full rotation
# credit while the lateral axis is within ~30° of horizontal, zero beyond
# ~60°: a side roll does not count as rotation, earns no progress, and never
# opens the landing gate.
_FLAT_FULL = 0.5    # |lateral_axis_z| = sin(30°): full credit below
_FLAT_ZERO = 0.866  # sin(60°): zero credit above


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z component of the body's lateral (y) axis. 0 = flat/sagittal."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """True where the head-top axis points at the floor (dot with -z > min)."""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # z-component of R(q) @ axis_local
    axis_world_z = (
        2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    )
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate forward pitch rate into the per-env rotation accumulator.

    Step-guarded so that multiple reward terms reading the accumulator in the
    same control step don't double-integrate. The frontier (max) only moves
    forward; backward rocking (wind-up) neither pays nor un-pays.

    SUPPORT GATE (run-1 fix): rotation is integrated only while the robot
    touches the terrain — a roulade is a supported motion; ballistic flips
    accumulate nothing, so they neither get paid nor open the completion gate.

    Also latches env._roulade_head_latch when the head touches the ground
    while accum is inside the first-quadrant window — the landing annuity
    requires this, making "over the head" a hard requirement of the task.
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # Sagittal flatness gate (run-5): side/shoulder rolls don't count.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (
                env._roulade_accum < _HEAD_LATCH_HI
            )
            # Run-5: contact must be with the FLAT TOP of the head (top axis
            # pointing at the floor) — face/side shell contacts don't latch.
            env._roulade_head_latch = env._roulade_head_latch | (
                head_contact & in_window & _head_top_down(env, asset)
            )
        env._roulade_last_update_step = step


def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """Smoothstep on the progress frontier: 0 below gate_lo rad, 1 above gate_hi.

    State-based replacement for the old phase-clock landing window — it can
    only be opened by actually rotating (while SUPPORTED — the accumulator is
    contact-gated), so neither pre-roll standing nor a ballistic flip collects.
    With require_head=True the gate additionally requires the head latch —
    the episode must have rolled over the head to unlock the landing annuity.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 0.0),
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
):
    """Reset to a standing start or a mid-roll state (reverse curriculum).

    Standing bucket: upright (±standing_tilt_max pitch/roll noise), random yaw,
    HOME joints (left from reset_robot_joints), z in [standing_z_min, _max].
    ``forward_vel_range`` is the élan hook: a per-env forward base velocity
    (body x, mapped to world through the spawn yaw) sampled uniformly — 0 for
    a standstill roll, widen it later to train rolls out of a walk.

    Mid-roll bucket: pitched ``midroll_pitch_min..max`` into the roll (90° =
    on the head, 180° = on the back), random yaw, legs lerped HOME→tuck by a
    per-env factor in ``tuck_factor_range``, z in [midroll_z_min, _max],
    optional forward angular momentum from ``midroll_omega_range``. The
    rotation accumulator is initialized to the spawn pitch so progress
    accounting (and the completion gates) stay consistent: a 170° spawn only
    gets paid for the remaining ~190°.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # Pitch per bucket: small noise for standing, mid-roll angle otherwise.
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = (
        torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min)
        + midroll_pitch_min
    )
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → quaternion (yaw * pitch * roll), as in
    # set_random_ground_state.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    # Mid-roll joints: lerp HOME → tuck on the overridden joints, noise on all
    # servo joints (passive_* backlash hinges must stay at 0).
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # Mid-roll forward angular momentum: rotation about body +y. MuJoCo free
    # joint qvel[3:6] is the angular velocity in the BODY frame, so [0, ω, 0]
    # is the forward-roll axis regardless of spawn yaw (verified in the smoke
    # test — a yawed spawn still rolls straight ahead in its own frame).
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device)
            * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    # Élan hook: forward base velocity for STANDING spawns, body x → world xy
    # through the spawn yaw. (0, 0) = standstill start, disabled.
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device)
            * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    # Progress accounting: standing starts at 0, mid-roll at the spawn pitch.
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    # Head latch: mid-roll spawns are considered already past the head phase
    # (the reverse curriculum teaches roll COMPLETION; requiring a latch they
    # never had the chance to earn would keep their landing gate shut forever).
    # Standing spawns must earn it by actually rolling over the head.
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Pay increments of the progress frontier, up to one full roll.

    reward = Δ(min(max_accum, target)) / (step_dt · target), CAPPED at
    max_paid_rate rad/s of paid rotation. Nothing to farm by camping
    face-down (0/step), rocking below the frontier (0/step), or spinning past
    2π (clamped). The accumulator is support-gated, so airborne rotation pays
    nothing either.

    max_paid_rate (run-1 fix): rotation faster than the cap FORFEITS the
    excess — the paid pointer still jumps to the frontier, it just pays the
    capped amount. A violent whip therefore collects LESS total progress
    reward than a controlled ≤cap roll, instead of the same total sooner.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    sensor_name: str = "head_ground_contact",
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward head-ground contact while rotating forward mid-roll.

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down).
    The rate factor is the anti-camping guard: a face-planted robot resting its
    head on the floor has ω_fwd ≈ 0 and earns nothing — the term only pays for
    pivoting OVER the head. The top_down factor (run-5) aligns this dense
    shaping with the latch: any head contact mid-roll pays 30%, contact on the
    FLAT TOP (chin tucked) pays full — the gradient that teaches the tuck.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing_composite_score × completion gate.

    The big annuity: once the roll is (nearly) complete, every step spent
    standing at HOME pose pays — finishing on the feet and staying there
    dominates every partial outcome. Zero before gate_lo of rotation, so the
    standing spawn cannot farm it by doing nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        target_overrides=target_overrides,
        asset_cfg=asset_cfg,
    )
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear cos(tilt) × completion gate — bootstrap pull toward vertical.

    Gradient from ANY orientation (the composite is near-zero far from the
    goal), but only after the roll: before gate_lo it is exactly zero, so it
    cannot oppose the flip the way the old always-on upright term did.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(
        env, gate_lo, gate_hi, require_head=True
    )


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Broad height Gaussian × completion gate — pull up to standing height."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    g = torch.exp(-((z - target_height) / std) ** 2)
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tight-std upright × height Gaussians × completion gate — the last mile.

    Run-4 fix for the 27°-lean / 1-cm-crouch end basin: the broad landing
    composite (upright_std 0.40) scores ~0.5 at that pose, so the policy
    parks there. This is standup's two-layer lesson — the broad layers reach,
    the sharp layers finish. At 27° tilt this term scores ~0.1 (real
    gradient); at vertical it pays ~1.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    height_g = torch.exp(-((z - target_height) / height_std) ** 2)
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING height L1 below target, active only after roll completion.

    Returns −max(0, target − z) × completion_gate — use a POSITIVE weight
    (penalty sign convention). The run-3 fix for post-roll crumple-camping:
    the gated landing rewards made standing better than lying in a heap, but
    the heap itself was FREE — with only positive gated terms, "stay crumpled"
    collects ≈0/step, a comfortable basin (the standup static-sit lesson:
    the basin must be net NEGATIVE to force the rise). The gate keeps the
    roll itself untaxed, and requires the head latch so a no-roll episode
    can't be punished into weird avoidance behaviors.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """com_upward_velocity × late-roll gate — bootstrap the exit rise.

    The second half of a roulade (supine → sitting-up → standing) is the
    face-up recovery problem, and the standup env proved end-state rewards
    alone have zero gradient at zero motion there: pay for rising vz directly.
    Gated to open from ~180° (on the back) so pre-roll bobbing earns nothing,
    and gated off above max_height so it can't be farmed by hopping.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(
    env: ManagerBasedRlEnv,
    omega_max: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """max(0, |ω_y| − omega_max)² — quadratic tax on whip-speed rotation.

    Positive quantity; use a negative weight. Complements the paid-rate cap
    in roulade_progress: the cap removes the INCENTIVE to rotate faster than
    ~3 rad/s, this adds an explicit COST above omega_max, so "violent" is
    strictly worse than "controlled" rather than merely not-better. A
    controlled full roll (~2–3 rad/s average) never touches it.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(lateral-axis world-z)² — dense gradient toward a sagittal roll.

    Positive quantity; use a negative weight. Zero when standing, zero
    through an arbitrarily deep CLEAN forward roll (pure pitch keeps the
    lateral axis horizontal), up to 1 when tipped fully onto a shoulder.
    The accumulator's flatness gate makes side rolls unprofitable; this term
    adds the per-step gradient that steers back toward the plane.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rotation out of the sagittal plane: body-frame ω_x² + ω_z² (positive;
    use a negative weight). ω_y is the roll axis and stays free."""
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Body-frame lateral (y) linear velocity² — keeps the roll straight."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)


def blind_stair_zero_slots(env, width: int):
    """Keep runtime command layout without stair state, clock, or commands."""
    return torch.zeros((env.num_envs, width), device=env.device)

# --- Blind printable floor-to-desk task (2026-09-11) --------------------------
def reset_floor_desk(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    geometry: _ladder.StairLadderGeometry = _ladder.LADDER_GEOMETRY,
    level_table: tuple[dict, ...] = (
        {"riser": (0.015, 0.018), "angle": (45.0, 52.0)},
    ),
    floor_spawn_prob: float = 0.3,
    max_start_tread: int = 8,
    floor_gap_m: float = 0.04,
    position_noise: float = 0.004,
    yaw_noise_deg: float = 5.0,
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.03,
    fixed_riser_m: float | None = None,
    fixed_angle_deg: float | None = None,
    fixed_level: int | None = None,
    ladder_y_noise: float = 0.01,
    swing_spawn_prob: float = 0.0,
    swing_fraction_range: tuple[float, float] = (0.25, 0.95),
    swing_clearance: float = 0.012,
    swing_lateral_shift: float = 0.02,
    level_mix_prob: float = 0.0,
    landing_spawn_prob: float = 0.0,
    flat_range: tuple[float, float] = (0.06, 0.18),
    landing_approach_prob: float = 0.0,
    top_spawn_prob: float = 0.0,
    approach_max_below: int = 3,
    nose_jitter_m: float = 0.0,
    path_spawn_frac: float = 0.0,
    approach_swing_prob: float = 0.7,
    min_start_tread: int = 0,
) -> None:
    """Retain standard randomized proprioceptive spawn; place the entire rigid design."""
    params=locals().copy();params.pop("env");params.pop("env_ids")
    reset_stair_ladder(env,env_ids,**params)
    ids=env_ids.to(env.device,dtype=torch.long);s=_stair_state(env);orig=env.scene.env_origins[ids]
    offset=orig+torch.stack((s.x0[ids],s.y0[ids],torch.zeros_like(s.x0[ids])),dim=1)
    quat=torch.zeros(len(ids),4,device=env.device);quat[:,0]=1
    for name in ['rail_printed_structure','tread_30','tread_31']:
        env.scene[name].write_mocap_pose_to_sim(torch.cat((offset,quat),dim=1),env_ids=ids)
    for name in ['rail_left','rail_right']:
        parked=orig.clone();parked[:,2]-=2
        env.scene[name].write_mocap_pose_to_sim(torch.cat((parked,quat),dim=1),env_ids=ids)
    # Physical top heights, including the small descent from landing to desk.
    for idx,pos,top in [(30,(.442,0,.742),.744),(31,(.710,0,.7275),.740)]:
        s.tread_centre[ids,idx]=offset+torch.tensor(pos,device=env.device)
        s.tread_top[ids,idx]=orig[:,2]+top
        s.tread_target_xy[ids,idx]=offset[:,:2]+torch.tensor((.390 if idx==30 else .550,0),device=env.device)
    if not hasattr(env,'_desk_hold'):
        env._desk_hold=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        env._desk_previous=torch.zeros(env.num_envs,device=env.device)
        env._desk_hold_step=-1
    env._desk_hold[ids]=0;env._desk_previous[ids]=0
    env._floor_desk=True


def floor_desk_status(env):
    s=_stair_state(env);robot=env.scene['robot'];c=_stair_contacts(env)
    local=robot.data.root_link_pos_w-env.scene.env_origins
    x=local[:,0]-s.x0;y=local[:,1]-s.y0
    feet=robot.data.site_pos_w[:,_stair_foot_sites(env,robot),:]-env.scene.env_origins[:,None,:]
    feet_x=feet[:,:,0]-s.x0[:,None]
    upright=(-robot.data.projected_gravity_b[:,2])>math.cos(math.radians(35))
    desk=(c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)
    # Beyond the landing taper, inside the tabletop boundary, not simply high up.
    safe=desk&(feet_x>.550).all(dim=1)&(feet_x<.960).all(dim=1)&(y.abs()<.25)&upright
    safe &= robot.data.root_link_lin_vel_w.norm(dim=1)<.25
    safe &= robot.data.root_link_ang_vel_w.norm(dim=1)<2.5
    safe &= torch.isfinite(robot.data.joint_pos).all(dim=1)
    return s,robot,c,x,y,feet_x,upright,safe


def floor_desk_success(env):
    *_,safe=floor_desk_status(env)
    step=int(env.common_step_counter)
    if env._desk_hold_step!=step:
        env._desk_hold=torch.where(safe,env._desk_hold+1,torch.zeros_like(env._desk_hold));env._desk_hold_step=step
    return env._desk_hold>=round(2./env.step_dt)


def floor_desk_off_side(env,asset_cfg=_DEFAULT_ASSET_CFG,max_lateral=.10):
    s,robot,c,x,y,*_=floor_desk_status(env)
    desk_support=(c['foot_tread']==31).any(dim=1)
    bound=torch.where(desk_support,torch.full_like(y,.28),torch.full_like(y,max_lateral))
    return y.abs()>bound


def floor_desk_progress(env):
    s,robot,c,x,y,feet_x,upright,safe=floor_desk_status(env)
    p=(torch.minimum(x,feet_x.mean(dim=1)+.06)-.360).clamp(0,.220)
    p=torch.where(upright & c['foot_support'].any(dim=1),p,torch.zeros_like(p))
    delta=(p-env._desk_previous).clamp(-.01,.01);env._desk_previous=p.detach().clone()
    return torch.nan_to_num(delta,nan=0.)


def floor_desk_stance(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    std: float = 0.035,
    tau_s: float = 0.3,
    upright_std: float = 0.45,
    forward_lean_allow: float = 0.55,
) -> torch.Tensor:
    params=locals().copy();params.pop("env")
    original=ladder_stance_composite(env,**params)
    s,robot,c,x,y,feet_x,upright,safe=floor_desk_status(env)
    on_desk=(c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)
    stationary=torch.exp(-robot.data.root_link_lin_vel_w.square().sum(dim=1)/.04)
    return torch.where(on_desk,upright.float()*stationary,original)


def floor_desk_recovery_region(env):
    """Simulator-only termination allowance; never supplied to the actor.

    Restrict recovery to above the landing/table footprint. Below-surface,
    lateral and far-edge falls continue to terminate normally.
    """
    s,robot,c,x,y,*_=floor_desk_status(env)
    z=(robot.data.root_link_pos_w-env.scene.env_origins)[:,2]
    return (x>.400)&(x<.980)&(y.abs()<.280)&(z>.740)&torch.isfinite(robot.data.joint_pos).all(dim=1)


def floor_desk_recoverable_termination(env, original, original_params):
    return original(env,**original_params) & ~floor_desk_recovery_region(env)


def reset_floor_desk_recovery(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    geometry: _ladder.StairLadderGeometry = _ladder.LADDER_GEOMETRY,
    level_table: tuple[dict, ...] = (
        {"riser": (0.015, 0.018), "angle": (45.0, 52.0)},
    ),
    floor_spawn_prob: float = 0.3,
    max_start_tread: int = 8,
    floor_gap_m: float = 0.04,
    position_noise: float = 0.004,
    yaw_noise_deg: float = 5.0,
    tilt_noise_deg: float = 2.0,
    joint_noise: float = 0.03,
    fixed_riser_m: float | None = None,
    fixed_angle_deg: float | None = None,
    fixed_level: int | None = None,
    ladder_y_noise: float = 0.01,
    swing_spawn_prob: float = 0.0,
    swing_fraction_range: tuple[float, float] = (0.25, 0.95),
    swing_clearance: float = 0.012,
    swing_lateral_shift: float = 0.02,
    level_mix_prob: float = 0.0,
    landing_spawn_prob: float = 0.0,
    flat_range: tuple[float, float] = (0.06, 0.18),
    landing_approach_prob: float = 0.0,
    top_spawn_prob: float = 0.0,
    approach_max_below: int = 3,
    nose_jitter_m: float = 0.0,
    path_spawn_frac: float = 0.0,
    approach_swing_prob: float = 0.7,
    min_start_tread: int = 0,
    desk_probability: float = .5,
) -> None:
    """Near-standing reverse curriculum; bank contains explicit validated qpos."""
    params=locals().copy();params.pop('env');params.pop('env_ids');params.pop('desk_probability')
    reset_floor_desk(env,env_ids,**params)
    ids=env_ids.to(env.device,dtype=torch.long);s=_stair_state(env)
    if not hasattr(env,'_desk_spawn'):
        env._desk_spawn=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        env._desk_bank_index=torch.full((env.num_envs,),-1,dtype=torch.long,device=env.device)
        env._desk_potential_valid=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        env._desk_potential_prev=torch.zeros(env.num_envs,device=env.device)
        env._desk_potential_step=-1
    env._desk_spawn[ids]=False;env._desk_bank_index[ids]=-1;env._desk_potential_valid[ids]=False
    selected=(~s.spawn_on_floor[ids]) & (torch.rand(len(ids),device=env.device)<desk_probability)
    chosen=ids[selected]
    if not len(chosen):return
    if not hasattr(env,'_desk_pose_bank'):
        import json
        from pathlib import Path
        bank=json.loads(Path('/scratch/floor-desk/desk-recovery-training/balanced-bank.json').read_text())
        env._desk_pose_bank=torch.tensor(bank['qpos'],device=env.device)
    indices=torch.randint(len(env._desk_pose_bank),(len(chosen),),device=env.device)
    q=env._desk_pose_bank[indices].clone();orig=env.scene.env_origins[chosen]
    q[:,:3]+=orig;q[:,0]+=s.x0[chosen];q[:,1]+=s.y0[chosen]
    robot=env.scene['robot'];robot.write_root_link_pose_to_sim(q[:,:7],env_ids=chosen)
    robot.write_root_link_velocity_to_sim(torch.zeros(len(chosen),6,device=env.device),env_ids=chosen)
    joints=robot.data.default_joint_pos[chosen].clone();m=env.sim.mj_model
    for j in range(m.njnt):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
        if name.startswith('robot/') and m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:
            ji,_=robot.find_joints('^'+name.split('/')[-1]+'$');assert len(ji)==1
            joints[:,ji[0]]=q[:,int(m.jnt_qposadr[j])]
    robot.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=chosen)
    env._desk_spawn[chosen]=True;env._desk_bank_index[chosen]=indices
    # Clear climbing support memories for a legitimate tabletop start.
    s.start_tread[chosen]=31;s.foot_last_tread[chosen]=31;s.foot_support_z[chosen]=.740
    s.overstep[chosen]=False


def floor_desk_climbing_reward(env,original,original_params):
    value=original(env,**original_params)
    return torch.where(floor_desk_recovery_region(env) & (value>0),torch.zeros_like(value),value)


def floor_desk_recovery_progress(env):
    step=int(env.common_step_counter)
    if env._desk_potential_step==step:return env._desk_potential_delta
    s,robot,c,x,y,feet_x,upright,safe=floor_desk_status(env)
    z=(robot.data.root_link_pos_w-env.scene.env_origins)[:,2]
    up=(-robot.data.projected_gravity_b[:,2]).clamp(0,1)
    p=up*torch.exp(-((z-.857)/.050).square())
    p=torch.where(floor_desk_recovery_region(env),p,torch.zeros_like(p))
    delta=torch.where(env._desk_potential_valid,p-env._desk_potential_prev,torch.zeros_like(p)).clamp(-.02,.02)
    env._desk_potential_prev=p.detach().clone();env._desk_potential_valid[:]=True
    env._desk_potential_step=step;env._desk_potential_delta=torch.nan_to_num(delta)
    return env._desk_potential_delta


def floor_desk_standing_quality(env, lin_vel_std=.2, ang_vel_std=2.):
    """Bounded goal-state reward; no positive reward for lying or airborne states."""
    s,robot,c,x,y,*_=floor_desk_status(env)
    up=(-robot.data.projected_gravity_b[:,2]).clamp(0,1)
    z=(robot.data.root_link_pos_w-env.scene.env_origins)[:,2]
    feet=robot.data.site_pos_w[:,_stair_foot_sites(env,robot),2]-env.scene.env_origins[:,None,2]
    supported=((c['foot_tread']==31)&c['foot_support']).any(dim=1)
    score=torch.exp(-((1-up)/.15).square()-((z-.854)/.035).square())
    score*=torch.exp(-((feet-.740)/.015).square().sum(dim=1))
    score*=torch.exp(-robot.data.root_link_lin_vel_w.square().sum(dim=1)/(lin_vel_std**2))
    score*=torch.exp(-robot.data.root_link_ang_vel_w.square().sum(dim=1)/(ang_vel_std**2))
    return torch.nan_to_num(torch.where(supported & (up>.5) & floor_desk_recovery_region(env),score,torch.zeros_like(score)),nan=0.)


def floor_desk_disable_stair_cost(env,original,original_params):
    value=original(env,**original_params)
    return torch.where(floor_desk_recovery_region(env),torch.zeros_like(value),value)


def floor_desk_short_episode(env,seconds=3.):
    return env._desk_spawn & (env.episode_length_buf>=round(seconds/env.step_dt))



def floor_desk_leg_pose_cost(env, std=.35):
    """Positive qpos cost towards measured balanced legs, only over the desk.

    Training reward only: no action override, observation, or runtime change.
    Resolve ten leg servos by name; ignore head and passive joints.
    """
    robot=env.scene['robot']
    if not hasattr(env,'_desk_leg_pose_target'):
        import json
        from pathlib import Path
        bank=json.loads(Path('/scratch/floor-desk/desk-recovery-training/balanced-bank.json').read_text())
        m=env.sim.mj_model;ids=[];columns=[]
        servo_ids=_servo_joint_ids(env,robot)
        head_ids,_=robot.find_joints('^(neck_pitch|head_pitch|head_yaw|head_roll)$')
        for j in range(m.njnt):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
            if not name.startswith('robot/') or m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
            local,_=robot.find_joints('^'+name.split('/')[-1]+'$')
            assert len(local)==1
            if local[0] in servo_ids and local[0] not in head_ids:
                ids.append(local[0]);columns.append(int(m.jnt_qposadr[j]))
        assert len(ids)==10, ids
        targets=torch.tensor(bank['qpos'],device=env.device)[:,columns]
        assert torch.allclose(targets,targets[:1].expand_as(targets)), 'bank leg poses differ'
        env._desk_leg_pose_ids=ids;env._desk_leg_pose_target=targets[0]
    value=((robot.data.joint_pos[:,env._desk_leg_pose_ids]-env._desk_leg_pose_target)/std).square().mean(dim=1)
    return torch.where(floor_desk_recovery_region(env),value,torch.zeros_like(value))


def floor_desk_leg_action_cost(env, std=.5):
    """Positive training-only cost; executed policy actions are never replaced."""
    if not hasattr(env, '_desk_leg_action_target'):
        import json
        from pathlib import Path
        robot=env.scene['robot'];servo_ids=_servo_joint_ids(env,robot)
        head_ids,_=robot.find_joints('^(neck_pitch|head_pitch|head_yaw|head_roll)$')
        ids=[i for i,j in enumerate(servo_ids) if j not in head_ids]
        assert len(ids)==10 and len(servo_ids)==14
        held=json.loads(Path('/scratch/floor-desk/desk-recovery-training/balanced-bank.json').read_text())['held_action']
        env._desk_leg_action_ids=ids
        env._desk_leg_action_target=torch.tensor(held,device=env.device)[ids]
    value=((env.action_manager.action[:,env._desk_leg_action_ids]-env._desk_leg_action_target)/std).square().mean(dim=1)
    return torch.where(floor_desk_recovery_region(env),value,torch.zeros_like(value))


# Corrected ladder-to-table arrival: physical support without upright requirement.
def floor_desk_supported_arrival(env):
    st,robot,c,x,y,fx,up,safe=floor_desk_status(env)
    return (c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)&(x>.45)&(x<1.80)&(y.abs()<.35)

def floor_desk_supported_arrival_reward(env):
    return floor_desk_supported_arrival(env).float()


def floor_desk_recoverable_termination_at_height(env, original, original_params, min_root_z):
    """Matched termination-only desk height gate; never an actor input.

    Footprint and all rewards intentionally retain the comparison recipe.
    Set min_root_z to the compiled tabletop height for the corrected arm.
    """
    s,robot,c,x,y,*_=floor_desk_status(env)
    z=(robot.data.root_link_pos_w-env.scene.env_origins)[:,2]
    region=(x>.400)&(x<.980)&(y.abs()<.280)&(z>min_root_z)&torch.isfinite(robot.data.joint_pos).all(dim=1)
    return original(env,**original_params) & ~region

"""MDP terms for the Microduck platform jump: a two-footed long jump DOWN
from a raised block A onto a lower block B across a pit (user sketch
2026-09-18, "kinda parkour like").

The state machine is the standing long jump's (``long_jump_mdp``, v11 graded
two-foot rules) with the floor replaced by the two blocks:

    phase 0 GROUND   both feet on A, no step taken
    phase 1 FLIGHT   both feet off after a two-foot takeoff
    phase 2 LANDED   feet on B after a flight of >= 3 steps
    phase 3 INVALID  a non-foot part touched anything in flight, a foot
                     touched the floor (the pit), the flight came back down
                     on A, or a second flight started after landing

Feet on the floor at any time = the jump failed (termination + fall cost).
The policy has no vision: the gap and drop of the current episode are given
through the (otherwise zero-padded) body-command observation slot, so the
operator supplies them on hardware.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import platform_stage as ps
from mjlab_microduck.tasks.long_jump_mdp import (
    FOOT_BODIES,
    LANDING_SYNC_FREE_STEPS,
    LANDING_SYNC_ZERO_STEPS,
    MIN_FLIGHT_STEPS,
    STUCK_STEPS,
    STUCK_TILT_DEG,
    TAKEOFF_SYNC_FREE_STEPS,
    TAKEOFF_SYNC_ZERO_STEPS,
    _distance_gate,
    _forward_lateral,
    _sync_factor,
    _yaw_of,
)
from mjlab_microduck.tasks.mdp import (
    _DEFAULT_ASSET_CFG,
    _sensor_any_contact,
    _servo_joint_ids,
    standing_composite_score,
)

FEET_A = "feet_a_contact"
FEET_B = "feet_b_contact"
FEET_FLOOR = "feet_floor_contact"
BODY_A = "body_a_contact"
BODY_B = "body_b_contact"
BODY_FLOOR = "body_floor_contact"
HEAD_ANY = ("head_a_contact", "head_b_contact", "head_floor_contact")

STEP_STEPS = 10                # one foot off everything this long before takeoff = a step (0.2 s)
FEET_CREEP_M = 0.08            # feet sliding forward on A before takeoff = a step
GAP_SCALE = 0.3                # command obs: gap / GAP_SCALE, drop / DROP_SCALE
DROP_SCALE = 0.3


def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_pj", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        li = lambda v: torch.full((n,), v, dtype=torch.long, device=dev)  # noqa: E731
        st = dict(
            phase=li(0), stepped=b(), failed=b(), step_landing=b(), bonus_paid=b(), feet_ref_set=b(),
            spawn_xy=torch.zeros(n, 2, device=dev), spawn_yaw=z(),
            gap=z(), drop=z(),
            feet_ref_x=z(), ground_feet_x=z(), ground_trunk_x=z(),
            takeoff_x=z(), takeoff_feet_x=z(), max_prog=z(), paid=z(), land_dist=z(),
            launch_paid=z(), takeoff_vz=z(), sync=torch.ones(n, device=dev),
            flight_steps=li(0), landed_step=li(-1), takeoff_step=li(-1),
            last_contact_l=li(0), last_contact_r=li(0), first_touch_step=li(-1), both_down_step=li(-1),
            last_step=-1, foot_ids=None,
        )
        env._pj = st
    return st


def _found(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    """Per-foot (n, 2) contact flags of a feet sensor."""
    sensor = env.scene.sensors[name]
    found = sensor.data.found
    return found.view(found.shape[0], 2, -1).any(dim=-1)


def _any(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
    v = _sensor_any_contact(env, name)
    return v if v is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _foot_ids(env: ManagerBasedRlEnv, asset: Entity, st: dict) -> torch.Tensor:
    if st["foot_ids"] is None:
        ids, _ = asset.find_bodies(list(FOOT_BODIES), preserve_order=True)
        st["foot_ids"] = torch.tensor(ids, device=env.device, dtype=torch.long)
    return st["foot_ids"]


def _update(env: ManagerBasedRlEnv, asset: Entity) -> dict:
    """Advance the per-env jump state machine once per control step."""
    st = _st(env)
    step = int(env.common_step_counter)
    if step == st["last_step"]:
        return st
    st["last_step"] = step

    root_xy = torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0)
    fwd, _ = _forward_lateral(st, root_xy)
    feet_xy = torch.nan_to_num(asset.data.body_link_pos_w[:, _foot_ids(env, asset, st), :2], nan=0.0).mean(dim=1)
    feet_fwd, _ = _forward_lateral(st, feet_xy)

    on_a, on_b, on_floor = _found(env, FEET_A), _found(env, FEET_B), _found(env, FEET_FLOOR)
    contact = on_a | on_b | on_floor
    lc, rc = contact[:, 0], contact[:, 1]
    both, none, any_foot = lc & rc, ~lc & ~rc, lc | rc
    foot_floor = on_floor.any(dim=-1)
    foot_b = on_b.any(dim=-1)
    body_touch = _any(env, BODY_A) | _any(env, BODY_B) | _any(env, BODY_FLOOR)

    phase = st["phase"]
    p0, p1, p2 = phase == 0, phase == 1, phase == 2

    # ---- GROUND (on A): track the stance, detect steps --------------------
    on_ground = p0 & both
    st["ground_feet_x"] = torch.where(on_ground, feet_fwd, st["ground_feet_x"])
    st["ground_trunk_x"] = torch.where(on_ground, fwd, st["ground_trunk_x"])
    first_ref = on_ground & ~st["feet_ref_set"]
    st["feet_ref_x"] = torch.where(first_ref, feet_fwd, st["feet_ref_x"])
    st["feet_ref_set"] = st["feet_ref_set"] | first_ref
    lifted_l = p0 & rc & ~lc & ((step - st["last_contact_l"]) > STEP_STEPS)
    lifted_r = p0 & lc & ~rc & ((step - st["last_contact_r"]) > STEP_STEPS)
    creep = on_ground & st["feet_ref_set"] & ((feet_fwd - st["feet_ref_x"]) > FEET_CREEP_M)
    st["stepped"] = st["stepped"] | lifted_l | lifted_r | creep

    st["last_contact_l"] = torch.where(lc, torch.full_like(st["last_contact_l"], step), st["last_contact_l"])
    st["last_contact_r"] = torch.where(rc, torch.full_like(st["last_contact_r"], step), st["last_contact_r"])

    # ---- takeoff (graded two-foot sync, v11) -------------------------------
    takeoff = p0 & none
    gap_t = (st["last_contact_l"] - st["last_contact_r"]).abs()
    st["sync"] = torch.where(takeoff, _sync_factor(gap_t, TAKEOFF_SYNC_FREE_STEPS, TAKEOFF_SYNC_ZERO_STEPS), st["sync"])
    st["stepped"] = st["stepped"] | (takeoff & (gap_t >= TAKEOFF_SYNC_ZERO_STEPS))
    st["takeoff_step"] = torch.where(takeoff, torch.full_like(st["takeoff_step"], step), st["takeoff_step"])
    st["takeoff_vz"] = torch.where(takeoff, torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0), st["takeoff_vz"])
    st["takeoff_x"] = torch.where(takeoff, st["ground_trunk_x"], st["takeoff_x"])
    st["takeoff_feet_x"] = torch.where(takeoff, st["ground_feet_x"], st["takeoff_feet_x"])
    st["max_prog"] = torch.where(takeoff, torch.zeros_like(fwd), st["max_prog"])
    st["paid"] = torch.where(takeoff, torch.zeros_like(fwd), st["paid"])
    st["flight_steps"] = torch.where(takeoff, torch.zeros_like(st["flight_steps"]), st["flight_steps"])
    phase = torch.where(takeoff, torch.ones_like(phase), phase)

    # ---- FLIGHT: progress, touchdown on B, invalid arrivals ------------------
    in_flight = p1 & ~takeoff
    st["flight_steps"] = torch.where(in_flight, st["flight_steps"] + 1, st["flight_steps"])
    st["max_prog"] = torch.where(in_flight, torch.maximum(st["max_prog"], fwd - st["takeoff_x"]), st["max_prog"])
    bad_touch = in_flight & ((body_touch & ~any_foot) | foot_floor)
    touchdown = in_flight & any_foot & ~bad_touch
    real = touchdown & (st["flight_steps"] >= MIN_FLIGHT_STEPS)
    real_landing = real & foot_b                 # arrived on B
    came_back = real & ~foot_b                   # a real flight that ended on A again
    chatter = touchdown & ~real
    st["first_touch_step"] = torch.where(real_landing, torch.full_like(st["first_touch_step"], step), st["first_touch_step"])
    st["land_dist"] = torch.where(real_landing, feet_fwd - st["takeoff_feet_x"], st["land_dist"])
    st["landed_step"] = torch.where(real_landing, torch.full_like(st["landed_step"], step), st["landed_step"])
    phase = torch.where(real_landing, torch.full_like(phase, 2), phase)
    phase = torch.where(chatter, torch.zeros_like(phase), phase)
    phase = torch.where(bad_touch | came_back, torch.full_like(phase, 3), phase)

    # ---- LANDED: graded landing sync, feet must stay on B --------------------
    since_touch = step - st["first_touch_step"]
    both_down = p2 & both & (st["both_down_step"] < 0) & (st["first_touch_step"] >= 0)
    st["both_down_step"] = torch.where(both_down, torch.full_like(st["both_down_step"], step), st["both_down_step"])
    st["sync"] = torch.where(both_down, st["sync"] * _sync_factor(since_touch, LANDING_SYNC_FREE_STEPS, LANDING_SYNC_ZERO_STEPS), st["sync"])
    late = p2 & ~both & (st["both_down_step"] < 0) & (since_touch >= LANDING_SYNC_ZERO_STEPS) & (st["first_touch_step"] >= 0)
    st["step_landing"] = late & ~st["stepped"]
    st["stepped"] = st["stepped"] | late
    rehop = p2 & none
    off_b = p2 & foot_floor
    phase = torch.where(rehop | off_b, torch.full_like(phase, 3), phase)

    fell = torch.zeros_like(both)
    try:
        fell = env.termination_manager.get_term("fell")
    except Exception:  # noqa: BLE001
        pass
    st["foot_floor"] = foot_floor
    st["failed"] = fell | (phase == 3)
    st["phase"] = phase
    return st


def _landed_valid(st: dict) -> torch.Tensor:
    return (st["phase"] == 2) & ~st["stepped"]


# --------------------------------------------------------------------------
# rewards (same semantics as the long jump v11; see long_jump_mdp)
# --------------------------------------------------------------------------
def pj_progress(
    env: ManagerBasedRlEnv,
    cap_margin: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Paid increments of the trunk's max forward displacement in flight, x takeoff sync;
    clawed back on a failure or a step-landing.

    v2 (2026-09-18): the pay is CAPPED at the gap plus ``cap_margin``.  An
    uncapped distance reward made the policy fly as far as it could (0.44 m
    over a 0.30 m gap), which is exactly what makes the landing hard; the
    task only needs the gap cleared with a bit of margin.  The battery
    showed the gap was never the binding constraint - holding the landing
    is (98 % arrive on B, 15-20 % stick).
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    capped = torch.minimum(st["max_prog"], st["gap"] + cap_margin)
    inc = torch.clamp(capped - st["paid"], min=0.0)
    inc = torch.where((st["phase"] == 1) & ~st["stepped"], inc, torch.zeros_like(inc))
    st["paid"] = st["paid"] + inc
    inc = inc * st["sync"]
    claw = st["failed"] | st["step_landing"]
    clawback = torch.where(claw, st["paid"] * st["sync"], torch.zeros_like(inc))
    st["paid"] = torch.where(claw, torch.zeros_like(inc), st["paid"])
    return inc - clawback


def pj_landing_bonus(env: ManagerBasedRlEnv, max_tilt_deg: float = STUCK_TILT_DEG,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """One-shot: landed distance (m) x sync once the landing on B is STUCK (0.3 s upright, both feet)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    on_b = _found(env, FEET_B)
    due = (st["landed_step"] >= 0) & (st["landed_step"] + STUCK_STEPS == int(env.common_step_counter))
    stuck = due & _landed_valid(st) & ~st["bonus_paid"] & on_b.all(dim=-1) & (tilt < math.radians(max_tilt_deg)) & ~st["failed"]
    st["bonus_paid"] = st["bonus_paid"] | due
    return torch.where(stuck, torch.clamp(st["land_dist"], min=0.0) * st["sync"], torch.zeros_like(st["land_dist"]))


def pj_stand(env: ManagerBasedRlEnv, target_height: float, height_std: float, upright_std: float, pose_std: float,
             joint_indices: list, dist_zero: float = 0.03, dist_full: float = 0.12,
             target_overrides: Optional[dict] = None, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Standing composite on B after a valid landing (target height = trunk above the env origin)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    score = standing_composite_score(env, target_height=target_height, height_std=height_std, upright_std=upright_std,
                                     pose_std=pose_std, joint_indices=joint_indices, target_overrides=target_overrides,
                                     asset_cfg=asset_cfg)
    gate = _landed_valid(st).float() * _distance_gate(st["land_dist"], dist_zero, dist_full) * st["sync"]
    return score * gate


def pj_stand_tax(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -(height shortfall) after landing (POSITIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.clamp(target_height - z, min=0.0) * (st["phase"] >= 2).float()


def pj_body_contact_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step while a non-foot part touches a block or the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return -(_any(env, BODY_A) | _any(env, BODY_B) | _any(env, BODY_FLOOR)).float()


def pj_launch(env: ManagerBasedRlEnv, vz_cap: float = 0.6, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Potential-based launch shaping: paid increments of the max upward CoM velocity on A (capped)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    vz = torch.clamp(torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0), min=0.0, max=vz_cap)
    inc = torch.clamp(vz - st["launch_paid"], min=0.0) * (st["phase"] == 0).float()
    st["launch_paid"] = st["launch_paid"] + inc
    return inc


def pj_takeoff_bonus(env: ManagerBasedRlEnv, max_tilt_deg: float = 45.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """One-shot at a two-foot takeoff: upward CoM velocity (m/s) x sync if the trunk is upright."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    fresh = (st["takeoff_step"] == int(env.common_step_counter)) & ~st["stepped"] & (tilt < math.radians(max_tilt_deg))
    return torch.where(fresh, torch.clamp(st["takeoff_vz"], min=0.0) * st["sync"], torch.zeros_like(st["takeoff_vz"]))


def pj_dawdle_penalty(env: ManagerBasedRlEnv, after_s: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step still on A (no takeoff) after ``after_s``."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    waited = env.episode_length_buf.float() * env.step_dt > after_s
    return -((st["phase"] == 0) & waited).float()


def pj_lateral_penalty(env: ManagerBasedRlEnv, yaw_weight: float = 0.3, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -(|lateral offset| + yaw_weight * |heading change|)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    _, lat = _forward_lateral(st, torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0))
    yaw = _yaw_of(asset.data.root_link_quat_w)
    dyaw = torch.atan2(torch.sin(yaw - st["spawn_yaw"]), torch.cos(yaw - st["spawn_yaw"]))
    return -(lat.abs() + yaw_weight * dyaw.abs())


# --------------------------------------------------------------------------
# termination / fall cost / obs
# --------------------------------------------------------------------------
def pj_fell(env: ManagerBasedRlEnv, max_tilt_deg: float = 70.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk past ``max_tilt_deg``, the head on anything, or a foot on the floor (in the pit)."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    fell = tilt > math.radians(max_tilt_deg)
    for name in HEAD_ANY:
        fell = fell | _any(env, name)
    return fell | _found(env, FEET_FLOOR).any(dim=-1)


def pj_fall_cost(env: ManagerBasedRlEnv, max_tilt_deg: float = 70.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot: -1 on the step the fall termination fires (POSITIVE weight)."""
    return -pj_fell(env, max_tilt_deg=max_tilt_deg, asset_cfg=asset_cfg).float()


def pj_command_obs(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Body-command slot: [gap / 0.3, drop / 0.3, 0, 0, 0, 0] - what the robot must clear."""
    st = _st(env)
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, 0] = st["gap"] / GAP_SCALE
    out[:, 1] = st["drop"] / DROP_SCALE
    return out


# --------------------------------------------------------------------------
# spawn / reset
# --------------------------------------------------------------------------
def reset_platform_jump_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gap_range: tuple = (0.0, 0.05),
    drop_range: tuple = (0.03, 0.08),
    standing_prob: float = 0.55,
    crouch_prob: float = 0.15,
    flight_prob: float = 0.30,
    edge_dist_range: tuple = (0.03, 0.10),
    yaw_noise: float = math.radians(5.0),
    standing_tilt_max: float = math.radians(3.0),
    crouch_overrides: Optional[dict] = None,
    crouch_factor_range: tuple = (0.3, 0.7),
    flight_x_range: tuple = (-0.03, 0.10),     # relative to B's near edge
    flight_z_range: tuple = (0.14, 0.20),      # trunk above B's top
    flight_vx_range: tuple = (0.2, 0.5),
    flight_vz_range: tuple = (-0.3, 0.0),
    flight_pitch_range: tuple = (math.radians(-5.0), math.radians(10.0)),
    flight_tuck_range: tuple = (0.1, 0.4),
    flight_omega_max: float = 1.0,
    joint_noise_std: float = 0.05,
):
    """Place the blocks, spawn the robot standing / crouched on A's edge or
    mid-flight over the gap heading for B, and reset the jump state.
    Runs after reset_robot_joints (HOME joints already written)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    dev = env.device
    asset: Entity = env.scene[asset_cfg.name]
    st = _st(env)
    origins = env.scene.terrain.env_origins[env_ids]

    def _u(lo_hi):
        return torch.rand(num, device=dev) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]

    gap, drop = _u(gap_range), _u(drop_range)
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(num, 1)
    ca = torch.tensor(ps.platform_a_center(0.0), device=dev).repeat(num, 1)
    ca[:, 2] += drop
    env.scene[ps.PLATFORM_A].write_mocap_pose_to_sim(torch.cat([origins + ca, quat0], -1), env_ids=env_ids)
    cb = torch.tensor(ps.platform_b_center(0.0), device=dev).repeat(num, 1)
    cb[:, 0] += gap
    env.scene[ps.PLATFORM_B].write_mocap_pose_to_sim(torch.cat([origins + cb, quat0], -1), env_ids=env_ids)

    probs = torch.tensor([standing_prob, crouch_prob, flight_prob], device=dev)
    probs = probs / probs.sum()
    kind = torch.multinomial(probs.expand(num, 3), 1).squeeze(1)  # 0 stand, 1 crouch, 2 flight
    is_crouch, is_flight = kind == 1, kind == 2

    yaw = (torch.rand(num, device=dev) * 2 - 1) * yaw_noise          # facing +x, toward B
    pitch = (torch.rand(num, device=dev) * 2 - 1) * standing_tilt_max
    pitch = torch.where(is_flight, _u(flight_pitch_range), pitch)
    roll = (torch.rand(num, device=dev) * 2 - 1) * standing_tilt_max
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    quat = torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=1)

    a_top = ps.a_top(0.0) + drop
    x_stand = -(_u(edge_dist_range) + 0.02)            # trunk ~2 cm behind the feet' front
    z_stand = a_top + 0.118
    z_crouch = a_top + _u((0.085, 0.105))
    x_flight = gap + _u(flight_x_range)
    z_flight = ps.B_TOP_M + _u(flight_z_range)
    x = torch.where(is_flight, x_flight, x_stand)
    z = torch.where(is_flight, z_flight, torch.where(is_crouch, z_crouch, z_stand))
    env.sim.data.qpos[env_ids, 0] = origins[:, 0] + x
    env.sim.data.qpos[env_ids, 1] = origins[:, 1] + (torch.rand(num, device=dev) * 2 - 1) * 0.02
    env.sim.data.qpos[env_ids, 2] = origins[:, 2] + z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    if crouch_overrides:
        fold = torch.where(is_crouch, _u(crouch_factor_range), torch.zeros(num, device=dev))
        fold = torch.where(is_flight, _u(flight_tuck_range), fold)
        for jnt_idx, angle in crouch_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[env_ids, col]
            env.sim.data.qpos[env_ids, col] = home + fold * (angle - home)
    if joint_noise_std > 0.0:
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += torch.randn(num, len(cols), device=dev) * joint_noise_std

    vx = torch.where(is_flight, _u(flight_vx_range), torch.zeros(num, device=dev))
    vz = torch.where(is_flight, _u(flight_vz_range), torch.zeros(num, device=dev))
    env.sim.data.qvel[env_ids, 0] = vx * torch.cos(yaw)
    env.sim.data.qvel[env_ids, 1] = vx * torch.sin(yaw)
    env.sim.data.qvel[env_ids, 2] = vz
    env.sim.data.qvel[env_ids, 4] = (torch.rand(num, device=dev) * 2 - 1) * flight_omega_max * is_flight.float()

    # state machine reset (spawn frame = the block frame: forward = +x)
    st["spawn_xy"][env_ids] = env.sim.data.qpos[env_ids, :2].clone()
    st["spawn_yaw"][env_ids] = yaw
    st["gap"][env_ids] = gap
    st["drop"][env_ids] = drop
    st["stepped"][env_ids] = False
    st["step_landing"][env_ids] = False
    st["feet_ref_set"][env_ids] = False
    st["feet_ref_x"][env_ids] = 0.0
    st["ground_feet_x"][env_ids] = 0.0
    st["ground_trunk_x"][env_ids] = 0.0
    # flight spawns: the takeoff point is A's edge, behind the spawn
    behind = x_flight + 0.05
    st["takeoff_x"][env_ids] = torch.where(is_flight, -behind, torch.zeros(num, device=dev))
    st["takeoff_feet_x"][env_ids] = torch.where(is_flight, -behind, torch.zeros(num, device=dev))
    st["max_prog"][env_ids] = 0.0
    st["paid"][env_ids] = 0.0
    st["flight_steps"][env_ids] = torch.where(is_flight, torch.full((num,), MIN_FLIGHT_STEPS, device=dev, dtype=torch.long),
                                              torch.zeros(num, device=dev, dtype=torch.long))
    st["land_dist"][env_ids] = 0.0
    st["landed_step"][env_ids] = -1
    st["bonus_paid"][env_ids] = False
    st["failed"][env_ids] = False
    st["launch_paid"][env_ids] = 0.0
    st["takeoff_step"][env_ids] = -1
    st["takeoff_vz"][env_ids] = 0.0
    st["last_contact_l"][env_ids] = int(env.common_step_counter)
    st["last_contact_r"][env_ids] = int(env.common_step_counter)
    st["first_touch_step"][env_ids] = -1
    st["both_down_step"][env_ids] = -1
    st["sync"][env_ids] = 1.0
    st["phase"][env_ids] = torch.where(is_flight, torch.ones(num, device=dev, dtype=torch.long),
                                       torch.zeros(num, device=dev, dtype=torch.long))


# --------------------------------------------------------------------------
# diagnostics (weight 1: episode sum / max steps)
# --------------------------------------------------------------------------
def pj_landed_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    fresh = (st["landed_step"] == int(env.common_step_counter)) & ~st["stepped"]
    return torch.where(fresh, st["land_dist"], torch.zeros_like(st["land_dist"]))


def pj_landed_flag(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 on the touchdown step of a valid landing on B."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return ((st["landed_step"] == int(env.common_step_counter)) & ~st["stepped"]).float()


def pj_sync(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    fresh = (st["both_down_step"] == int(env.common_step_counter)) & ~st["stepped"]
    return torch.where(fresh, st["sync"], torch.zeros_like(st["sync"]))


def pj_stepped(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return st["stepped"].float()


def pj_pit(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 on a step with a foot on the floor (fell into the pit)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return st.get("foot_floor", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()


def pj_gap_obs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Gap (m) on the landed step of a valid landing (episode sum = the cleared gap)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    fresh = (st["landed_step"] == int(env.common_step_counter)) & ~st["stepped"]
    return torch.where(fresh, st["gap"], torch.zeros_like(st["gap"]))

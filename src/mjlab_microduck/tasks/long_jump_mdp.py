"""MDP terms for the Microduck standing long jump.

Episode structure (state machine kept on the env, one slot per environment):

    phase 0 GROUND   both feet on the floor, no step taken yet
    phase 1 FLIGHT   both feet off the floor after a two-foot takeoff
    phase 2 LANDED   feet back on the floor after a flight of >= 3 steps
    phase 3 INVALID  a non-foot body part touched the floor in flight, or a
                     second flight was started after landing (hop-run)

Everything is measured in the SPAWN frame: the forward axis is the trunk's
yaw at reset, so a jump is "forward" relative to where the robot started
facing.  A standing long jump has no run-up: a foot in the air while the
other stands (a step) or the feet creeping forward on the floor before the
takeoff latches ``stepped``; a stepped episode earns no jump reward.

Rewards
  * ``long_jump_progress``   potential-based: the increase of the trunk's
                             max forward displacement since takeoff, paid
                             only in FLIGHT (walking pays nothing).
  * ``long_jump_landing``    one-shot at touchdown: feet-midpoint distance
                             from the takeoff point (m), if the trunk is
                             upright (< 50 deg).
  * ``long_jump_stand``      standing composite (height x upright x pose)
                             after landing, scaled by a smoothstep of the
                             landed distance so a 1 cm hop earns nothing.
  * ``long_jump_stand_tax``  SELF-NEGATING height shortfall after landing
                             (crumpling on landing is net negative).
  * ``long_jump_body_contact`` SELF-NEGATING: any non-foot part on the floor.
  * ``long_jump_lateral``    SELF-NEGATING: lateral drift + heading change.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.mdp import (
    _DEFAULT_ASSET_CFG,
    _sensor_any_contact,
    _servo_joint_ids,
    standing_composite_score,
)

FEET_SENSOR = "feet_ground_contact"
BODY_SENSOR = "body_ground_contact"
HEAD_SENSOR = "head_ground_contact"
FOOT_BODIES = ("ankle_left", "ankle_right")

MIN_FLIGHT_STEPS = 3           # shorter airborne spells are contact chatter
STUCK_STEPS = 15               # 0.3 s upright on both feet after touchdown = a stuck landing
STUCK_TILT_DEG = 30.0
STEP_AIR_TIME_S = 0.2          # one foot in the air this long before takeoff = a step
FEET_CREEP_M = 0.08            # feet sliding forward on the floor before takeoff = a step
# v11 graded two-foot rule: the takeoff/landing gap between the feet (control
# steps, 20 ms each) scales every jump reward through a ``sync`` factor -
# full pay up to the FREE gap, linearly down to zero at the ZERO gap, and a
# gap at or beyond ZERO is a step (no jump rewards, flight progress clawed
# back).  v10's hard 2-step / 3-step cutoffs (j15) had no gradient: the
# policy collapsed to standing within 700 iterations.  j13 measured 48 ms
# mean / 120 ms p90 takeoff gaps and 74 / 200 ms landing gaps (a leap).
TAKEOFF_SYNC_FREE_STEPS = 1
TAKEOFF_SYNC_ZERO_STEPS = 5    # >= 100 ms between the feet leaving = a step
LANDING_SYNC_FREE_STEPS = 1
LANDING_SYNC_ZERO_STEPS = 6    # >= 120 ms between the touchdowns = a step
                               # (v7: 3 cm flagged the standing shuffle in most episodes, after which
                               # every jump reward was dead for the rest of the episode)


def _lj_state(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_lj", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        st = dict(
            phase=torch.zeros(n, dtype=torch.long, device=dev),
            stepped=torch.zeros(n, dtype=torch.bool, device=dev),
            spawn_xy=torch.zeros(n, 2, device=dev),
            spawn_yaw=z(),
            feet_ref_set=torch.zeros(n, dtype=torch.bool, device=dev),
            feet_ref_x=z(),
            ground_feet_x=z(),
            ground_trunk_x=z(),
            takeoff_x=z(),
            takeoff_feet_x=z(),
            max_prog=z(),
            paid=z(),
            flight_steps=torch.zeros(n, dtype=torch.long, device=dev),
            land_dist=z(),
            landed_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            bonus_paid=torch.zeros(n, dtype=torch.bool, device=dev),
            failed=torch.zeros(n, dtype=torch.bool, device=dev),
            launch_paid=z(),
            takeoff_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            takeoff_vz=z(),
            last_contact_l=torch.zeros(n, dtype=torch.long, device=dev),
            last_contact_r=torch.zeros(n, dtype=torch.long, device=dev),
            first_touch_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            both_down_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            sync=torch.ones(n, device=dev),
            step_landing=torch.zeros(n, dtype=torch.bool, device=dev),
            last_step=-1,
            foot_ids=None,
        )
        env._lj = st
    return st


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _forward_lateral(st: dict, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = xy - st["spawn_xy"]
    c, s = torch.cos(st["spawn_yaw"]), torch.sin(st["spawn_yaw"])
    return d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c


def _feet_contact(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensor = env.scene.sensors[FEET_SENSOR]
    found = sensor.data.found
    per_foot = found.view(found.shape[0], 2, -1).any(dim=-1)
    air = sensor.data.current_air_time
    if air is None:
        air = torch.zeros_like(per_foot, dtype=torch.float32)
    return per_foot[:, 0], per_foot[:, 1], air


def _foot_ids(env: ManagerBasedRlEnv, asset: Entity, st: dict) -> torch.Tensor:
    if st["foot_ids"] is None:
        ids, _ = asset.find_bodies(list(FOOT_BODIES), preserve_order=True)
        st["foot_ids"] = torch.tensor(ids, device=env.device, dtype=torch.long)
    return st["foot_ids"]


def _lj_update(env: ManagerBasedRlEnv, asset: Entity) -> dict:
    """Advance the per-env jump state machine once per control step."""
    st = _lj_state(env)
    step = int(env.common_step_counter)
    if step == st["last_step"]:
        return st
    st["last_step"] = step

    root_xy = torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0)
    fwd, _ = _forward_lateral(st, root_xy)
    feet_xy = torch.nan_to_num(asset.data.body_link_pos_w[:, _foot_ids(env, asset, st), :2], nan=0.0).mean(dim=1)
    feet_fwd, _ = _forward_lateral(st, feet_xy)
    lc, rc, air = _feet_contact(env)
    both = lc & rc
    none = ~lc & ~rc
    any_foot = lc | rc
    body_touch = _sensor_any_contact(env, BODY_SENSOR)
    if body_touch is None:
        body_touch = torch.zeros_like(both)

    phase = st["phase"]
    p0, p1, p2 = phase == 0, phase == 1, phase == 2

    # ---- GROUND: track the last two-foot stance, detect steps -------------
    on_ground = p0 & both
    st["ground_feet_x"] = torch.where(on_ground, feet_fwd, st["ground_feet_x"])
    st["ground_trunk_x"] = torch.where(on_ground, fwd, st["ground_trunk_x"])
    first_ref = on_ground & ~st["feet_ref_set"]
    st["feet_ref_x"] = torch.where(first_ref, feet_fwd, st["feet_ref_x"])
    st["feet_ref_set"] = st["feet_ref_set"] | first_ref
    one_foot_step = p0 & (lc ^ rc) & (air.max(dim=-1).values > STEP_AIR_TIME_S)
    creep = on_ground & st["feet_ref_set"] & ((feet_fwd - st["feet_ref_x"]) > FEET_CREEP_M)
    st["stepped"] = st["stepped"] | one_foot_step | creep

    # per-foot last contact step (for the two-foot takeoff rule)
    st["last_contact_l"] = torch.where(lc, torch.full_like(st["last_contact_l"], step), st["last_contact_l"])
    st["last_contact_r"] = torch.where(rc, torch.full_like(st["last_contact_r"], step), st["last_contact_r"])

    # ---- takeoff ------------------------------------------------------------
    takeoff = p0 & none
    # v11: a standing long jump leaves with BOTH feet at once - graded
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

    # ---- FLIGHT: progress, touchdown, invalid landings ----------------------
    in_flight = p1 & ~takeoff
    st["flight_steps"] = torch.where(in_flight, st["flight_steps"] + 1, st["flight_steps"])
    st["max_prog"] = torch.where(in_flight, torch.maximum(st["max_prog"], fwd - st["takeoff_x"]), st["max_prog"])
    bad_touch = in_flight & body_touch & ~any_foot
    touchdown = in_flight & any_foot & ~bad_touch
    real_landing = touchdown & (st["flight_steps"] >= MIN_FLIGHT_STEPS)
    st["first_touch_step"] = torch.where(real_landing, torch.full_like(st["first_touch_step"], step), st["first_touch_step"])
    chatter = touchdown & ~real_landing
    st["land_dist"] = torch.where(real_landing, feet_fwd - st["takeoff_feet_x"], st["land_dist"])
    st["landed_step"] = torch.where(real_landing, torch.full_like(st["landed_step"], step), st["landed_step"])
    phase = torch.where(real_landing, torch.full_like(phase, 2), phase)
    phase = torch.where(chatter, torch.zeros_like(phase), phase)
    phase = torch.where(bad_touch, torch.full_like(phase, 3), phase)

    # ---- LANDED: the second foot's delay after the first touchdown grades the
    # landing (v11); at LANDING_SYNC_ZERO_STEPS it is a step-landing.
    since_touch = step - st["first_touch_step"]
    both_down = p2 & both & (st["both_down_step"] < 0) & (st["first_touch_step"] >= 0)
    st["both_down_step"] = torch.where(both_down, torch.full_like(st["both_down_step"], step), st["both_down_step"])
    st["sync"] = torch.where(both_down, st["sync"] * _sync_factor(since_touch, LANDING_SYNC_FREE_STEPS, LANDING_SYNC_ZERO_STEPS), st["sync"])
    late_second_foot = p2 & ~both & (st["both_down_step"] < 0) & (since_touch >= LANDING_SYNC_ZERO_STEPS) & (st["first_touch_step"] >= 0)
    st["step_landing"] = late_second_foot & ~st["stepped"]
    st["stepped"] = st["stepped"] | late_second_foot
    # a second flight (hop-run) invalidates the episode
    rehop = p2 & none
    phase = torch.where(rehop, torch.full_like(phase, 3), phase)

    # ---- failure: fall termination (terminations run before rewards) -------
    fell = torch.zeros_like(both)
    try:
        fell = env.termination_manager.get_term("fell")
    except Exception:  # noqa: BLE001
        pass
    st["failed"] = fell | (phase == 3)
    st["phase"] = phase
    return st


def _landed_valid(st: dict) -> torch.Tensor:
    return (st["phase"] == 2) & ~st["stepped"]


def _sync_factor(gap_steps: torch.Tensor, free: int, zero: int) -> torch.Tensor:
    """1 for a gap <= free steps, linear to 0 at zero steps (v11)."""
    return torch.clamp(1.0 - (gap_steps.float() - free) / float(zero - free), 0.0, 1.0)


def _distance_gate(d: torch.Tensor, d_zero: float, d_full: float) -> torch.Tensor:
    t = torch.clamp((d - d_zero) / max(d_full - d_zero, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
def long_jump_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Paid increments of the trunk's max forward displacement while in flight."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    inc = torch.clamp(st["max_prog"] - st["paid"], min=0.0)
    inc = torch.where((st["phase"] == 1) & ~st["stepped"], inc, torch.zeros_like(inc))
    st["paid"] = st["paid"] + inc
    # v11: the takeoff sync factor scales the flight pay (the landing factor
    # is unknown until the second foot is down; it scales the bonus/stand).
    inc = inc * st["sync"]
    # v3 clawback: a flight that ends in a fall or a non-foot landing pays
    # back everything it collected (a head-first dive is net zero, not +);
    # v11: so does a step-landing (second foot >= 120 ms late).
    claw = st["failed"] | st["step_landing"]
    clawback = torch.where(claw, st["paid"] * st["sync"], torch.zeros_like(inc))
    st["paid"] = torch.where(claw, torch.zeros_like(inc), st["paid"])
    return inc - clawback


def long_jump_landing_bonus(
    env: ManagerBasedRlEnv,
    max_tilt_deg: float = STUCK_TILT_DEG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-shot: landed distance (m, >= 0) once the landing is STUCK.

    Paid ``STUCK_STEPS`` control steps after touchdown if the episode is still
    in LANDED (no re-hop, no fall), both feet are on the floor and the trunk
    tilt is below ``max_tilt_deg``.  A touchdown followed by a fall pays 0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    lc, rc, _ = _feet_contact(env)
    due = (st["landed_step"] >= 0) & (st["landed_step"] + STUCK_STEPS == int(env.common_step_counter))
    stuck = due & _landed_valid(st) & ~st["bonus_paid"] & lc & rc & (tilt < math.radians(max_tilt_deg)) & ~st["failed"]
    st["bonus_paid"] = st["bonus_paid"] | due
    return torch.where(stuck, torch.clamp(st["land_dist"], min=0.0) * st["sync"], torch.zeros_like(st["land_dist"]))


def long_jump_stand(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    dist_zero: float = 0.03,
    dist_full: float = 0.12,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Standing composite after a valid landing, scaled by the landed distance."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
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
    gate = _landed_valid(st).float() * _distance_gate(st["land_dist"], dist_zero, dist_full) * st["sync"]
    return score * gate


def long_jump_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING: -(height shortfall) after landing (POSITIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * (st["phase"] >= 2).float()


def long_jump_body_contact_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step while a non-foot part touches the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    _lj_update(env, asset)
    touch = _sensor_any_contact(env, BODY_SENSOR)
    if touch is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -touch.float()


def long_jump_launch(env: ManagerBasedRlEnv, vz_cap: float = 0.6, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Potential-based launch shaping: paid increments of the max upward CoM
    velocity reached while still on the ground (capped).  v6: standing still
    had no gradient toward pushing off (v5 paid the waiting tax instead of
    hopping); a push-off now pays as it happens, bounded by vz_cap per episode."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    vz = torch.clamp(torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0), min=0.0, max=vz_cap)
    inc = torch.clamp(vz - st["launch_paid"], min=0.0) * (st["phase"] == 0).float()
    st["launch_paid"] = st["launch_paid"] + inc
    return inc


def long_jump_takeoff_bonus(env: ManagerBasedRlEnv, max_tilt_deg: float = 45.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """One-shot at a two-foot takeoff: upward CoM velocity (m/s) if the trunk is upright
    (the flip's takeoff bonus; v7 for the long jump, whose standing start never hopped)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    fresh = (st["takeoff_step"] == int(env.common_step_counter)) & ~st["stepped"] & (tilt < math.radians(max_tilt_deg))
    return torch.where(fresh, torch.clamp(st["takeoff_vz"], min=0.0) * st["sync"], torch.zeros_like(st["takeoff_vz"]))


def long_jump_dawdle_penalty(env: ManagerBasedRlEnv, after_s: float = 1.0, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step still on the ground (no takeoff) after ``after_s``.

    v4: v3 converged to standing still (upright pays, jumping is risky); the
    tax makes waiting cost more than a careful attempt."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    waited = env.episode_length_buf.float() * env.step_dt > after_s
    return -((st["phase"] == 0) & waited).float()


def long_jump_lateral_penalty(
    env: ManagerBasedRlEnv,
    yaw_weight: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING: -(|lateral offset| + yaw_weight * |heading change|)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    _, lat = _forward_lateral(st, torch.nan_to_num(asset.data.root_link_pos_w[:, :2], nan=0.0))
    yaw = _yaw_of(asset.data.root_link_quat_w)
    dyaw = torch.atan2(torch.sin(yaw - st["spawn_yaw"]), torch.cos(yaw - st["spawn_yaw"]))
    return -(lat.abs() + yaw_weight * dyaw.abs())


# --------------------------------------------------------------------------
# terminations
# --------------------------------------------------------------------------
def long_jump_fell(
    env: ManagerBasedRlEnv,
    max_tilt_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Trunk past ``max_tilt_deg`` or the head on the floor: the jump failed."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    fell = tilt > math.radians(max_tilt_deg)
    head = _sensor_any_contact(env, HEAD_SENSOR)
    if head is not None:
        fell = fell | head
    return fell


def long_jump_fall_cost(
    env: ManagerBasedRlEnv,
    max_tilt_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING one-shot: -1 on the step the fall termination fires (POSITIVE weight)."""
    return -long_jump_fell(env, max_tilt_deg=max_tilt_deg, asset_cfg=asset_cfg).float()


# --------------------------------------------------------------------------
# spawn / reset
# --------------------------------------------------------------------------
def reset_long_jump_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.55,
    crouch_prob: float = 0.15,
    flight_prob: float = 0.30,
    standing_z_range: tuple = (0.11, 0.12),
    standing_tilt_max: float = math.radians(3.0),
    crouch_overrides: Optional[dict] = None,
    crouch_factor_range: tuple = (0.3, 0.7),
    crouch_z_range: tuple = (0.085, 0.105),
    flight_z_range: tuple = (0.14, 0.20),
    flight_vx_range: tuple = (0.4, 1.2),
    flight_vz_range: tuple = (-0.2, 0.8),
    flight_pitch_range: tuple = (math.radians(-10.0), math.radians(15.0)),
    flight_tuck_range: tuple = (0.1, 0.6),
    flight_omega_max: float = 1.0,
    joint_noise_std: float = 0.05,
):
    """Standing / crouched / mid-flight spawns (reverse curriculum) + state reset.

    Runs after ``reset_robot_joints`` (HOME joints already written).  Flight
    spawns are born in phase FLIGHT with the takeoff point just behind them,
    so landing and standing up are practised from step 0 (the roulade
    mid-roll lesson: the last mile needs on-policy data early).
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    dev = env.device
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_state(env)

    probs = torch.tensor([standing_prob, crouch_prob, flight_prob], device=dev)
    probs = probs / probs.sum()
    kind = torch.multinomial(probs.expand(num, 3), 1).squeeze(1)  # 0 stand, 1 crouch, 2 flight
    is_crouch, is_flight = kind == 1, kind == 2

    yaw = torch.rand(num, device=dev) * 2 * np.pi - np.pi
    pitch = (torch.rand(num, device=dev) * 2 - 1) * standing_tilt_max
    fp = torch.rand(num, device=dev) * (flight_pitch_range[1] - flight_pitch_range[0]) + flight_pitch_range[0]
    pitch = torch.where(is_flight, fp, pitch)
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

    def _u(lo_hi):
        return torch.rand(num, device=dev) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]

    z = _u(standing_z_range)
    z = torch.where(is_crouch, _u(crouch_z_range), z)
    z = torch.where(is_flight, _u(flight_z_range), z)

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)

    # Leg fold: crouch spawns lerp HOME -> crouch, flight spawns tuck a little.
    fold = torch.where(is_crouch, _u(crouch_factor_range), torch.zeros(num, device=dev))
    fold = torch.where(is_flight, _u(flight_tuck_range), fold)
    if crouch_overrides:
        for jnt_idx, angle in crouch_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[env_ids, col]
            env.sim.data.qpos[env_ids, col] = home + fold * (angle - home)
    if joint_noise_std > 0.0:
        noise = torch.randn(num, len(cols), device=dev) * joint_noise_std
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # Flight velocity: forward along the spawn yaw, some vertical, a little spin.
    vx = torch.where(is_flight, _u(flight_vx_range), torch.zeros(num, device=dev))
    vz = torch.where(is_flight, _u(flight_vz_range), torch.zeros(num, device=dev))
    env.sim.data.qvel[env_ids, 0] = vx * torch.cos(yaw)
    env.sim.data.qvel[env_ids, 1] = vx * torch.sin(yaw)
    env.sim.data.qvel[env_ids, 2] = vz
    omega = (torch.rand(num, device=dev) * 2 - 1) * flight_omega_max * is_flight.float()
    env.sim.data.qvel[env_ids, 4] = omega

    # State machine reset.
    xy = env.sim.data.qpos[env_ids, :2].clone()
    st["spawn_xy"][env_ids] = xy
    st["spawn_yaw"][env_ids] = yaw
    st["stepped"][env_ids] = False
    st["feet_ref_set"][env_ids] = False
    st["feet_ref_x"][env_ids] = 0.0
    st["ground_feet_x"][env_ids] = 0.0
    st["ground_trunk_x"][env_ids] = 0.0
    behind = 0.03 + torch.rand(num, device=dev) * 0.05   # so a flight-spawn landing clears the 3 cm distance gate
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
    st["step_landing"][env_ids] = False
    st["phase"][env_ids] = torch.where(is_flight, torch.ones(num, device=dev, dtype=torch.long),
                                       torch.zeros(num, device=dev, dtype=torch.long))


# --------------------------------------------------------------------------
# diagnostics (logged through a zero-weight reward or read by probes)
# --------------------------------------------------------------------------
def long_jump_landed_distance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Landed distance (m) on the touchdown step, else 0 (episode sum = distance)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    fresh = st["landed_step"] == int(env.common_step_counter)
    return torch.where(fresh & ~st["stepped"], st["land_dist"], torch.zeros_like(st["land_dist"]))


def long_jump_stepped(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 per step while the episode is flagged as stepped (diagnostic)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    return st["stepped"].float()


def long_jump_sync(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Two-foot sync factor (0-1) on the step both feet are down after a landing, else 0
    (episode sum = the landing's sync factor; diagnostic)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    fresh = (st["both_down_step"] == int(env.common_step_counter)) & ~st["stepped"]
    return torch.where(fresh, st["sync"], torch.zeros_like(st["sync"]))


def long_jump_landed_flag(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 on the touchdown step of a valid (non-stepped) landing, else 0."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _lj_update(env, asset)
    fresh = (st["landed_step"] == int(env.common_step_counter)) & ~st["stepped"]
    return fresh.float()

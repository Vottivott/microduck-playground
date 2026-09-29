"""MDP terms for the platform flip (backflip / front flip onto a crash mat).

Per-env state machine (spawn frame = platform frame, edge at x = 0):

    phase 0 PLATFORM  on the platform (feet or anything touching it)
    phase 1 AIRBORNE  nothing touches platform, mat or floor
    phase 2 LANDED    feet touched the mat after being airborne
    phase 3 FAILED    a non-foot part hit the mat/floor first, or a fall

Rotation about the trunk's pitch axis is integrated ONLY while airborne
(the roulade integrates only while supported; a flip is the opposite).
``dir_sign`` = +1 front flip (nose down = positive body-frame pitch rate,
the roulade's convention), -1 backflip.

Rewards
  * ``flip_progress``      potential-based: paid increments of the max
                           airborne rotation, capped at 2 pi + 0.35 rad
  * ``flip_stand``         standing composite on the mat x completion gate
                           (rotation 300-420 deg) x landed on feet
  * ``flip_stand_tax``     SELF-NEGATING height shortfall after a landing
  * ``flip_fall_cost``     SELF-NEGATING one-shot on the fall termination
  * ``flip_body_contact``  SELF-NEGATING non-foot contact with mat/floor
  * ``flip_dawdle``        SELF-NEGATING per step on the platform after 1 s
"""
from __future__ import annotations

import math
import os as _os
from typing import Optional

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import flip_stage
from mjlab_microduck.tasks.mdp import (
    _DEFAULT_ASSET_CFG,
    _sensor_any_contact,
    _servo_joint_ids,
    standing_composite_score,
)

FEET_MAT = "feet_mat_contact"
FEET_PLATFORM = "feet_platform_contact"
BODY_MAT = "body_mat_contact"
BODY_PLATFORM = "body_platform_contact"
BODY_FLOOR = "body_floor_contact"
HEAD_ANY = ("head_mat_contact", "head_platform_contact", "head_floor_contact")

import os

ROT_CAP = 2.0 * math.pi + 0.35
GATE_LO = math.radians(300.0)
GATE_HI = math.radians(420.0)
DAWDLE_AFTER_S = 1.0
# v3 (user 2026-09-17: "jump off the edge into the flip, not fall backwards"):
# the takeoff must be a spring, not a topple - trunk still near upright and
# the CoM moving UP when the feet leave the platform, no body part having
# leaned on the platform just before.  Rotation / landing pay only then.
JUMP_TILT_MAX = math.radians(float(os.getenv("MICRODUCK_FLIP_TAKEOFF_TILT_DEG", "35")))
JUMP_VZ_MIN = float(os.getenv("MICRODUCK_FLIP_TAKEOFF_VZ_MIN", "0.15"))
JUMP_LEAN_WINDOW_STEPS = 10
KNEE_JOINTS = (3, 12)
HIP_JOINTS = (2, 11)


def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_flip", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        st = dict(
            phase=torch.zeros(n, dtype=torch.long, device=dev),
            accum=z(), max_accum=z(), paid=z(),
            height=torch.full((n,), 0.7, device=dev),
            dir_sign=torch.ones(n, device=dev),
            spawn_step=torch.zeros(n, dtype=torch.long, device=dev),
            landed_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            failed=torch.zeros(n, dtype=torch.bool, device=dev),
            jump_ok=torch.zeros(n, dtype=torch.bool, device=dev),
            takeoff_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            takeoff_vz=z(),
            takeoff_tilt=z(),
            hops=torch.zeros(n, dtype=torch.long, device=dev),
            hop_step=torch.full((n,), -1, dtype=torch.long, device=dev),
            last_lean_step=torch.full((n,), -1000, dtype=torch.long, device=dev),
            launch_paid=z(),
            last_step=-1,
        )
        env._flip = st
    return st


def _any(env, name):
    v = _sensor_any_contact(env, name)
    return v if v is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _update(env: ManagerBasedRlEnv, asset: Entity) -> dict:
    st = _st(env)
    step = int(env.common_step_counter)
    if step == st["last_step"]:
        return st
    st["last_step"] = step
    feet_mat = _any(env, FEET_MAT)
    feet_plat = _any(env, FEET_PLATFORM)
    body_touch = _any(env, BODY_MAT) | _any(env, BODY_PLATFORM) | _any(env, BODY_FLOOR)
    head_touch = torch.zeros_like(body_touch)
    for name in HEAD_ANY:
        head_touch = head_touch | _any(env, name)
    any_touch = feet_mat | feet_plat | body_touch
    phase = st["phase"]
    p0, p1, p2 = phase == 0, phase == 1, phase == 2

    # rotation integrates only while airborne (nothing touching)
    omega = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0) * st["dir_sign"]
    airborne_now = ~any_touch
    delta = torch.where(airborne_now & (p1 | p0), omega * env.step_dt, torch.zeros_like(omega))
    st["accum"] = st["accum"] + delta
    st["max_accum"] = torch.maximum(st["max_accum"], st["accum"])

    body_plat = _any(env, BODY_PLATFORM)
    st["last_lean_step"] = torch.where(body_plat & p0, torch.full_like(st["last_lean_step"], step), st["last_lean_step"])
    takeoff = p0 & airborne_now
    g0 = asset.data.projected_gravity_b
    tilt0 = torch.acos(torch.clamp(-torch.nan_to_num(g0[:, 2], nan=-1.0), -1.0, 1.0))
    vz0 = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    recent_lean = (step - st["last_lean_step"]) <= JUMP_LEAN_WINDOW_STEPS
    ok = takeoff & (tilt0 < JUMP_TILT_MAX) & (vz0 > JUMP_VZ_MIN) & ~recent_lean
    st["jump_ok"] = torch.where(takeoff, ok, st["jump_ok"])
    st["takeoff_step"] = torch.where(takeoff, torch.full_like(st["takeoff_step"], step), st["takeoff_step"])
    st["takeoff_vz"] = torch.where(takeoff, vz0, st["takeoff_vz"])
    st["takeoff_tilt"] = torch.where(takeoff, tilt0, st["takeoff_tilt"])
    phase = torch.where(takeoff, torch.ones_like(phase), phase)
    # v8 (user 2026-09-18, watching the clip: "it first does a small jump and
    # then does the actual jump"): a PREPARATORY HOP must not steal the takeoff
    # latch.  Before this, the first instant airborne latched jump_ok, the
    # takeoff vz and the tilt - so the gate and every takeoff metric described
    # the hop, and the real jump was never measured or rewarded.  Landing back
    # on the PLATFORM now returns the episode to GROUND, so the takeoff belongs
    # to the departure that actually leads to the flight, and the hop is
    # counted and priced.
    back_on_platform = p1 & feet_plat & ~airborne_now & ~_any(env, BODY_MAT) & ~_any(env, BODY_FLOOR)
    if bool(back_on_platform.any()):
        z = torch.zeros_like(st["accum"])
        st["accum"] = torch.where(back_on_platform, z, st["accum"])
        st["max_accum"] = torch.where(back_on_platform, z, st["max_accum"])
        st["paid"] = torch.where(back_on_platform, z, st["paid"])
        st["launch_paid"] = torch.where(back_on_platform, z, st["launch_paid"])
    st["hops"] = st["hops"] + back_on_platform.long()
    st["hop_step"] = torch.where(back_on_platform, torch.full_like(st["hop_step"], step), st["hop_step"])
    phase = torch.where(back_on_platform, torch.zeros_like(phase), phase)

    # v6: a crash is a non-foot part on the MAT or FLOOR while airborne; brushing
    # the platform edge on the way out (30 % of platform starts ended as
    # "fell" at 0.57 s with 47 deg of rotation, still at platform height) is
    # priced, not fatal.
    body_mat_floor = _any(env, BODY_MAT) | _any(env, BODY_FLOOR)
    landed = p1 & feet_mat & ~body_mat_floor
    crash = p1 & body_mat_floor
    phase = torch.where(landed, torch.full_like(phase, 2), phase)
    phase = torch.where(crash, torch.full_like(phase, 3), phase)
    st["landed_step"] = torch.where(landed, torch.full_like(st["landed_step"], step), st["landed_step"])

    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    fell = head_touch | (p2 & body_touch & (tilt > math.radians(60.0))) | crash
    st["failed"] = st["failed"] | fell
    st["phase"] = phase
    return st


def _completion_gate(st: dict) -> torch.Tensor:
    t = torch.clamp((st["max_accum"] - GATE_LO) / (GATE_HI - GATE_LO), 0.0, 1.0)
    # full credit inside [GATE_LO, GATE_HI], fading out past it (over-rotation)
    over = torch.clamp((st["max_accum"] - GATE_HI) / math.radians(60.0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t) * (1.0 - over)


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
HOP_DECAY = float(_os.getenv("MICRODUCK_FLIP_HOP_DECAY", "1.0"))   # 1.0 = off
HOP_MIN_VZ = float(_os.getenv("MICRODUCK_FLIP_HOP_MIN_VZ", "0.05"))   # a hop PUSHES off; a spawn settle drops


def _hop_discount(st: dict) -> torch.Tensor:
    """Scale the flip payoff by ``HOP_DECAY ** hops``.

    Measured 2026-09-18 on f22 model_13250 (genuine platform starts only):
    every one of 256 flips hops at least once, 89 % hop exactly three times, and
    hopping more buys nothing - departure vz is 0.174 at two hops and 0.180 at
    three, clean-jump rate 79 % and 76 %.  So the hops are a fixed ritual, not a
    countermovement storing energy.

    Raising the flat hop tax from 3 to 8 over 1250 iterations changed the hop
    count not at all, because a tax only works if a cheaper alternative is
    already in the repertoire, and here none is: the policy has never sampled a
    hop-free flip.  A discount on the PAYOFF is different - it makes every hop
    removed multiply what the flip earns, which is a gradient rather than a
    cliff.  A hard "hops must be zero" gate would zero today's entire payoff and
    collapse the policy the way the long jump's hard two-foot cutoff did.
    """
    return torch.pow(torch.tensor(HOP_DECAY, device=st["hops"].device), st["hops"].float())


def flip_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    capped = torch.clamp(st["max_accum"], max=ROT_CAP)
    inc = torch.clamp(capped - st["paid"], min=0.0)
    st["paid"] = st["paid"] + inc
    return inc * st["jump_ok"].float() * _hop_discount(st)


def flip_stand(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: Optional[dict] = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Standing composite on the mat, gated on a completed flip and a feet landing."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    score = standing_composite_score(
        env, target_height=target_height + flip_stage.MAT_THICKNESS_M, height_std=height_std,
        upright_std=upright_std, pose_std=pose_std, joint_indices=joint_indices,
        target_overrides=target_overrides, asset_cfg=asset_cfg,
    )
    return score * _completion_gate(st) * (st["phase"] == 2).float() * st["jump_ok"].float() * _hop_discount(st)


def flip_stand_tax(env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -(height shortfall over the mat) once landed (POSITIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    shortfall = torch.clamp(target_height + flip_stage.MAT_THICKNESS_M - z, min=0.0)
    return -shortfall * (st["phase"] >= 2).float()


def flip_fall_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot on the step the fall termination fires."""
    return -flip_fell(env, asset_cfg=asset_cfg).float()


def flip_body_contact_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    touch = _any(env, BODY_MAT) | _any(env, BODY_FLOOR) | _any(env, BODY_PLATFORM)
    return -touch.float()


def flip_dawdle_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step still on the platform after DAWDLE_AFTER_S."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    waited = (int(env.common_step_counter) - st["spawn_step"]) * env.step_dt > DAWDLE_AFTER_S
    return -((st["phase"] == 0) & waited).float()


def flip_launch(env: ManagerBasedRlEnv, vz_cap: float = 0.6, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Paid increments of the max upward CoM velocity while still on the platform
    (capped): the gradient toward a push-off that standing on the platform lacks."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    vz = torch.clamp(torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0), min=0.0, max=vz_cap)
    inc = torch.clamp(vz - st["launch_paid"], min=0.0) * (st["phase"] == 0).float()
    st["launch_paid"] = st["launch_paid"] + inc
    return inc


def flip_takeoff_bonus(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Upward CoM velocity at the jump-like takeoff, paid ON LANDING on the mat.

    Paid AT the takeoff it was a farm: every hop back onto the platform
    re-latched the takeoff and collected again, which is the preparatory hop the
    user spotted in the clip (and which the parkour chain turned into hopping on
    the spot - 100 % of second jumps landed back on the same ledge).  Paid on
    the landing, a hop that goes nowhere is worth nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    landed_now = (st["landed_step"] == int(env.common_step_counter)) & st["jump_ok"]
    return torch.where(landed_now, torch.clamp(st["takeoff_vz"], min=0.0), torch.zeros_like(st["takeoff_vz"])) * _hop_discount(st)


def flip_tuck_reward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Knee + hip flexion (0..1) while airborne after a jump-like takeoff."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    ids = _servo_joint_ids(env, asset)
    q = asset.data.joint_pos[:, [ids[j] for j in KNEE_JOINTS + HIP_JOINTS]].abs()
    tuck = torch.clamp(q / 1.2, 0.0, 1.0).mean(dim=-1)
    return tuck * ((st["phase"] == 1) & st["jump_ok"]).float()


def flip_takeoff_flag(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Diagnostic: 1 on a jump-like takeoff step (x100 = fraction of episodes)."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return ((st["takeoff_step"] == int(env.common_step_counter)) & st["jump_ok"]).float()


def flip_lateral_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -|lateral (y) offset from the platform centre line|."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    y = torch.nan_to_num(asset.data.root_link_pos_w[:, 1] - env.scene.terrain.env_origins[:, 1], nan=0.0)
    return -y.abs()


# diagnostics
def flip_max_rotation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Airborne rotation reached (rad) reported once, on the landing or failure step."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    if not hasattr(env, "_flip_reported"):
        env._flip_reported = torch.zeros_like(st["failed"])
    done_now = ((st["landed_step"] == int(env.common_step_counter)) | st["failed"]) & ~env._flip_reported
    env._flip_reported = env._flip_reported | done_now
    return torch.where(done_now, st["max_accum"], torch.zeros_like(st["max_accum"]))


def flip_landed_flag(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return (st["landed_step"] == int(env.common_step_counter)).float()


# --------------------------------------------------------------------------
# observations / terminations
# --------------------------------------------------------------------------
def flip_command_obs(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Body-command slot: [drop height (m, centred at 0.7, /0.2), direction, 0...]."""
    st = _st(env)
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, 0] = (st["height"] - 0.7) / 0.2
    out[:, 1] = st["dir_sign"]
    return out


def flip_hop_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot: -1 each time it comes back down on the platform
    after leaving it (the preparatory hop).  A flip should be ONE departure."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return -(st["hop_step"] == int(env.common_step_counter)).float()


def flip_hops(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Preparatory hops so far this episode (diagnostic)."""
    asset: Entity = env.scene[asset_cfg.name]
    return _update(env, asset)["hops"].float()


def flip_hopped(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Landing back on the platform ENDS the episode.

    Two graded pressures failed to reduce the preparatory hop.  The flat tax
    went 3 -> 8 (1250 iterations, hop count unmoved at ~1.9) and then the payoff
    was discounted to 0.6 ** hops, leaving 22 % of the reward at the usual three
    hops (1000 iterations, hop count rose from 2.00 to 2.12).  Neither moved it,
    which says the policy has no hop-free flip to fall back on: it would have to
    cross a valley - fewer hops means a worse flip before it means a better one
    - and a gradient cannot pay for that crossing.

    So the hop is made impossible rather than expensive.  Mid-flip spawns (a
    quarter of training) never hop, so reward keeps flowing while the platform
    start relearns its departure from nothing.
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    step = int(env.common_step_counter)
    hopped_now = (st["hop_step"] == step) & (st["hops"] > 0)
    # Telling a spawn settle from a preparatory hop.  Run f24 tried a minimum
    # airborne time and killed itself - 308 of ~390 episodes per window ended
    # "hopped" at 9.5 steps - because the robot settling onto the platform reads
    # as a departure and a return.  Run f25 then set that minimum at 6 steps,
    # which never fired, because a traced hop is only ~40 ms clear of the
    # platform.  A time window cannot separate them, and a grace period after
    # the spawn cannot either: the real hop happens at ~0.16 s, inside it.
    #
    # The trace shows what does separate them.  A hop is a PUSH: trunk velocity
    # +0.20 m/s, the body rising ~1.6 cm above its resting height and coming
    # back down over ~0.24 s.  A settle is a DROP, with velocity at or below
    # zero.  So gate on the departure velocity, not on time.
    return hopped_now & (st["takeoff_vz"] > HOP_MIN_VZ)


def flip_fell(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Head on any surface, a crash landing, or lying on the mat after landing."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return st["failed"]


# --------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------
def reset_flip_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    direction: str = "back",
    height_range: tuple = (0.6, 0.8),
    edge_dist_range: tuple = (0.03, 0.08),
    yaw_noise: float = math.radians(8.0),
    midflip_prob: float = 0.4,
    midflip_angle_range: tuple = (math.radians(350.0), math.radians(360.0)),
    midflip_omega_range: tuple = (0.0, 2.0),
    midflip_u_range: tuple = (0.05, 0.12),
    midflip_x_range: tuple = (0.15, 0.45),
    midflip_vx_range: tuple = (0.1, 0.4),
    midflip_vz_scale: float = 0.15,
    tuck_overrides: Optional[dict] = None,
    tuck_factor_range: tuple = (0.0, 0.2),
    joint_noise_std: float = 0.05,
):
    """Place platform + mat, spawn the robot on the platform edge (or mid-flip
    above the mat), and reset the flip state.  Runs after reset_robot_joints."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    dev = env.device
    asset: Entity = env.scene[asset_cfg.name]
    st = _st(env)
    origins = env.scene.terrain.env_origins[env_ids]
    sign = 1.0 if direction == "front" else -1.0

    def _u(lo_hi):
        return torch.rand(num, device=dev) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]

    height = _u(height_range)
    # stage
    pc = torch.tensor(flip_stage.platform_center(0.0), device=dev).repeat(num, 1)
    pc[:, 2] += height
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(num, 1)
    env.scene[flip_stage.PLATFORM_ENTITY].write_mocap_pose_to_sim(torch.cat([origins + pc, quat0], -1), env_ids=env_ids)
    mc = torch.tensor(flip_stage.mat_center(), device=dev).repeat(num, 1)
    env.scene[flip_stage.MAT_ENTITY].write_mocap_pose_to_sim(torch.cat([origins + mc, quat0], -1), env_ids=env_ids)

    is_mid = torch.rand(num, device=dev) < midflip_prob
    # facing: front flip faces +x (toward the edge), backflip faces -x (edge behind)
    face_yaw = 0.0 if sign > 0 else math.pi
    yaw = face_yaw + (torch.rand(num, device=dev) * 2 - 1) * yaw_noise
    # standing spawn: feet edge distance -> trunk x (trunk over the ankles)
    edge = _u(edge_dist_range)
    x_stand = -(edge + 0.02)
    z_stand = flip_stage.MAT_THICKNESS_M + height + 0.118
    # mid-flip spawn
    u = _u(midflip_u_range)
    x_mid = _u(midflip_x_range)
    z_mid = flip_stage.MAT_THICKNESS_M + 0.12 + u * height
    ang = _u(midflip_angle_range)
    pitch = torch.where(is_mid, sign * ang, torch.zeros(num, device=dev))
    x = torch.where(is_mid, x_mid, x_stand)
    z = torch.where(is_mid, z_mid, z_stand)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    quat = torch.stack([cp * cy, -sp * sy, sp * cy, cp * sy], dim=1)   # yaw then pitch about body y
    env.sim.data.qpos[env_ids, 0] = origins[:, 0] + x
    env.sim.data.qpos[env_ids, 1] = origins[:, 1] + (torch.rand(num, device=dev) * 2 - 1) * 0.02
    env.sim.data.qpos[env_ids, 2] = origins[:, 2] + z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    # mid-flip velocity: forward along +x, free-fall vz, spin about body y
    vx = torch.where(is_mid, _u(midflip_vx_range), torch.zeros(num, device=dev))
    # falling speed of a body that has dropped from the platform top to the
    # spawn height u*H above the mat (physically 2.7-4 m/s near the mat), scaled
    # by midflip_vz_scale so the landing curriculum can start slow
    vz = torch.where(is_mid, -torch.sqrt(torch.clamp(2 * 9.81 * (1 - u) * height, min=0.0)) * midflip_vz_scale, torch.zeros(num, device=dev))
    env.sim.data.qvel[env_ids, 0] = vx
    env.sim.data.qvel[env_ids, 2] = vz
    env.sim.data.qvel[env_ids, 4] = torch.where(is_mid, sign * _u(midflip_omega_range), torch.zeros(num, device=dev))

    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    if tuck_overrides:
        fold = torch.where(is_mid, _u(tuck_factor_range), torch.zeros(num, device=dev))
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[env_ids, col]
            env.sim.data.qpos[env_ids, col] = home + fold * (angle - home)
    if joint_noise_std > 0.0:
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += torch.randn(num, len(cols), device=dev) * joint_noise_std

    st["height"][env_ids] = height
    st["dir_sign"][env_ids] = sign
    st["accum"][env_ids] = torch.where(is_mid, ang, torch.zeros(num, device=dev))
    st["max_accum"][env_ids] = st["accum"][env_ids]
    st["paid"][env_ids] = st["accum"][env_ids]
    st["phase"][env_ids] = torch.where(is_mid, torch.ones(num, dtype=torch.long, device=dev), torch.zeros(num, dtype=torch.long, device=dev))
    st["spawn_step"][env_ids] = int(env.common_step_counter)
    st["landed_step"][env_ids] = -1
    st["failed"][env_ids] = False
    st["jump_ok"][env_ids] = is_mid
    st["takeoff_step"][env_ids] = -1
    st["takeoff_vz"][env_ids] = 0.0
    st["takeoff_tilt"][env_ids] = 0.0
    st["last_lean_step"][env_ids] = -1000
    st["hops"][env_ids] = 0
    st["hop_step"][env_ids] = -1
    st["launch_paid"][env_ids] = 0.0
    if hasattr(env, "_flip_reported"):
        env._flip_reported[env_ids] = False

"""MDP terms for the corridor APPROACH: walk in off the floor and stop standing
between the walls, in the pose the climb policy takes over from.

Why this is its own policy (user 2026-09-19).  The climb starts from a standing
handover pose and w12 now takes it 100 % of the time, so the missing piece is
getting there.  The cheap alternative was measured first and failed: walker_v1,
commanded straight at a 12.5 cm corridor from 32 cm out, gains about 10 cm and
NEVER ends standing between the walls (`experiments/chimney/MEASUREMENT.md`).

Reward shape follows the house rules:

  * approach is PAID AS PROGRESS (potential-based, the drop in distance to the
    handover point), so standing still pays zero and there is nothing to farm;
  * the arrival bonus is gated on the full handover state - inside the throat,
    upright at standing height, and slow - not on any single component, and the
    episode ENDS on success, so it cannot be collected twice;
  * jamming against a wall is charged, because a 13.5 cm body in a 12.5 cm slot
    can wedge and be squeezed upward rather than walk in;
  * falling ends the episode and is charged once.

The corridor's width and its direction relative to the robot go in the
otherwise zero-padded body-command slot: an operator measures the width once,
and the bearing is what any approach controller would have from odometry.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.tasks.mdp import _DEFAULT_ASSET_CFG, _servo_joint_ids

STAND_Z = 0.115             # measured standing trunk height
# the climb's own upright spawn angles, which is the pose it trains from
_STAND_TARGETS = {2: -0.4579, 3: -0.0049, 4: 0.4529, 11: 0.4579, 12: 0.0049, 13: -0.4529}
STAND_BAND = 0.025          # ... and how far off it still counts as standing
WIDTH_SCALE = 0.13          # command obs: (width - WIDTH_SCALE) / 0.02
INSIDE_Y_M = 0.5 * cs.WALL_DEPTH_M   # the throat runs +-15 cm along y
# ...but the CLIMB cannot start from just anywhere in it.  Measured 2026-09-20,
# placing a duck in the climb's own env and running the climb:
#
#   y = -13 cm :   0 % climb past 25 cm
#   y =  -9 cm :  73 %
#   y =  -5 cm :  92 %
#   y =   0    :  93 %
#   y = +5..13 :  92-96 %
#
# a9 stopped at y = -13.1 cm - it walked just far enough to be "inside the
# throat" by the old test and no further, which is precisely the one place the
# climb cannot start.  That single number is why the chain climbed nothing
# while every other quantity matched a spawn that climbs 98 %.
HANDOVER_Y_M = 0.06
SIDE = float(os.getenv("MICRODUCK_AP_SIDE", "-1"))  # -1 far side (a11), +1 camera side
SLOW_MPS = 0.06             # "stopped" for the handover
TILT_MAX_DEG = 35.0


def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_ap", None)
    if st is None:
        z = torch.zeros(env.num_envs, device=env.device)
        st = {
            "width": z.clone() + 0.125,
            "prev_dist": z.clone(),
            "arrived_steps": z.clone(),
            "best_dist": z.clone() + 99.0,
            "speed_ema": z.clone(),
        }
        env._ap = st
    return st


def _root(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Robot root position relative to its env origin, its quat, and its lin vel."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.root_link_pos_w[:, :3] - env.scene.env_origins
    return pos, asset.data.root_link_quat_w, asset.data.root_com_lin_vel_w


def _dist_to_goal(pos: torch.Tensor) -> torch.Tensor:
    """Planar distance to the handover point, the corridor centre at (0, 0)."""
    return torch.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + 1e-9)


def _upright(quat: torch.Tensor) -> torch.Tensor:
    """cos(tilt) of the trunk's up axis against world up."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return 1.0 - 2.0 * (x * x + y * y)


# The climb is handed a POSE, not just a position.  Measured 2026-09-20: with
# only a trunk-speed test, the approach "arrived" 99.6 % of the time while
# caught mid-stride - hips off the climb's stance by 19-23 degrees, a knee off
# by 21, and joints still swinging at 1.12 rad/s.  The climb trains from a
# static symmetric stance, so that state is outside its distribution and it
# gained 1.9 cm instead of climbing.  A trunk moving slowly is not a robot
# standing still.
STAND_JOINT_TOL = 0.20       # rad, per joint, against the climb's stance
STAND_JOINT_VEL = 1.0        # rad/s, mean absolute over the servos
# ...and FACING ACROSS THE SLOT.  Measured 2026-09-20: a nose-first approach
# hands over at |yaw| = 74 degrees, and the climb then gains nothing, because it
# braces back-against-one-wall and feet-against-the-other and so must face
# ACROSS the corridor (yaw 0), not along it.  This is the geometric reason the
# user asked for a side-on entry; the earlier probe that preferred forward entry
# measured how far the walker got IN, not whether the pose it ended in was
# usable.
STAND_YAW_TOL = math.radians(20.0)


def in_handover_pose(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
                     stand_overrides: Optional[dict] = None) -> torch.Tensor:
    """The full state the climb policy takes over from: standing upright at
    standing height, between the walls, inside the throat, stopped, AND in the
    climb's own stance with the legs at rest."""
    st = _st(env)
    asset: Entity = env.scene[asset_cfg.name]
    pos, quat, vel = _root(env, asset_cfg)
    inside = (pos[:, 0].abs() < 0.5 * st["width"]) & (pos[:, 1].abs() < HANDOVER_Y_M)
    tall = (pos[:, 2] - STAND_Z).abs() < STAND_BAND
    level = _upright(quat) > math.cos(math.radians(TILT_MAX_DEG))
    slow = torch.linalg.norm(vel[:, :2], dim=-1) < SLOW_MPS
    settled = asset.data.joint_vel.abs().mean(dim=-1) < STAND_JOINT_VEL
    w, qx, qy, qz = (quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])
    yaw = torch.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    across = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
    facing_across = across.abs() < STAND_YAW_TOL
    targets = stand_overrides if stand_overrides is not None else _STAND_TARGETS
    in_stance = torch.ones_like(slow)
    if targets:
        ids = _servo_joint_ids(env, asset)
        jp = asset.data.joint_pos
        for jnt_idx, angle in targets.items():
            in_stance &= (jp[:, ids[jnt_idx]] - angle).abs() < STAND_JOINT_TOL
    return inside & tall & level & slow & settled & in_stance & facing_across


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
# Speed limits, from a MEASUREMENT of what the duck can actually walk
# (`probe_walkspeed.py`, walker_v1 on flat ground):
#
#   command | sustained | instantaneous peak (mean / max)
#     0.20  |   0.187   |  0.50 / 0.92
#     0.60  |   0.296   |  0.69 / 0.89
#
# The lesson that cost three runs: a normal gait's INSTANTANEOUS centre-of-mass
# speed peaks at 0.5-0.9 m/s because the trunk lurches on every push-off, while
# the SUSTAINED speed tops out near 0.30.  Penalising instantaneous speed above
# 0.25 therefore taxes ordinary walking, and the "skating at three times
# walking speed" reading of a1 was comparing an instantaneous peak against a
# sustained figure.  a1 really averaged 38 cm/s against an achievable 30 - too
# fast, but by a quarter, not a factor of three.
#
# So: pay progress up to a little above the sustained maximum, and charge only
# a SMOOTHED speed, which lets the per-step oscillation cancel and prices the
# part that is actually escapable.
MAX_PAID_SPEED = 0.35        # m/s of closing speed that still earns progress
SUSTAINED_CAP = 0.32         # m/s of SMOOTHED speed above which the dash is charged
SPEED_TAU_S = 0.50           # smoothing window for that measurement


def approach_progress(env: ManagerBasedRlEnv, max_speed: float = MAX_PAID_SPEED,
                      asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Potential-based: the metres of distance to the handover point given up
    this step, CAPPED at ``max_speed``.  Holding still pays zero, backing off
    pays negative, and closing faster than a duck can walk pays no more than
    walking - so there is nothing to camp in and nothing to win by diving."""
    st = _st(env)
    pos, _, _ = _root(env, asset_cfg)
    d = _dist_to_goal(pos)
    gain = torch.clamp(st["prev_dist"] - d, max=max_speed * env.step_dt)
    st["prev_dist"] = d
    return gain


def approach_arrive(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Per-step bonus for standing in the handover pose.  The episode ends once
    it has been held (see ``approach_arrived``), so this is bounded."""
    return in_handover_pose(env, asset_cfg).float()


def approach_dash_penalty(env: ManagerBasedRlEnv, max_speed: float = SUSTAINED_CAP,
                          tau_s: float = SPEED_TAU_S,
                          asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Self-negating (POSITIVE weight): charged for SMOOTHED planar speed above
    what the robot can sustain.

    Smoothed, not instantaneous.  A walking gait's centre-of-mass speed peaks
    at 0.5-0.9 m/s on every push-off while sustaining under 0.30, so an
    instantaneous threshold charges ordinary walking and the policy cannot
    escape it by walking better.  A half-second EMA lets the per-step
    oscillation cancel and charges only a genuinely fast crossing."""
    st = _st(env)
    _, _, vel = _root(env, asset_cfg)
    sp = torch.linalg.norm(vel[:, :2], dim=-1)
    a = float(env.step_dt / max(tau_s, env.step_dt))
    ema = st.setdefault("speed_ema", torch.zeros_like(sp))
    if ema.shape != sp.shape:
        ema = torch.zeros_like(sp)
    st["speed_ema"] = ema + a * (sp - ema)
    return -torch.clamp(st["speed_ema"] - max_speed, min=0.0)


def approach_yaw_penalty(env: ManagerBasedRlEnv, tol: float = STAND_YAW_TOL,
                         asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Self-negating (POSITIVE weight): charged every step for not being square
    to the walls.

    Orientation has to be shaped CONTINUOUSLY, not only tested at the end.
    Progress pays for closing distance whatever way the duck faces, so turning
    and walking forwards out-earns side-stepping, and the terminal handover
    bonus - the only thing that wanted a square duck - was reached 0 % of the
    time from a real outside spawn (a8 ended at |yaw| = 94 degrees).  Charging
    the misalignment per step makes "turn and walk in nose-first" unprofitable
    on the way rather than only at the goal.
    """
    _, quat, _ = _root(env, asset_cfg)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    across = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
    return -torch.clamp(across.abs() - tol, min=0.0)


def approach_jam_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Self-negating (POSITIVE weight): charged for being squeezed off the floor.

    A 13.5 cm standing footprint in a 12.5 cm slot can wedge and get squirted
    upward instead of walking in - the entry probe measured trunk heights of
    30-44 cm doing exactly that.  Price height above standing."""
    pos, _, _ = _root(env, asset_cfg)
    return -torch.clamp(pos[:, 2] - (STAND_Z + STAND_BAND), min=0.0)


def approach_fall_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Self-negating (POSITIVE weight): charged once, on the step it falls.

    Covers EVERY policy-triggerable termination, not just `fell`: the approach
    inherits `fell_over` from the velocity base, and an unpriced termination is
    a free exit from the per-step penalties (a12 learned a 0.24 s squat-death
    through `fell`; pricing only that would move it to `fell_over`).
    Terminations are computed before rewards each step, so the manager's flags
    are current here."""
    fell = approach_fell(env, asset_cfg)
    tm = env.termination_manager
    for name in ("fell_over",):
        if name in tm.active_terms:
            fell = fell | tm.get_term(name)
    return -fell.float()


def gait_gate(env: ManagerBasedRlEnv, base_func, base_params: dict,
              far_m: float = 0.12, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Wrap a gait reward so it pays only while the duck is still WALKING IN.

    The velocity recipe gates its air-time reward on the magnitude of the twist
    COMMAND, which in this task is a leftover sampled from the walking recipe
    and has nothing to do with the approach - so the policy would be paid for
    stepping in proportion to a command it is not being asked to follow, and
    paid to march on the spot once it arrived.  Gate it on distance to the
    handover point instead: gait shaping during the walk, stillness at the goal.
    """
    pos, _, _ = _root(env, asset_cfg)
    far = (_dist_to_goal(pos) > far_m).float()
    return base_func(env, **base_params) * far


# --------------------------------------------------------------------------
# terminations
# --------------------------------------------------------------------------
def approach_fell(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    pos, quat, _ = _root(env, asset_cfg)
    return (pos[:, 2] < 0.065) | (_upright(quat) < math.cos(math.radians(70.0)))


def approach_arrived(env: ManagerBasedRlEnv, hold_s: float = 0.4,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Success: the handover pose held for ``hold_s``.  Holding is required so a
    policy cannot clip through the goal at speed and call it an arrival."""
    st = _st(env)
    now = in_handover_pose(env, asset_cfg)
    st["arrived_steps"] = torch.where(now, st["arrived_steps"] + 1.0, torch.zeros_like(st["arrived_steps"]))
    return st["arrived_steps"] >= (hold_s / env.step_dt)


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------
def approach_command_obs(env: ManagerBasedRlEnv, dim: int = 6,
                         asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Body-command slot: where the corridor is, in the robot's own frame, plus
    its width.  [dx, dy, sin(yaw_err), cos(yaw_err), (width - 0.13) / 0.02, 0]"""
    st = _st(env)
    pos, quat, _ = _root(env, asset_cfg)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    dx_w, dy_w = -pos[:, 0], -pos[:, 1]          # world vector to the goal
    c, s = torch.cos(yaw), torch.sin(yaw)
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, 0] = c * dx_w + s * dy_w              # forward component
    out[:, 1] = -s * dx_w + c * dy_w             # lateral component
    # the brace faces ACROSS the slot, so the heading to line up on is 0
    yaw_err = yaw
    out[:, 2] = torch.sin(yaw_err)
    out[:, 3] = torch.cos(yaw_err)
    out[:, 4] = (st["width"] - WIDTH_SCALE) / 0.02
    return out


# --------------------------------------------------------------------------
# spawn
# --------------------------------------------------------------------------
def reset_approach_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    width_range: tuple = (0.125, 0.125),
    distance_range: tuple = (0.25, 0.50),
    lateral_range: tuple = (-0.10, 0.10),
    yaw_noise: float = 0.6,
    stand_overrides: Optional[dict] = None,
    joint_noise_std: float = 0.04,
    frontier_prob: float = 0.35,
    nearly_done_prob: float = 0.15,
):
    """Walls at the sampled width; robot standing on the floor OUTSIDE the mouth.

    The robot is placed through the event manager, which is the only way a
    forced spawn actually reaches the simulation - writing qpos from a probe
    after ``env.reset()`` is silently ignored, and cost three wrong versions of
    the entry measurement before it was caught.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    dev = env.device
    asset: Entity = env.scene[asset_cfg.name]
    st = _st(env)
    origins = env.scene.env_origins[env_ids]

    def _u(lo_hi):
        return torch.rand(num, device=dev) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]

    width = _u(width_range)
    st["width"][env_ids] = width
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(num, 1)
    for side, name in ((-1.0, cs.WALL_LEFT), (1.0, cs.WALL_RIGHT)):
        pos = torch.zeros(num, 3, device=dev)
        pos[:, 0] = side * (0.5 * width + 0.5 * cs.WALL_THICKNESS_M)
        pos[:, 2] = cs.wall_z()
        env.scene[name].write_mocap_pose_to_sim(torch.cat([origins + pos, quat0], -1), env_ids=env_ids)

    # REVERSE CURRICULUM, for the same reason the exit needed one: once the
    # handover also required facing ACROSS the slot, a7 scored 0 % arrivals -
    # the bonus never fired, so there was no gradient toward the new behaviour
    # and the warm-started policy kept turning to walk in nose-first at
    # |yaw| = 101 degrees.  Some episodes now start already there.
    roll = torch.rand(num, device=dev)
    is_done = roll < nearly_done_prob
    is_frontier = (~is_done) & (roll < nearly_done_prob + frontier_prob)

    # Which side of the slot the duck walks in from.  -1 (default) is the far
    # side, which is how a11 was trained; +1 is the CAMERA side, so the duck
    # enters and later exits on the same side (user 2026-09-21).  Rotating the
    # duck 180 deg does NOT achieve this: it side-steps toward one fixed BODY
    # direction, and the exit's command is world-frame, so a rotated duck walks
    # off the back of the platform.  Entering from +y facing the same way means
    # side-stepping the other way, which is a different gait and has to be
    # trained.
    y0 = SIDE * (_u(distance_range) + INSIDE_Y_M)
    y0 = torch.where(is_frontier, SIDE * _u((0.04, 0.22)), y0)
    y0 = torch.where(is_done, _u((-0.05, 0.05)), y0)
    x0 = _u(lateral_range)
    x0 = torch.where(is_done, torch.zeros_like(x0), x0)
    # Facing ACROSS the slot, so the duck must side-step in and arrives in the
    # orientation the climb braces from.  It used to spawn facing along the
    # slot (pi/2) and walk in nose-first, which is unusable downstream.
    yaw = (torch.rand(num, device=dev) * 2 - 1) * yaw_noise
    # a nearly-done spawn is already square to the walls, so the handover test
    # passes at once and the critic gets something to back up
    yaw = torch.where(is_done, (torch.rand(num, device=dev) * 2 - 1) * 0.08, yaw)
    a = 0.5 * yaw
    env.sim.data.qpos[env_ids, 0] = origins[:, 0] + x0
    env.sim.data.qpos[env_ids, 1] = origins[:, 1] + y0
    env.sim.data.qpos[env_ids, 2] = origins[:, 2] + STAND_Z
    env.sim.data.qpos[env_ids, 3] = torch.cos(a)
    env.sim.data.qpos[env_ids, 4:6] = 0.0
    env.sim.data.qpos[env_ids, 6] = torch.sin(a)
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)
    if stand_overrides:
        for jnt_idx, angle in stand_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            env.sim.data.qpos[env_ids, col] = angle
    noise = torch.randn(num, len(servo_ids), device=dev) * joint_noise_std
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    env.sim.data.qpos[env_ids[:, None], cols[None, :]] += noise

    pos_now = env.sim.data.qpos[env_ids, :3] - origins
    st["prev_dist"][env_ids] = _dist_to_goal(pos_now)
    st["arrived_steps"][env_ids] = 0.0
    st["best_dist"][env_ids] = st["prev_dist"][env_ids]

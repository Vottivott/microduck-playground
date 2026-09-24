"""MDP terms for LEAVING the vertical corridor onto a platform at the top.

The user's route (sketch 2026-09-20): climb the slot with the climb policy
until just above platform level, switch policies, travel out of the slot along
+y while still braced between the overhanging walls, then descend and stand on
the platform.  So this policy starts WEDGED - the climb's own terminal state -
and its job is the corner: vertical brace to horizontal traverse to a landing.

Reward shape follows the house rules, and in particular the one this project
learned the expensive way: progress is potential-based AND rate-limited, so
neither camping nor diving can farm it (the corridor-approach policy hit 99 %
"success" by skating at three times walking speed when the cap was missing).

  * travel is paid as PROGRESS toward a goal point standing on the platform,
    capped at a speed the robot can really produce;
  * the arrival bonus is gated on the whole landed state - on the platform,
    upright at standing height, past the wall overhang, and stopped - and the
    episode ends on success so it cannot be collected twice;
  * falling off the platform, or dropping back down the corridor, ends the
    episode and is charged once;
  * the brace is a foothold and is paid almost nothing, because w3 proved that
    paying to hang still beats paying to move.
"""
from __future__ import annotations

import math
import os as _os
from typing import Optional

import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.robot import exit_stage as ex
from mjlab_microduck.tasks.mdp import _DEFAULT_ASSET_CFG, _servo_joint_ids

STAND_Z = 0.115             # measured standing trunk height
STAND_BAND = 0.06       # was 0.026 as the anti-sprawl guard (the sprawl sat 2.5 cm
# low); that job now belongs to `standing_tall` (head carriage), and a LIVE
# standing policy leaves a 2.6 cm band within 0.5 s - measured 2026-09-22 from
# the success spawn itself: on_platform 23 % at t=0, 0 % at 0.5 s, so `landed`
# could never fire and x21 had no terminal to climb toward.
# Measured, not assumed (`probe_walkspeed.py`): the duck SUSTAINS about
# 0.30 m/s while its instantaneous centre-of-mass speed peaks at 0.5-0.9 on
# every push-off.  Charge the smoothed speed, or the penalty taxes ordinary
# locomotion and cannot be escaped by moving better.
MAX_PAID_SPEED = 0.35       # m/s of closing speed that still earns progress
SUSTAINED_CAP = 0.32        # m/s of SMOOTHED speed above which the dash is charged
SPEED_TAU_S = 0.50
# Standing on an open platform, not bracing.  At 40 deg the policy finished at
# a MEASURED 32.7 deg of tilt - leaning hard, which reads as a sprawl on video -
# because a tolerance is a target: whatever is allowed is what you get.  The
# climb's brace legitimately tolerates 60 deg; standing does not.
# 40 deg let it finish at a measured 32.7 (a sprawl on video); 18 deg removed
# the reward entirely and x14 collapsed to 99 % falls in 700 iterations.  26 is
# a real tightening that still leaves the bonus reachable from where the policy
# already is - tighten a gate the policy is sitting on, not past it.
TILT_MAX_DEG = 26.0
SLOW_MPS = 0.08
WIDTH_SCALE = 0.13


def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_ex", None)
    if st is None:
        z = torch.zeros(env.num_envs, device=env.device)
        st = {"width": z.clone() + 0.125, "prev_dist": z.clone(),
              "landed_steps": z.clone(), "spawn_z": z.clone(),
              "speed_ema": z.clone(), "stood": z.clone(), "lean_steps": z.clone()}
        env._ex = st
    return st


def _root(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg):
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.root_link_pos_w[:, :3] - env.scene.env_origins
    return pos, asset.data.root_link_quat_w, asset.data.root_com_lin_vel_w


def _upright(quat: torch.Tensor) -> torch.Tensor:
    x, y = quat[:, 1], quat[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def goal_point(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Standing on the platform, out past the wall overhang."""
    g = torch.zeros(env.num_envs, 3, device=env.device)
    g[:, 1] = ex.goal_y()
    g[:, 2] = ex.PLATFORM_TOP_M + STAND_Z
    return g


def _dist(env: ManagerBasedRlEnv, pos: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pos - goal_point(env), dim=-1)


def on_platform(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Standing anywhere on the platform - the intermediate state, which the
    walled part already reaches ~99 % of the time."""
    pos, quat, vel = _root(env, asset_cfg)
    over = (pos[:, 1] > ex.platform_near_y() + 0.06) & (pos[:, 1] < ex.platform_far_y())
    across = pos[:, 0].abs() < 0.5 * ex.PLATFORM_WIDTH_M
    tall = (pos[:, 2] - (ex.PLATFORM_TOP_M + STAND_Z)).abs() < STAND_BAND
    level = _upright(quat) > math.cos(math.radians(TILT_MAX_DEG))
    slow = torch.linalg.norm(vel, dim=-1) < SLOW_MPS
    return over & across & tall & level & slow


# --------------------------------------------------------------------------
# posture
# --------------------------------------------------------------------------
# Root-link tilt CANNOT tell standing from the sprawl this task finishes in:
# the hip assembly stays near vertical while the shell and head go down, so it
# reads 16.7 deg braced and 22.2 deg collapsed.  Head CARRIAGE separates them
# 2.7x - measured 2026-09-21 over 256 envs: jaw-above-ankle is 0.186 m on a
# braced arrival and 0.069 m after the policy settles into the sprawl.
# Where the exit takes over, as trunk height above the standing height.  It was
# (0.02, 0.22), i.e. FEET +6..+26 cm above the platform, and the climb duly
# handed over at the top of that band - measured 2026-09-21, feet 18-29 cm up
# (mean 22.6), which reads on camera as climbing out of the corridor.  Tightening
# only the render-side gate fought the policy: at feet +5 cm completions fell
# from 4/8 to 2/8 because that is BELOW what the exit was ever trained on.  So
# train it lower instead: feet roughly +2..+15 cm.
ABOVE_PLATFORM_RANGE = (
    float(_os.getenv("MICRODUCK_EX_ABOVE_LO", "-0.02")),
    float(_os.getenv("MICRODUCK_EX_ABOVE_HI", "0.12")),
)
HEAD_CARRY_STAND = 0.186        # measured, standing
HEAD_CARRY_MIN = float(_os.getenv("MICRODUCK_EX_HEAD_CARRY", "0.155"))  # was 0.130: x21 parked at 0.14
HEAD_CARRY_COLLAPSE = 0.080     # below this it is down, not merely crouched
HEAD_CARRY_STD = float(_os.getenv("MICRODUCK_EX_CARRY_STD", "0.05"))
LEVEL_STD_DEG = float(_os.getenv("MICRODUCK_EX_LEVEL_STD_DEG", "25.0"))  # 0.10 paid 0.81 at a 0.14 crouch; 0.05 pays 0.43 there and 1.0 standing
_BODY_IDX: dict = {}


def _carriage(env: ManagerBasedRlEnv,
              asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Jaw height above the lower foot: how tall the robot is CARRYING itself,
    independent of where the platform is or which way the hips point."""
    asset = env.scene[asset_cfg.name]
    key = id(asset)
    if key not in _BODY_IDX:
        nm = asset.body_names
        _BODY_IDX[key] = (
            [i for i, b in enumerate(nm) if "jaw" in b.lower()],
            [i for i, b in enumerate(nm) if "ankle" in b.lower()],
        )
    ji, ai = _BODY_IDX[key]
    bp = asset.data.body_link_pos_w
    return bp[:, ji, 2].mean(dim=1) - bp[:, ai, 2].min(dim=1).values


def standing_tall(env: ManagerBasedRlEnv,
                  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return _carriage(env, asset_cfg) > HEAD_CARRY_MIN


def collapsed(env: ManagerBasedRlEnv,
              asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Down on the platform.  A termination, because the old stack scored this
    as a SUCCESS: the trunk sits 2.5 cm under standing height against a 2.6 cm
    band, so the sprawl passed the height test by a millimetre.

    Gated on being OUT from between the walls.  Inside the slot the duck is
    braced and pitched and its head is low by design; ungated, this fired in
    112 of every 120 episodes within 0.7 s of spawning (x21, 2026-09-22) and
    the policy never got to try anything."""
    # Latched: a collapse is standing tall FIRST and then going down.  Gating on
    # position alone was not enough - frontier spawns are pitched and wedged
    # PAST the wall end, so it still ended 58 of 60 episodes at spawn.
    st = _st(env)
    pos, _, _ = _root(env, asset_cfg)
    car = _carriage(env, asset_cfg)
    tall_now = (pos[:, 1] > ex.wall_end_y()) & (car > HEAD_CARRY_MIN)
    st["stood"] = torch.maximum(st["stood"], tall_now.float())
    return (st["stood"] > 0.5) & (car < HEAD_CARRY_COLLAPSE)


def stillness_reward(env: ManagerBasedRlEnv,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Per-step pay for coming to REST once tall and out past the line.

    `landed` needs root speed < SLOW_MPS held for 2 s on top of everything
    else, and x21 reached carriage 1.06 / 410-step episodes with `landed` still
    0.0000 and `ex_clear` flat at 0.014 - the conjunction never fired by chance
    (2026-09-22), so nothing paid for slowing down.  Gaussian on root speed,
    std 0.15 so a pacing policy at 0.24 m/s already scores 0.08 and stillness
    scores 1; gated so it cannot be farmed inside the slot or lying down."""
    pos, _, vel = _root(env, asset_cfg)
    speed = torch.linalg.norm(vel, dim=-1)
    g = torch.exp(-((speed / 0.15) ** 2))
    gate = (pos[:, 1] > ex.clear_line_y()) & standing_tall(env, asset_cfg)
    return g * gate.float()


def landed_bonus(env: ManagerBasedRlEnv,
                 asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """One-shot on the step `landed` fires: finishing must be worth more than
    the per-step stream it cuts off (carriage + stillness for the rest of the
    episode), or camping before the goal is the argmax.  dt-scaled like every
    term, so weight 1500 = 30 units against ~12 s x (2+3) x dt = 1.2 units of
    camping - decisive, not marginal."""
    st = _st(env)
    return (st["landed_steps"] >= (2.0 / env.step_dt)).float()


def level_reward(env: ManagerBasedRlEnv,
                 asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Per-step pay for an UPRIGHT trunk once out past the line.

    Decomposed 2026-09-22 under x21 from the success spawn: past the line
    100 %, tall 100 %, speed 0.000 - and `level` 0 %: it converged to a leaned
    crouch (root tilt > 26 deg, carriage parked at the 0.13 minimum).  Nothing
    shaped tilt, so nothing pulled it upright.  Gaussian on root tilt, std
    12 deg, gated past the line and tall."""
    pos, quat, _ = _root(env, asset_cfg)
    tilt = torch.acos(_upright(quat).clamp(-1.0, 1.0))
    # std must make the CURRENT policy score visibly (same lesson as the
    # carriage std): at 12 deg a 30 deg lean scored 0.002 and ex_level sat at
    # 0.006; at 25 deg it scores 0.24 and upright scores 1.
    g = torch.exp(-((tilt / math.radians(LEVEL_STD_DEG)) ** 2))
    gate = (pos[:, 1] > ex.clear_line_y()) & standing_tall(env, asset_cfg)
    return g * gate.float()


def leaning(env: ManagerBasedRlEnv, hold_s: float = 1.0,
            asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Out past the line and leaned beyond TILT_MAX_DEG for `hold_s`: the
    resting crouch x21 converged to (2026-09-22: tall, still, carriage parked
    at the minimum, level 0 %).  It was a FREE rest pose - nothing ended or
    charged it - so nothing the shaping paid for could compete with it.
    A termination with a cost, like collapse.  Transient tilt while walking
    out does not count: past the line, held a full second."""
    st = _st(env)
    pos, quat, _ = _root(env, asset_cfg)
    # Anywhere OFF the slot, not just past the line: gated on the line, the
    # policy stepped 9 cm back behind it and leaned there (2026-09-22 19:28).
    lean = (pos[:, 1] > ex.platform_near_y() + 0.06) & (_upright(quat) < math.cos(math.radians(TILT_MAX_DEG)))
    st["lean_steps"] = torch.where(lean, st["lean_steps"] + 1.0, torch.zeros_like(st["lean_steps"]))
    return st["lean_steps"] >= (hold_s / env.step_dt)


def lean_cost(env: ManagerBasedRlEnv,
              asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot (POSITIVE weight) on the step `leaning` fires."""
    st = _st(env)
    return -(st["lean_steps"] >= (1.0 / env.step_dt)).float()


def collapse_cost(env: ManagerBasedRlEnv,
                  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot (POSITIVE weight) on the step `collapsed` fires.

    Any termination the policy can trigger must cost more than the rest of the
    episode, or triggering it becomes the cheapest policy.  With FALL_W at 200
    the fall exploit vanished and the SAME curve reappeared through this
    termination instead (x21, 2026-09-22: carriage 0.063 -> 0.006 as collapse
    went 18 -> 43 per window).  The latch in `collapsed` keeps this from
    charging the braced spawn."""
    return -collapsed(env, asset_cfg).float()


def carriage_reward(env: ManagerBasedRlEnv,
                    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Per-step pay for carrying the head high once out on the platform.

    A curriculum makes a rare reward FIRE; only per-step shaping makes the
    wanted route profitable, and standing up is exactly a route the old stack
    never paid for.  Gaussian on the MEASURED standing carriage, gated on being
    out past the walls so it cannot be farmed inside the slot.
    """
    c = _carriage(env, asset_cfg)
    # std must make the CURRENT policy score visibly or there is no gradient:
    # at 0.045 a sprawl (0.07) scored exp(-6.6) and ex_carry sat at 0.001 for
    # 34 iterations while collapse fell - i.e. the only live pressure was the
    # latch, which is avoidable by never standing.  At 0.10 the sprawl scores
    # 0.26 and standing 1.0, so rising pays every step.
    g = torch.exp(-(((c - HEAD_CARRY_STAND) / HEAD_CARRY_STD) ** 2))
    # Gated PAST THE LINE, not merely on the platform.  Paid anywhere on the
    # platform, x21 learned to stand tall just short of the finish and camp:
    # own-spawn probe 2026-09-22, carriage 0.198 / past-line 0.0 % at 15 s.
    # Reaching the line ends the episode and the income, so a per-step reward
    # that is available before the goal is an incentive never to reach it.
    pos, _, _ = _root(env, asset_cfg)
    return g * (on_platform(env, asset_cfg) & (pos[:, 1] > ex.clear_line_y())).float()


def clear_of_the_walls(env: ManagerBasedRlEnv,
                       asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Standing WELL OUT on the open part of the platform.

    The real finish (user 2026-09-20): the walls stop part way along the
    platform and the duck has to side-walk out from between them onto open
    platform with nothing left to brace against.

    The threshold is the middle of the open region, not the wall end itself.
    With the line drawn at the wall end, x8 cleared it 100 % of the time and
    stopped 2 cm past - because crossing ends the episode, so the argmax is to
    cross by a centimetre and stop.  A success test IS the objective; putting
    it at the edge buys an edge-case behaviour.
    """
    pos, _, _ = _root(env, asset_cfg)
    return (on_platform(env, asset_cfg)
            & (pos[:, 1] > ex.clear_line_y())
            & standing_tall(env, asset_cfg))


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
def exit_progress(env: ManagerBasedRlEnv, max_speed: float = MAX_PAID_SPEED,
                  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Metres of distance to the landing point given up this step, capped."""
    st = _st(env)
    pos, _, _ = _root(env, asset_cfg)
    d = _dist(env, pos)
    gain = torch.clamp(st["prev_dist"] - d, max=max_speed * env.step_dt)
    st["prev_dist"] = d
    return gain


def exit_landed(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Intermediate: on the platform at all, still possibly under the walls."""
    return on_platform(env, asset_cfg).float()


def exit_cleared(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """The finish: out past the wall end, standing on open platform."""
    return clear_of_the_walls(env, asset_cfg).float()


def exit_dash_penalty(env: ManagerBasedRlEnv, max_speed: float = SUSTAINED_CAP,
                      tau_s: float = SPEED_TAU_S,
                      asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Self-negating (POSITIVE weight): SMOOTHED horizontal speed above what the
    robot can sustain.

    Planar only, because charging 3-D speed bills a free fall as a dash and
    falling already costs.  Smoothed, because a gait's instantaneous speed
    peaks far above its sustained speed and an instantaneous threshold would
    charge ordinary locomotion."""
    st = _st(env)
    _, _, vel = _root(env, asset_cfg)
    sp = torch.linalg.norm(vel[:, :2], dim=-1)
    a = float(env.step_dt / max(tau_s, env.step_dt))
    ema = st.setdefault("speed_ema", torch.zeros_like(sp))
    if ema.shape != sp.shape:
        ema = torch.zeros_like(sp)
    st["speed_ema"] = ema + a * (sp - ema)
    return -torch.clamp(st["speed_ema"] - max_speed, min=0.0)


def exit_fall_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    return -exit_fell(env, asset_cfg=asset_cfg).float()


# --------------------------------------------------------------------------
# terminations
# --------------------------------------------------------------------------
def exit_fell(env: ManagerBasedRlEnv, drop_m: float = 0.35,
              asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Dropped back down the corridor, or off the platform.

    Measured against the platform top rather than the floor: this task lives a
    metre up, and a robot that has slid 40 cm back down the slot has failed the
    exit even though it is nowhere near the ground."""
    pos, quat, _ = _root(env, asset_cfg)
    fell_far = pos[:, 2] < (ex.PLATFORM_TOP_M - drop_m)
    tipped = _upright(quat) < math.cos(math.radians(80.0))
    return fell_far | tipped


def exit_landed_done(env: ManagerBasedRlEnv, hold_s: float = 0.5,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Success: standing CLEAR OF THE WALLS for ``hold_s``.  Landing under the
    overhang no longer ends the episode - it is the half-way point."""
    st = _st(env)
    now = clear_of_the_walls(env, asset_cfg)
    st["landed_steps"] = torch.where(now, st["landed_steps"] + 1.0,
                                     torch.zeros_like(st["landed_steps"]))
    return st["landed_steps"] >= (hold_s / env.step_dt)


# --------------------------------------------------------------------------
# observation
# --------------------------------------------------------------------------
def exit_command_obs(env: ManagerBasedRlEnv, dim: int = 6,
                     asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Body-command slot: where the landing is, relative to the robot, plus the
    corridor width.  [dy, dz, dx, 0, (width - 0.13) / 0.02, 0]"""
    st = _st(env)
    pos, _, _ = _root(env, asset_cfg)
    g = goal_point(env)
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, 0] = g[:, 1] - pos[:, 1]
    out[:, 1] = g[:, 2] - pos[:, 2]
    out[:, 2] = g[:, 0] - pos[:, 0]
    out[:, 4] = (st["width"] - WIDTH_SCALE) / 0.02
    return out


# --------------------------------------------------------------------------
# spawn
# --------------------------------------------------------------------------
_ARRIVAL_CACHE: dict = {}


def _arrival_bank(path: str, dev) -> Optional[dict]:
    key = (path, str(torch.device(dev)))
    if key in _ARRIVAL_CACHE:
        return _ARRIVAL_CACHE[key]
    try:
        blob = torch.load(path, map_location="cpu")
        out = {k: blob[k].to(dev) for k in ("qpos", "qvel", "origins")}
        out["platform_top"] = float(blob.get("platform_top", ex.PLATFORM_TOP_M))
    except Exception:
        out = None
    _ARRIVAL_CACHE[key] = out
    return out


def reset_exit_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    width_range: tuple = (0.125, 0.125),
    above_platform_range: tuple = ABOVE_PLATFORM_RANGE,
    brace_overrides: Optional[dict] = None,
    pitch_range: tuple = (-0.30, -0.10),
    joint_noise_std: float = 0.05,
    along_y_range: tuple = (-0.10, 0.02),
    frontier_prob: float = 0.40,
    nearly_done_prob: float = 0.20,
    stand_overrides: Optional[dict] = None,
    frontier_vy_range: tuple = (0.05, 0.30),
    handover_bank: Optional[str] = None,        # REAL climb-arrival states
    handover_prob: float = 0.0,
):
    """Walls (extended over the platform), platform, and the robot WEDGED just
    above platform level - the state the climb policy hands over in."""
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
        # The stepped wall carries its own two segments, positioned by height
        # inside the spec, so its BODY belongs on the floor.  Offsetting it by
        # half the old wall height put the whole corridor 1.5 m too high and
        # the duck spawned in mid-air with nothing to brace on.
        pos[:, 1] = 0.0
        pos[:, 2] = ex.stepped_wall_mocap_z()
        env.scene[name].write_mocap_pose_to_sim(torch.cat([origins + pos, quat0], -1), env_ids=env_ids)

    ppos = torch.tensor(ex.platform_mocap_pos(), device=dev).repeat(num, 1)
    env.scene[ex.PLATFORM].write_mocap_pose_to_sim(
        torch.cat([origins + ppos, quat0], -1), env_ids=env_ids)

    # REVERSE CURRICULUM.  x1/x2 spawned only at the handover - wedged in the
    # slot - and after 1250 iterations the landing bonus had fired exactly zero
    # times (`Episode_Reward/ex_land` 0.0000 in both runs).  Nothing was pulling
    # the robot out of a safe wedge because the goal state had never been
    # visited, so the critic had no value to back up.  Three spawn types now:
    #
    #   nearly done  - already standing over the platform: the landing reward
    #                  fires immediately and the hold gets learned;
    #   frontier     - wedged partway out over the platform: the traverse's
    #                  last stretch, where the on-policy data was missing;
    #   handover     - wedged in the slot, the state the climb really hands over.
    roll = torch.rand(num, device=dev)
    is_done = roll < nearly_done_prob
    is_frontier = (~is_done) & (roll < nearly_done_prob + frontier_prob)

    z = ex.PLATFORM_TOP_M + STAND_Z + _u(above_platform_range)
    y = _u(along_y_range)
    y = torch.where(is_frontier, _u((ex.platform_near_y() + 0.02, ex.goal_y())), y)
    # nearly-done must start PAST the success line (goal == clear line), or
    # half of them begin short of it and the hold never completes
    y = torch.where(is_done, _u((ex.goal_y() + 0.01, ex.goal_y() + 0.08)), y)
    z = torch.where(is_done, torch.full_like(z, ex.PLATFORM_TOP_M + STAND_Z), z)
    x = -0.5 * width + 0.055           # back shell on the left wall, feet on the right
    x = torch.where(is_done, torch.zeros_like(width), torch.full_like(width, 0.0) + x)
    pitch = _u(pitch_range)
    pitch = torch.where(is_done, torch.zeros_like(pitch), pitch)
    a = pitch / 2.0
    quat = torch.stack([torch.cos(a), torch.zeros_like(a), torch.sin(a), torch.zeros_like(a)], dim=1)
    env.sim.data.qpos[env_ids, 0] = origins[:, 0] + x
    env.sim.data.qpos[env_ids, 1] = origins[:, 1] + y
    env.sim.data.qpos[env_ids, 2] = origins[:, 2] + z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    # A frontier spawn must arrive the way a TRAVERSE arrives: moving.  x3
    # reaches out over the platform 99.6 % of the time from the real handover
    # and still lands only 2 %, while the same policy lands 67 % when the
    # battery includes frontier spawns - because those start from rest and the
    # traverse delivers the robot with momentum.  Practising a landing the
    # robot never actually experiences teaches the wrong landing.
    env.sim.data.qvel[env_ids, 1] = torch.where(
        is_frontier, _u(frontier_vy_range), torch.zeros(num, device=dev))

    servo_ids = _servo_joint_ids(env, asset)
    if brace_overrides:
        for jnt_idx, angle in brace_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            # A nearly-done spawn must be given STANDING joints explicitly: the
            # robot's default pose IS the brace, so "leave the joints alone"
            # produces a braced duck hovering over the platform, not a standing
            # one.  This exact mistake cost a withdrawn result on the climb.
            stand = stand_overrides.get(jnt_idx, angle) if stand_overrides else angle
            env.sim.data.qpos[env_ids, col] = torch.where(
                is_done,
                torch.full((num,), stand, device=dev),
                torch.full((num,), angle, device=dev),
            )
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    env.sim.data.qpos[env_ids[:, None], cols[None, :]] += (
        torch.randn(num, len(servo_ids), device=dev) * joint_noise_std)

    pos_now = env.sim.data.qpos[env_ids, :3] - origins
    full = torch.zeros(env.num_envs, 3, device=dev)
    full[env_ids] = pos_now
    st["prev_dist"][env_ids] = _dist(env, full)[env_ids]
    st["landed_steps"][env_ids] = 0.0
    st["stood"][env_ids] = 0.0
    st["lean_steps"][env_ids] = 0.0
    st["spawn_z"][env_ids] = z

    # REAL arrival states, captured off the climb at the instant the exit
    # takes over, applied LAST so every other per-env field is already set.
    # The exit is 94.9 % reliable from its own wedged-at-rest spawn and 0 for
    # 4 when handed a genuine climbing arrival at 3 m - the same gap the climb
    # had against the approach, and the same fix: train on the distribution
    # you will actually be handed.  Stored positions are absolute, so each is
    # shifted from its source env origin to this one; z is kept as-is only if
    # the platform height matches, so the bank is re-based onto PLATFORM_TOP.
    if handover_bank and handover_prob > 0.0:
        bank = _arrival_bank(handover_bank, dev)
        if bank is not None and bank["qpos"].shape[0] > 0:
            take = torch.rand(num, device=dev) < handover_prob
            if bool(take.any()):
                pick = torch.randint(0, bank["qpos"].shape[0], (num,), device=dev)
                bq = bank["qpos"][pick].clone()
                bv = bank["qvel"][pick].clone()
                bq[:, :3] += origins - bank["origins"][pick]
                bq[:, 2] += ex.PLATFORM_TOP_M - float(bank["platform_top"])
                sel = env_ids[take]
                env.sim.data.qpos[sel] = bq[take]
                env.sim.data.qvel[sel] = bv[take]
                st["spawn_z"][sel] = bq[take][:, 2] - origins[take][:, 2]
                pos_b = bq[take][:, :3] - origins[take]
                st["prev_dist"][sel] = _dist(env, torch.zeros(env.num_envs, 3, device=dev).index_copy_(0, sel, pos_b))[sel]
                st["landed_steps"][sel] = 0.0

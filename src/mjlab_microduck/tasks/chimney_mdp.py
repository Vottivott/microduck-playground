"""MDP terms for braced corridor climbing: wedge between two close walls and
work upward.

Grounded in the 2026-09-18 bracing measurement (`scripts/probe_chimney.py`,
`experiments/chimney/MEASUREMENT.md`): with the hips at -90 deg, back shell
on one wall and both feet on the other, the robot holds a 12 cm corridor at
wall friction as low as 0.3 and creeps upward 0.6 cm in 2 s, using a quarter
of the servo torque limit.  So the brace is free; what has to be learned is
the STEP - alternating press and lift to gain height repeatedly.

Reward shape follows the house rules:

  * height is PAID AS PROGRESS (increments of the max trunk height reached),
    so holding still pays zero and camping cannot be farmed;
  * nothing positive is gated on being in a bad state;
  * falling ends the episode and is charged once;
  * a brace bonus pays only while genuinely wedged (both feet on one wall and
    a non-foot part on the other), which is the state the climb happens from.

The policy is told the corridor width through the otherwise zero-padded
body-command slot: it is one constant an operator measures off the real
corridor once, and the parkour A/B on 2026-09-18 showed that withholding
such a constant while the curriculum varies it costs roughly half the
achievable difficulty.
"""
from __future__ import annotations

import math
from typing import Optional

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.tasks.mdp import _DEFAULT_ASSET_CFG, _servo_joint_ids

WIDTH_SCALE = 0.13           # command obs: (width - WIDTH_SCALE) / 0.02
FALL_MARGIN_M = 0.12         # dropping this far below the best height = fell
BRACE_TILT_MAX_DEG = 60.0


# --------------------------------------------------------------------------
# contacts (one raw scan per step, the stair-ladder pattern)
# --------------------------------------------------------------------------
def _geom_ids(env: ManagerBasedRlEnv) -> dict:
    cache = getattr(env, "_ch_geom_ids", None)
    if cache is not None:
        return cache
    model = env.sim.mj_model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    feet = [names.index("robot/left_foot_collision"), names.index("robot/right_foot_collision")]
    body = [g for g, n in enumerate(names)
            if n.startswith("robot/") and int(model.geom_contype[g]) != 0 and g not in feet]
    head = [g for g, n in enumerate(names)
            if n.startswith("robot/") and int(model.geom_contype[g]) != 0
            and any(k in n for k in ("jaw", "head"))]
    left = [g for g, n in enumerate(names) if n.startswith(cs.WALL_LEFT)]
    right = [g for g, n in enumerate(names) if n.startswith(cs.WALL_RIGHT)]
    if not left or not right:
        raise ValueError("corridor walls missing from the scene")
    terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    terrain = []
    if terrain_body >= 0:
        start = int(model.body_geomadr[terrain_body])
        terrain = list(range(start, start + int(model.body_geomnum[terrain_body])))
    dev = env.device
    t = lambda xs: torch.tensor(xs, device=dev, dtype=torch.int32)  # noqa: E731
    cache = {"feet": t(feet), "body": t(body), "head": t(head),
             "left": t(left), "right": t(right), "terrain": t(terrain)}
    env._ch_geom_ids = cache
    return cache


def _contacts(env: ManagerBasedRlEnv) -> dict:
    step = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_ch_contacts", None)
    if cache is not None and cache.get("step") == step:
        return cache
    n, dev = env.num_envs, env.device
    ids = _geom_ids(env)
    z = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    feet_left, feet_right = z(), z()
    body_left, body_right = z(), z()
    on_floor, head_touch = z(), z()
    count = int(env.sim.data.nacon[0].item())
    if count > 0:
        pairs = env.sim.data.contact.geom[:count]
        world = env.sim.data.contact.worldid[:count].long()
        touching = env.sim.data.contact.dist[:count] <= 0.0
        g0, g1 = pairs[:, 0], pairs[:, 1]

        def hit(a, b):
            return touching & ((torch.isin(g0, a) & torch.isin(g1, b)) | (torch.isin(g1, a) & torch.isin(g0, b)))

        for flag, a, b in ((feet_left, ids["feet"], ids["left"]), (feet_right, ids["feet"], ids["right"]),
                           (body_left, ids["body"], ids["left"]), (body_right, ids["body"], ids["right"]),
                           (head_touch, ids["head"], torch.cat([ids["left"], ids["right"]]))):
            m = hit(a, b)
            if bool(m.any()):
                flag[world[m]] = True
        if len(ids["terrain"]):
            m = hit(torch.cat([ids["feet"], ids["body"]]), ids["terrain"])
            if bool(m.any()):
                on_floor[world[m]] = True
    cache = {"step": step, "feet_left": feet_left, "feet_right": feet_right,
             "body_left": body_left, "body_right": body_right,
             "on_floor": on_floor, "head_touch": head_touch}
    env._ch_contacts = cache
    return cache


def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_ch", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        st = dict(width=z(), spawn_z=z(), max_z=z(), held_z=z(), paid=z(),
                  ground_start=torch.zeros(n, dtype=torch.bool, device=dev),
                  ever_braced=torch.zeros(n, dtype=torch.bool, device=dev),
                  upright_start=torch.zeros(n, dtype=torch.bool, device=dev),
                  last_gain=torch.zeros(n, dtype=torch.long, device=dev),
                  failed=torch.zeros(n, dtype=torch.bool, device=dev), last_step=-1)
        env._ch = st
    return st


def _update(env: ManagerBasedRlEnv, asset: Entity) -> dict:
    st = _st(env)
    step = int(env.common_step_counter)
    if step == st["last_step"]:
        return st
    st["last_step"] = step
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    st["max_z"] = torch.maximum(st["max_z"], z)
    # Height only counts while the robot is actually WEDGED and upright.  Paying
    # the instantaneous peak was a jackpot: run w4 learned to lunge, banked
    # 14 cm of transient height (800 x 0.14 = 112) and fell (cost 10), every
    # episode, braced for only 17 % of the time and surviving 1.3 s.  A lunge
    # that breaks the brace now pays nothing.
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    holding = braced(env) & (tilt < math.radians(BRACE_TILT_MAX_DEG))
    held = torch.where(holding, torch.maximum(st["held_z"], z), st["held_z"])
    st["ever_braced"] = st["ever_braced"] | holding
    gained = held > (st["held_z"] + 1e-4)
    st["held_z"] = held
    st["last_gain"] = torch.where(gained, torch.full_like(st["last_gain"], step), st["last_gain"])
    c = _contacts(env)
    # The head touching a wall is NOT a failure: in a 12 cm corridor the duck's
    # head spans the gap by construction (measured at spawn: 98 % correctly
    # wedged, 89 % with the head on a wall), so terminating on it ended every
    # episode in one step.  It is priced instead - the landing-roll measurement
    # showed the neck servo is the fragile part.
    # A GROUND start begins on the floor, so touching it cannot be what ends the
    # episode - not until the robot has been wedged at least once.  Without this
    # latch every floor spawn would terminate on its first step, the way the
    # flip's fatal-hop rule killed itself on the spawn bounce.
    floor_fatal = c["on_floor"] & (~st["ground_start"] | st["ever_braced"])
    st["failed"] = floor_fatal | (z < (st["held_z"] - FALL_MARGIN_M))
    return st


def braced(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Wedged: both feet pressing one wall and a non-foot part the other."""
    c = _contacts(env)
    return (c["feet_right"] & c["body_left"]) | (c["feet_left"] & c["body_right"])


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
def climb_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Paid increments of the highest trunk height reached WHILE WEDGED (m).

    Potential-based, so rising pays, hanging pays zero and sliding back earns
    nothing on the way up again - and, since only braced height counts, a lunge
    that leaves the wedge pays nothing at all (w4 learned exactly that lunge).
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    inc = torch.clamp(st["held_z"] - st["paid"], min=0.0)
    st["paid"] = st["paid"] + inc
    return inc


def climb_brace(env: ManagerBasedRlEnv, max_tilt_deg: float = BRACE_TILT_MAX_DEG,
                asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 per step while genuinely wedged and roughly upright: the state the climb
    happens from.  Small weight - it is a foothold, not the goal."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    g = asset.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-torch.nan_to_num(g[:, 2], nan=-1.0), -1.0, 1.0))
    return (braced(env) & (tilt < math.radians(max_tilt_deg))).float()


def climb_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Height gained above the spawn (m) on each step: diagnostic, weight 1."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.clamp(z - st["spawn_z"], min=0.0)


def climb_fall_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING one-shot when the fall termination fires (POSITIVE weight)."""
    return -climb_fell(env, asset_cfg=asset_cfg).float()


def climb_slip_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step not braced (both walls need to be loaded)."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return -(~braced(env)).float()


def climb_head_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """SELF-NEGATING: -1 per step with the head loaded against a wall.  Unavoidable
    in a narrow corridor, so it is priced, not fatal - but the neck servo is the
    part the landing-roll measurement flagged as fragile, so it is not free."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return -_contacts(env)["head_touch"].float()


def climb_stall_penalty(
    env: ManagerBasedRlEnv,
    stall_after_s: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING: -1 per step once the robot has gone ``stall_after_s``
    without reaching a new high point.

    Run w3 found the obvious local optimum: hold the wedge and never move (98 %
    of steps braced, no falls, 4 mm climbed in 6 s).  Holding has to cost
    something, or standing still beats every attempt to step - the same trap the
    long jump hit, where the fix was a dawdle tax.
    """
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    idle = (int(env.common_step_counter) - st["last_gain"]).float() * env.step_dt
    return -(idle > stall_after_s).float()


def climb_fell(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Reached the floor, or dropped FALL_MARGIN_M below the best height reached.

    Head-on-wall is deliberately NOT here: see the note in ``_update``."""
    asset: Entity = env.scene[asset_cfg.name]
    st = _update(env, asset)
    return st["failed"]


def chimney_command_obs(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Body-command slot: [(width - 0.13) / 0.02, 0, ...] - one constant, measured
    off the real corridor once (no perception)."""
    st = _st(env)
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, 0] = (st["width"] - WIDTH_SCALE) / 0.02
    return out


# --------------------------------------------------------------------------
# spawn
# --------------------------------------------------------------------------
_BANK_CACHE: dict = {}


def _handover_bank(path: str, dev) -> Optional[dict]:
    """Load and cache a bank of real handover states (qpos, qvel, origins)."""
    if path in _BANK_CACHE:
        return _BANK_CACHE[path]
    try:
        blob = torch.load(path, map_location="cpu")
        out = {k: blob[k].to(dev) for k in ("qpos", "qvel", "origins")}
    except Exception:
        out = None
    _BANK_CACHE[path] = out
    return out


def reset_chimney_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    width_range: tuple = (cs.WIDTH_MIN_M, cs.WIDTH_MAX_M),
    height_range: tuple = (0.25, 1.30),
    brace_overrides: Optional[dict] = None,
    pitch_range: tuple = (-0.30, -0.10),
    joint_noise_std: float = 0.06,
    ground_prob: float = 0.0,
    ground_height_range: tuple = (0.06, 0.13),
    upright_prob: float = 0.0,     # of the ground starts, this fraction stand upright
    upright_overrides: Optional[dict] = None,   # joint angles for the standing pose
    upright_height: float = 0.115,  # measured standing trunk height
    upright_yaw_range: tuple = (0.0, 0.0),      # a real handover is not square
    upright_x_range: tuple = (0.0, 0.0),        # ... nor dead centre
    upright_jvel_std: float = 0.0,              # ... nor perfectly still
    handover_bank: Optional[str] = None,        # a file of REAL handover states
    handover_prob: float = 0.0,
):
    """Place the walls and spawn the robot already WEDGED at a random height.

    Starting from the measured brace is the reverse curriculum: getting into
    the wedge from the floor is a separate skill, and without mid-corridor
    spawns the climb itself would never be practised.
    """
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

    width = _u(width_range)
    quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(num, 1)
    for side, name in ((-1.0, cs.WALL_LEFT), (1.0, cs.WALL_RIGHT)):
        pos = torch.zeros(num, 3, device=dev)
        pos[:, 0] = side * (0.5 * width + 0.5 * cs.WALL_THICKNESS_M)
        pos[:, 2] = cs.wall_z()
        env.scene[name].write_mocap_pose_to_sim(torch.cat([origins + pos, quat0], -1), env_ids=env_ids)

    is_ground = torch.rand(num, device=dev) < ground_prob
    # The bottom of the climb is its own skill: every run so far has started
    # already wedged between 25 cm and 1.3 m up, so entering the corridor has
    # never been trained at all.  A ground start begins just above the floor,
    # still in the measured brace, and the floor is not fatal until it has been
    # wedged once (see _update).
    # An UPRIGHT ground start: legs down, standing on the floor midway between
    # the walls, which is the state an approach policy would hand over (user
    # 2026-09-19: a separate policy walks in, then switches to this one).  The
    # climb's own ground starts are already braced, so an upright handover is
    # outside its distribution unless it is trained here too.
    is_upright = is_ground & (torch.rand(num, device=dev) < upright_prob)
    z = torch.where(is_ground, _u(ground_height_range), _u(height_range))
    z = torch.where(is_upright, torch.full_like(z, upright_height), z)
    pitch = torch.where(is_upright, torch.zeros(num, device=dev), _u(pitch_range))
    # the back shell sits against the left wall, the feet reach the right one -
    # but an upright start stands in the middle of the corridor
    x = torch.where(is_upright, torch.zeros(num, device=dev),
                    torch.full((num,), 0.0, device=dev) + (-0.5 * width + 0.055))
    a = pitch / 2.0
    quat = torch.stack([torch.cos(a), torch.zeros_like(a), torch.sin(a), torch.zeros_like(a)], dim=1)
    # An UPRIGHT spawn that is exactly square, exactly centred and exactly still
    # is not what an approach policy delivers.  Measured from a9: yaw 5.8 deg,
    # x 1.0 cm off centre, joint velocity 0.03 rad/s, joints within 0.086 rad -
    # and the climb, trained only on the perfect version, falls over instead of
    # bracing when handed the real one.  Train on the real distribution.
    if is_upright.any() and (upright_yaw_range[1] > 0.0 or upright_x_range[1] > 0.0):
        yaw_j = _u(upright_yaw_range) * torch.sign(torch.randn(num, device=dev))
        half = 0.5 * yaw_j
        cy, sy = torch.cos(half), torch.sin(half)
        cp, sp = torch.cos(a), torch.sin(a)
        # pitch about y then yaw about z
        q_up = torch.stack([cy * cp, -sy * sp, cy * sp, sy * cp], dim=1)
        quat = torch.where(is_upright[:, None], q_up, quat)
        x = torch.where(is_upright, _u(upright_x_range) * torch.sign(torch.randn(num, device=dev)), x)
    env.sim.data.qpos[env_ids, 0] = origins[:, 0] + x
    env.sim.data.qpos[env_ids, 1] = origins[:, 1] + (torch.rand(num, device=dev) * 2 - 1) * 0.01
    env.sim.data.qpos[env_ids, 2] = origins[:, 2] + z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0
    if upright_jvel_std > 0.0:
        jit = torch.randn(num, len(_servo_joint_ids(env, asset)), device=dev) * upright_jvel_std
        jcols = torch.tensor([6 + j for j in _servo_joint_ids(env, asset)], device=dev, dtype=torch.long)
        cur = env.sim.data.qvel[env_ids[:, None], jcols[None, :]]
        env.sim.data.qvel[env_ids[:, None], jcols[None, :]] = torch.where(
            is_upright[:, None], jit, cur)

    servo_ids = _servo_joint_ids(env, asset)
    cols = torch.tensor([7 + j for j in servo_ids], device=dev, dtype=torch.long)
    # An UPRIGHT ground start: legs down, standing on the floor between the
    # walls, which is the state an approach policy would hand over (user
    # 2026-09-19: a separate policy walks in and then switches to this one).
    # The climb's own ground starts are already braced, so an upright handover
    # is out of its distribution unless it is trained here too.
    if brace_overrides:
        for jnt_idx, angle in brace_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            # An upright start must be written EXPLICITLY.  Keeping "whatever
            # qpos already holds" does not give a standing pose: the robot's
            # default init_state IS the brace (that is what makes holding still
            # hold the wedge), so the reset had already written braced legs and
            # the upright spawn came out braced - it only differed in height,
            # pitch and x.  Caught by the user watching the clip, 2026-09-19.
            up = upright_overrides.get(jnt_idx, angle) if upright_overrides else angle
            env.sim.data.qpos[env_ids, col] = torch.where(
                is_upright,
                torch.full_like(is_upright, up, dtype=torch.float),
                torch.full_like(is_upright, angle, dtype=torch.float),
            )
    if joint_noise_std > 0.0:
        env.sim.data.qpos[env_ids.unsqueeze(1), cols.unsqueeze(0)] += torch.randn(num, len(cols), device=dev) * joint_noise_std

    st["width"][env_ids] = width
    st["spawn_z"][env_ids] = z
    st["max_z"][env_ids] = z
    st["held_z"][env_ids] = z
    st["paid"][env_ids] = z
    st["failed"][env_ids] = False
    st["ground_start"][env_ids] = is_ground
    st["upright_start"][env_ids] = is_upright
    st["ever_braced"][env_ids] = False

    # REAL handover states, sampled from a bank captured off the approach
    # policy, applied LAST so every other per-env field is already correct and
    # only the pose is replaced.  Synthetic noise around the perfect standing
    # pose did not transfer: a duck PLACED standing climbs ~92 % while the same
    # duck ARRIVING from the approach climbs 0 %, with every summary statistic
    # of the two states matching.  Train on the actual distribution instead of
    # continuing to hunt the difference.
    if handover_bank and handover_prob > 0.0:
        bank = _handover_bank(handover_bank, dev)
        if bank is not None and bank["qpos"].shape[0] > 0:
            take = torch.rand(num, device=dev) < handover_prob
            if bool(take.any()):
                pick = torch.randint(0, bank["qpos"].shape[0], (num,), device=dev)
                bq = bank["qpos"][pick].clone()
                bv = bank["qvel"][pick].clone()
                bq[:, :3] += origins - bank["origins"][pick]
                sel = env_ids[take]
                env.sim.data.qpos[sel] = bq[take]
                env.sim.data.qvel[sel] = bv[take]
                zb = bq[take][:, 2] - origins[take][:, 2]
                st["spawn_z"][sel] = zb
                st["max_z"][sel] = zb
                st["held_z"][sel] = zb
                st["paid"][sel] = zb
                st["ground_start"][sel] = True      # it is standing on the floor
                st["upright_start"][sel] = True
                st["ever_braced"][sel] = False

"""The helpers the butt walk adds to mjlab_microduck/tasks/mdp.py (append them there; they use its imports: math,
torch, ManagerBasedRlEnv, SceneEntityCfg, Entity, _DEFAULT_ASSET_CFG, and its head_pose_tracking)."""

# ── Butt walk (seated locomotion on command) ─────────────────────────────────

def root_height_above(
    env: ManagerBasedRlEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the trunk rises above ``max_height`` (world z above the env origin).

    The butt walk's "stood up" guard: walking on the feet is the easy way to move, so leaving the seated band ends
    the episode (losing every future per-step reward) instead of being priced with a small penalty.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    return torch.nan_to_num(z, nan=0.0) > max_height


def height_above_l1_penalty(
    env: ManagerBasedRlEnv,
    max_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """One-sided L1 on trunk z above ``max_height`` (≤ 0: use a POSITIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.clamp(z - max_height, min=0.0)


def command_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    axis: str = "lin_x",
    min_cmd: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Progress along one commanded rate, linear and capped: achieved / commanded, clipped to [-1, 1] (moving the wrong way
    costs), 0 while that command is below ``min_cmd``.

    The Gaussian tracking terms are flat near zero motion when the commands are small (a seated duck that never moves
    already earns most of them), so a seated stroke that is only starting to work gets no gradient. This pays for any
    motion the right way, and nothing beyond the commanded rate (no jackpot). ``axis``: "lin_x" or "ang_z".
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    if axis == "lin_x":
        c, v = cmd[:, 0], asset.data.root_link_lin_vel_b[:, 0]
    else:
        c, v = cmd[:, 2], asset.data.root_link_ang_vel_b[:, 2]
    active = c.abs() > min_cmd
    r = torch.clamp(v / torch.where(active, c, torch.ones_like(c)), -1.0, 1.0)
    return torch.nan_to_num(torch.where(active, r, torch.zeros_like(r)), nan=0.0)


def _head_pose_error(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N, 4) neck/head joint error vs the commanded deltas from the default pose (as head_pose_tracking measures it)."""
    head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)      # caches the joint ids
    asset: Entity = env.scene[asset_cfg.name]
    ids = env._head_pose_neck_ids
    measured = asset.data.joint_pos[:, ids] + asset.data.joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    return measured - asset.data.default_joint_pos[:, ids] - env.command_manager.get_command(command_name)


def head_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean |error| of the 4 neck/head joints vs the commanded pose (≤ 0: use a POSITIVE weight).

    The Gaussian head_pose_tracking is flat once the head is far off, so a policy that gains elsewhere by moving its
    (heavy) head simply gives it up. The butt walk did: it bowed the head ~85 deg forward to put its weight on its feet.
    A linear price keeps a constant pull back toward the commanded pose."""
    return torch.nan_to_num(-_head_pose_error(env, command_name, asset_cfg).abs().mean(dim=-1), nan=0.0)


def head_off_command(
    env: ManagerBasedRlEnv,
    max_error: float,
    command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when any neck/head joint is more than ``max_error`` rad off the commanded pose."""
    return torch.nan_to_num(_head_pose_error(env, command_name, asset_cfg).abs(), nan=0.0).max(dim=-1).values > max_error

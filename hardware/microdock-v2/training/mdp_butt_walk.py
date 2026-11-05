"""The helpers the butt walk adds to mjlab_microduck/tasks/mdp.py, for reading: butt_walk.patch adds them (and the
straight-command option of VelocityCommandCommandOnly) to a microduck_rl checkout at bae39f59. They use mdp.py's
imports: math, torch, ManagerBasedRlEnv, SceneEntityCfg, Entity, _DEFAULT_ASSET_CFG, and its head_pose_tracking."""

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


def yaw_rate_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    cap: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """-|yaw rate - commanded yaw rate| (rad/s, capped at ``cap``): <= 0, POSITIVE weight.

    The Gaussian yaw tracking (std 0.25 rad/s) is flat by 0.5 rad/s of error, which is where the butt walk's undocking
    runs settled: backing up at -0.05 m/s with no turn commanded they spun at 0.3-0.65 rad/s, more than the docking's
    heading hold (0.3 rad/s at most) can take back. A linear cost keeps its slope at any error."""
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err = (asset.data.root_link_ang_vel_b[:, 2] - cmd[:, 2]).abs().clamp(max=cap)
    return torch.nan_to_num(-err, nan=0.0)


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


def tilt_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear tilt cost: -|projected gravity xy| (~ -sin(tilt); ≤ 0: use a POSITIVE weight).

    A Gaussian upright term has no gradient far from upright: a policy that settles ~9 deg off earns ~nothing from it and
    nothing pulls it back. This pulls equally at any tilt."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(-torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1), nan=0.0)


def pitch_target_l1(
    env: ManagerBasedRlEnv,
    target_deg: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear cost on the trunk's pitch away from ``target_deg`` (positive = leaning forward) plus its roll (rad; ≤ 0:
    use a POSITIVE weight). The butt walk scoots leaning ~2-3 deg forward and drifted to ~6.5 deg at rest; a dock that
    meets the trunk in its own frame needs one consistent seated pitch, and pulling to 0 instead killed the stroke."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2]); roll = torch.atan2(g[:, 1], -g[:, 2])
    return torch.nan_to_num(-(torch.abs(pitch - math.radians(target_deg)) + torch.abs(roll)), nan=0.0)


def rest_pitch_l1(
    env: ManagerBasedRlEnv,
    target_deg: float = 3.0,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """At the zero command only: linear cost on the trunk's pitch away from ``target_deg`` (rad; ≤ 0: POSITIVE weight).

    The seated stroke rocks the trunk (pitch and roll) to unload the seat; any tilt cost while moving killed it. At rest
    the policy drifted to ~7-8 deg forward, past the ~2-3 deg it scoots at, so a dock it backed into at the scooting pitch
    lost the connection when it settled. This pins the rest pitch to the scooting lean and leaves the stroke free."""
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    still = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < command_threshold
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    return torch.nan_to_num(-torch.abs(pitch - math.radians(target_deg)) * still.float(), nan=0.0)


# ── Butt walk: undocking (the MicroDock's magnet and hood) ─────────────────────────────────────────────────────────────
# Docked, the magnetic tip in the duck's USB-C port is held by the adapter in the dock's hood, and the hood's top plate
# rests on the duck's back. To leave, the butt walk has to pull the tip off the magnets and scoot out from under the hood
# (standing up under it pushes the trunk into the hood). Some episodes start "docked": the port's position on the first
# step is the adapter's face. The magnet pulls the port back toward it (the CPU docking simulation's model: the full pull
# up to 1 mm, a quarter of it at 4 mm, nothing beyond), the adapter face is a damped stop behind it, and while the top
# face is under the hood the trunk can rise at most 6 mm (the hood's lift in the board's window), then the hood's lip
# lets it out. All three act on trunk_base as external forces at their points (the trunk's COM + the matching torque).

UNDOCK_PORT_B = (-0.05053, 0.0, 0.0265)     # the magnetic tip's face in the trunk_base frame (m)
UNDOCK_TOP_B = (-0.030, 0.0, 0.0406)        # where the hood's skids press on the top face (m)
UNDOCK_UNDER_MM, UNDOCK_LIP_MM, UNDOCK_LIFT = 21.0, 7.0, 0.006
UNDOCK_YAW_FREE_DEG, UNDOCK_YAW_OPEN_DEG_PER_MM, UNDOCK_YAW_GAP_MM = 3.0, 7.0/11.0, 20.0


def magnet_axial_force(gap_mm: torch.Tensor, fmax: torch.Tensor, v_axis: torch.Tensor | None = None,
                       range_mm: float = 4.0, k_stop: float = 2000.0, c_stop: float = 40.0, lead_s: float = 0.01) -> torch.Tensor:
    """Force (N) on the port along the dock axis (positive: away from the dock) at gap_mm (mm, positive: away from the
    adapter face): the magnet's pull and the adapter face as a damped stop (k_stop N/m, c_stop N s/m) behind it. The
    force is held for a whole control step (20 ms), so the stop acts on the gap lead_s ahead: a stop computed on the
    step's start state returns more energy than it took (a 1-D check: hit at 0.05 m/s, out at 0.061 m/s); this one
    returns less, and none with the seat's friction. The real face is rigid: a springy stop must not be worth bouncing."""
    pull = fmax*torch.clamp(1.25 - gap_mm.clamp(min=0.0)/range_mm, max=1.0)*(gap_mm < range_mm).float()
    ahead = gap_mm if v_axis is None else gap_mm + 1000.0*lead_s*v_axis
    stop = k_stop*1e-3*(-ahead).clamp(min=0.0)
    if v_axis is not None:
        stop = stop - c_stop*v_axis*(ahead < 0.0).float()
    return stop - pull


def hood_ceiling(gap_mm: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """The highest the trunk origin may be (m) with the port gap_mm forward of docked: the hood's lift above the docked
    height, rising with the lip as the top face slides out (the caller drops the ceiling past the lip)."""
    ramp = ((gap_mm - UNDOCK_UNDER_MM)/UNDOCK_LIP_MM).clamp(0.0, 1.0)
    return z0 + UNDOCK_LIFT*(1.0 + ramp)


def hood_yaw_torque(dyaw: torch.Tensor, wz: torch.Tensor, gap_mm: torch.Tensor, k: float = 2.0, c: float = 0.03,
                    lead_s: float = 0.01) -> torch.Tensor:
    """Yaw torque (N m, about world z) of the hood on the docked duck turned dyaw (rad) from its docked heading: free
    within a play that opens as the port comes out (3 deg docked, +7 deg per 11 mm), a spring-damper stop beyond it
    (on the yaw lead_s ahead, as the face's stop), and nothing from UNDOCK_YAW_GAP_MM out. From the CPU docking
    simulation: a yaw torque of 0.2 N m turns the docked duck 4 deg, 10 deg 11 mm out, and freely past ~20 mm.
    Without it the magnet only acts along the axis, and turning 30-45 deg in place slid the port (5 cm behind the
    trunk) 7-15 mm out along it: the magnet let go and the progress paid. The round-2 policies twisted out of the
    dock that way, and kept spinning when backing up."""
    free = torch.deg2rad(UNDOCK_YAW_FREE_DEG + UNDOCK_YAW_OPEN_DEG_PER_MM*gap_mm.clamp(min=0.0))
    ahead = dyaw + lead_s*wz
    over = torch.sign(ahead)*(ahead.abs() - free).clamp(min=0.0)
    tau = -k*over - c*wz*(over != 0).float()
    return tau*(gap_mm < UNDOCK_YAW_GAP_MM).float()


def _undock_state(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_undock", None)
    if st is None:
        n, d = env.num_envs, env.device
        st = dict(active=torch.zeros(n, dtype=torch.bool, device=d), set=torch.zeros(n, dtype=torch.bool, device=d),
                  fmax=torch.zeros(n, device=d), vx=torch.zeros(n, device=d), anchor=torch.zeros(n, 3, device=d),
                  axis=torch.zeros(n, 3, device=d), z0=torch.zeros(n, device=d), yaw0=torch.zeros(n, device=d), gap=torch.zeros(n, device=d),
                  gap_prev=torch.zeros(n, device=d), cmd_s=0.0)
        env._undock = st
    return st


def undock_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, prob: float, fmax_range: tuple[float, float],
                 vx_range: tuple[float, float], cmd_s: float) -> None:
    """Reset event: which episodes start docked (prob), their magnet's pull (N, uniform in fmax_range) and their forward
    command (m/s, held for the first cmd_s seconds). The dock's position is taken on the first step (undock_forces)."""
    st = _undock_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    st["active"][env_ids] = torch.rand(n, device=env.device) < prob
    st["fmax"][env_ids] = torch.empty(n, device=env.device).uniform_(*fmax_range)
    st["vx"][env_ids] = torch.empty(n, device=env.device).uniform_(*vx_range)
    st["set"][env_ids] = False
    st["gap"][env_ids] = 0.0
    st["gap_prev"][env_ids] = 0.0
    st["cmd_s"] = cmd_s


def undock_forces(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str = "twist",
                  k_ceiling: float = 3000.0, c_ceiling: float = 40.0, lead_s: float = 0.01, k_yaw: float = 2.0,
                  c_yaw: float = 0.03, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=("trunk_base",))) -> None:
    """Step event: the magnet, the adapter face, the hood's ceiling and its hold on the heading (hood_yaw_torque, k_yaw
    0 = off) on the docked episodes (zero force elsewhere), and their forward command for the first cmd_s seconds
    (written after the command manager resamples)."""
    del env_ids
    st = _undock_state(env)
    asset: Entity = env.scene[asset_cfg.name]
    bid = int(asset_cfg.body_ids[0])
    q = asset.data.body_link_quat_w[:, bid]; p = asset.data.body_link_pos_w[:, bid]
    com = asset.data.body_com_pos_w[:, bid]; v = asset.data.body_link_lin_vel_w[:, bid]
    port = p + quat_apply(q, torch.tensor(UNDOCK_PORT_B, device=env.device, dtype=p.dtype).expand_as(p))
    top = p + quat_apply(q, torch.tensor(UNDOCK_TOP_B, device=env.device, dtype=p.dtype).expand_as(p))
    new = st["active"] & ~st["set"]
    if bool(new.any()):
        fwd = quat_apply(q[new], torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=p.dtype).expand(int(new.sum()), 3)).clone()
        fwd[:, 2] = 0.0
        st["axis"][new] = fwd/fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        st["yaw0"][new] = torch.atan2(fwd[:, 1], fwd[:, 0])
        st["anchor"][new] = port[new]; st["z0"][new] = p[new, 2]; st["set"] |= new
    act = (st["active"] & st["set"]).float()
    gap = ((port - st["anchor"])*st["axis"]).sum(-1)*1000.0
    f_ax = magnet_axial_force(gap, st["fmax"], (v*st["axis"]).sum(-1))*act
    f_port = st["axis"]*f_ax[:, None]
    pen = (p[:, 2] + lead_s*v[:, 2] - hood_ceiling(gap, st["z0"])).clamp(min=0.0)*act*(gap < UNDOCK_UNDER_MM + UNDOCK_LIP_MM).float()
    f_top = torch.zeros_like(p)
    f_top[:, 2] = -(k_ceiling*pen + c_ceiling*v[:, 2].clamp(min=0.0)*(pen > 0).float())
    force = f_port + f_top
    torque = torch.cross(port - com, f_port, dim=-1) + torch.cross(top - com, f_top, dim=-1)
    if k_yaw > 0:
        x_w = quat_apply(q, torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=p.dtype).expand_as(p))
        dyaw = wrap_to_pi(torch.atan2(x_w[:, 1], x_w[:, 0]) - st["yaw0"])
        wz = asset.data.body_link_ang_vel_w[:, bid, 2]
        torque[:, 2] += hood_yaw_torque(dyaw, wz, gap, k=k_yaw, c=c_yaw)*act
    asset.write_external_wrench_to_sim(forces=force[:, None, :], torques=torque[:, None, :], body_ids=[bid])
    st["gap_prev"] = st["gap"]
    st["gap"] = gap*act
    hold = (act > 0) & (env.episode_length_buf.float()*env.step_dt < st["cmd_s"])
    if bool(hold.any()):
        cmd = env.command_manager.get_term(command_name).vel_command_b
        cmd[hold, 0] = st["vx"][hold]; cmd[hold, 1] = 0.0; cmd[hold, 2] = 0.0


def undock_progress(env: ManagerBasedRlEnv, lo_mm: float = -3.0, hi_mm: float = 12.0) -> torch.Tensor:
    """Docked episodes only: potential-based progress away from the adapter face (≥ 0 when moving out: POSITIVE weight).

    Stuck at the magnet, the duck moves nothing, so the velocity terms give no sign that pushing harder helps until it
    is free. But the adapter face is a spring here: pushing harder already moves the port out of it (about -2 mm at 4 N
    resting, toward 0 as the push grows, free past 1 mm). This pays the change of the gap, clipped to [lo_mm, hi_mm], per
    second: an episode earns weight x its net progress (14 mm from resting at 4 N to free), and rocking back and forth
    earns nothing. One step late (the gap is measured after the rewards), which the telescoping sum doesn't mind.

    An episode that terminates (falls, head off, stood up) pays its progress back: the end state's potential counts as 0.
    Without that, lunging out and falling kept the lunge's pay: with weights 2-3 every episode ended in 0.2 s."""
    st = getattr(env, "_undock", None)
    if st is None:
        return torch.zeros(env.num_envs, device=env.device)
    phi = lambda g: g.clamp(lo_mm, hi_mm)
    dead = env.termination_manager.terminated.float()
    return torch.nan_to_num((phi(st["gap"]) - phi(st["gap_prev"]) - dead*phi(st["gap"]))/env.step_dt, nan=0.0)


def undock_critic_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged (critic only): docked or not, the magnet's pull (N/8) and the gap (mm/20, clipped), so the value can
    tell a 1 N episode from an 8 N one."""
    st = getattr(env, "_undock", None)
    if st is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    act = (st["active"] & st["set"]).float()
    return torch.stack([act, st["fmax"]*act/8.0, (st["gap"].clamp(-5.0, 20.0)/20.0)*act], dim=-1)

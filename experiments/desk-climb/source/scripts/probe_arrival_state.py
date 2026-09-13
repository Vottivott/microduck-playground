"""Arrival-state study: the state when the highest supported foot first reaches
tread 5, for natural climbs (floor spawns) vs static approach spawns at k=4."""
import sys, math, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.robot import ladder as L
ck = sys.argv[1]; task = "Mjlab-StaircaseLanding-MicroDuck"

def build(spawn):
    cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 256; cfg.terminations.clear()
    cfg.commands["twist"].ranges.lin_vel_x = (0.03, 0.03)
    p = cfg.events["reset_stair_ladder"].params
    p["fixed_level"] = 0; p["level_mix_prob"] = 0.0; p["approach_swing_prob"] = 0.0
    if spawn == "floor":
        p["floor_spawn_prob"] = 1.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 0.0
    else:
        p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 1.0; p["max_start_tread"] = 7; p["approach_max_below"] = 3
    agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
    return raw, env, runner.get_inference_policy(device="cuda:0")

def snapshot(raw, ids):
    robot = raw.scene["robot"]; st = raw._stair; o = raw.scene.env_origins
    head_id = robot.body_names.index("yaw_roll_motion")
    root = robot.data.root_link_pos_w[ids] - o[ids]
    head = robot.data.body_link_pos_w[ids, head_id] - o[ids]
    v = robot.data.root_link_lin_vel_b[ids]
    g = robot.data.projected_gravity_b[ids]
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    yaw = m._yaw_from_quat(robot.data.root_link_quat_w[ids])
    fyaw = st.flight_yaw[ids, 0] if hasattr(st, "flight_yaw") else torch.zeros_like(yaw)
    dyaw = torch.rad2deg(torch.atan2(torch.sin(yaw - fyaw), torch.cos(yaw - fyaw)))
    c = m._stair_contacts(raw)
    ft = c["foot_tread"][ids]
    return {
        "vx_b": v[:, 0], "vz_b": v[:, 2], "pitch_deg": pitch, "dyaw_deg": dyaw,
        "root_z": root[:, 2], "head_dx": head[:, 0] - root[:, 0], "head_dz": head[:, 2] - root[:, 2],
        "root_y": root[:, 1], "left_on5": (ft[:, 0] == 5).float(), "other_tread": ft.min(dim=1).values.float(),
    }

def summarize(tag, snaps):
    keys = snaps[0].keys()
    for k in keys:
        x = torch.cat([s[k] for s in snaps])
        x = x[torch.isfinite(x)]
        print("%s %-12s n=%3d mean %+.3f std %.3f p10 %+.3f p90 %+.3f" % (tag, k, len(x), x.mean(), x.std(), x.quantile(0.1), x.quantile(0.9)))

for spawn in ("floor", "approach"):
    raw, env, policy = build(spawn); obs = env.get_observations()
    n = raw.num_envs; seen = torch.zeros(n, dtype=torch.bool, device="cuda:0"); snaps = []
    outcome_over = torch.zeros(n, dtype=torch.bool, device="cuda:0")
    with torch.inference_mode():
        for step in range(500):
            obs, *_ = env.step(policy(obs))
            c = m._stair_contacts(raw); best = c["foot_tread"].max(dim=1).values
            new = (best == 5) & ~seen
            if spawn == "approach" and step == 5:
                new = ~seen  # static stance: snapshot shortly after spawn
            ids = new.nonzero().flatten()
            if len(ids):
                snaps.append({k: v.detach().clone() for k, v in snapshot(raw, ids).items()}); seen |= new
            if step == 150 and spawn == "approach":
                break
    print("==", spawn, "snapshots", int(seen.sum()))
    if snaps: summarize(spawn, snaps)

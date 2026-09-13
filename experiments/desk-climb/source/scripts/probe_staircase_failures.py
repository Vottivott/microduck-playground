"""Headless corner-staircase probe: where and why do floor-start climbs end?

usage: MICRODUCK_STAIRCASE_DEMO=1 uv run python scripts/probe_staircase_failures.py <ckpt> [seconds] [num_envs] [task]
Per finished episode: highest tread reached (foot contact), the termination
term(s) that fired.  Prints a histogram of highest tread by termination cause.
"""
import sys, collections, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m

ck = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
num_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 256
task = sys.argv[4] if len(sys.argv) > 4 else "Mjlab-StaircaseDemo-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = num_envs
p = cfg.events["reset_stair_ladder"].params
import os
top_only = os.getenv("STAIR_PROBE_TOP", "0") == "1"
p["floor_spawn_prob"] = 0.0 if top_only else 1.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
for k in ("landing_spawn_prob", "landing_approach_prob", "top_spawn_prob"):
    if k in p: p[k] = 0.0
if top_only:
    p["top_spawn_prob"] = 1.0
cfg.commands["twist"].ranges.lin_vel_x = (0.03, 0.03)
cfg.episode_length_s = seconds
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
steps = int(round(seconds / raw.step_dt))
best = torch.full((num_envs,), -1, dtype=torch.long, device="cuda:0")
hist = collections.defaultdict(collections.Counter)
tm = raw.termination_manager
for step in range(steps):
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
    c = m._stair_contacts(raw)
    tread = c["foot_tread"].max(dim=1).values
    best = torch.maximum(best, tread)
    if bool(dones.any()):
        ids = torch.nonzero(dones).squeeze(1)
        flags = {name: tm.get_term(name) for name in tm.active_terms}
        for i in ids.tolist():
            cause = "+".join(n for n, f in flags.items() if bool(f[i])) or "none"
            hist[cause][int(best[i])] += 1
        best[ids] = -1
print("STAIR_PROBE steps", steps, "episodes", sum(sum(v.values()) for v in hist.values()))
for cause, cnt in sorted(hist.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(cnt.values()); top = sorted(cnt.items())
    print(f"  {cause:32s} n={n:4d}  highest-tread histogram: {top}")

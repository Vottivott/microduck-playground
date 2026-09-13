"""Landing-step study: spawn two treads below the first landing and measure
the step onto it under controlled variants.

    uv run python scripts/probe_landing_study.py <checkpoint> [--target-up M] [--speed V]
        [--setback M] [--level N] [--envs N] [--seconds S]

Prints one STUDY line: upright fraction after the run, the fraction whose
right foot reached the landing (tread 7), whose trunk reached the landing
level, head/landing contacts, and where the falls happen (median tread of
the last support).
"""
import argparse, collections, math
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--target-up", type=float, default=None, help="landing foot target above the landing top (m)")
ap.add_argument("--setback", type=float, default=None, help="landing setback (m)")
ap.add_argument("--target-ahead", type=float, default=None, help="landing foot target past the nose (m)")
ap.add_argument("--speed", type=float, default=0.04, help="climb command (m/s)")
ap.add_argument("--level", type=int, default=0)
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--spawn", default="approach", choices=("approach", "floor", "ladder"))
ap.add_argument("--approach-swing", type=float, default=0.7, help="fraction of approach spawns placed mid-swing")
a = ap.parse_args()
task = "Mjlab-Staircase-MicroDuck"
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = a.envs
cfg.terminations.clear()
cmd = cfg.commands["twist"]
cmd.ranges.lin_vel_x = (a.speed, a.speed)
p = cfg.events["reset_stair_ladder"].params
from dataclasses import replace as _replace

g = p["geometry"]  # frozen dataclass: replace, then hand the copy to the reset
changes = {}
if a.target_up is not None:
    changes["landing_target_up_m"] = a.target_up
if a.setback is not None:
    changes["landing_setback_m"] = a.setback
if a.target_ahead is not None:
    changes["landing_target_ahead_m"] = a.target_ahead
if changes:
    g = _replace(g, **changes)
    p["geometry"] = g
p["swing_spawn_prob"] = 0.0
p["fixed_level"] = a.level
p["level_mix_prob"] = 0.0
p["path_spawn_frac"] = 0.0
p["approach_swing_prob"] = a.approach_swing
if a.spawn == "approach":
    p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 1.0; p["max_start_tread"] = 7  # the landing index must be allowed or the approach forcing is skipped
elif a.spawn == "floor":
    p["floor_spawn_prob"] = 1.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 0.0
else:
    p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 0.0; p["max_start_tread"] = 2
agent = load_rl_cfg(task)
raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0")
runner.load(a.ckpt, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
obs = env.get_observations()
robot = raw.scene["robot"]; st = raw._stair; o = raw.scene.env_origins
n = a.envs
fell_step = torch.full((n,), -1, dtype=torch.long, device="cuda:0")
last_support = torch.full((n,), -1, dtype=torch.long, device="cuda:0")
reached_landing = torch.zeros(n, dtype=torch.bool, device="cuda:0")
trunk_over = torch.zeros(n, dtype=torch.bool, device="cuda:0")
head_hits = torch.zeros(n, dtype=torch.bool, device="cuda:0")
tilt_run = torch.zeros(n, dtype=torch.long, device="cuda:0")
t_over = torch.full((n,), -1, dtype=torch.long, device="cuda:0")
landing_top = st.tread_top[:, 7]
steps = int(a.seconds * 50)
for step in range(steps):
    with torch.inference_mode():
        obs, *_ = env.step(policy(obs))
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    # a fall = tilted past 60 deg for 0.3 s (the landing step itself is a
    # deep but recoverable forward lean), or the trunk down near the feet
    tilted = tilt > math.radians(60)
    tilt_run = torch.where(tilted, tilt_run + 1, torch.zeros_like(tilt_run)) if step > 0 else tilted.long()
    low = robot.data.root_link_pos_w[:, 2] - o[:, 2] < 0.06
    down = (tilt_run >= 15) | low
    newly = down & (fell_step < 0)
    fell_step[newly] = step
    c = m._stair_contacts(raw)
    sup = c["foot_tread"]
    has = sup.max(dim=1).values
    last_support = torch.where((fell_step < 0) & (has >= 0), has, last_support)
    reached_landing |= (sup == 7).any(dim=1) & ~down
    over_now = (robot.data.root_link_pos_w[:, 2] > landing_top + 0.10) & ~down
    t_over = torch.where(over_now & (t_over < 0), torch.full_like(t_over, step), t_over)
    trunk_over |= over_now
    if "head_touch" in c:
        head_hits |= c["head_touch"] & (fell_step < 0)
tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
up = tilt < math.radians(45)
fell = fell_step >= 0
t_fall = (fell_step[fell].float() / 50).median().item() if bool(fell.any()) else float("nan")
supports = collections.Counter(last_support[fell].tolist()).most_common(4)
end_support = collections.Counter(m._stair_contacts(raw)["foot_tread"][up].max(dim=1).values.tolist()).most_common(4)
t_over_p50 = (t_over[t_over >= 0].float() / 50).median().item() if bool((t_over >= 0).any()) else float("nan")
print(
    f"STUDY spawn={a.spawn} swing={a.approach_swing:.1f} level={a.level} target_up={g.landing_target_up_m:.3f} ahead={g.landing_target_ahead_m:.3f} setback={g.landing_setback_m:.3f} speed={a.speed:.2f} "
    f"upright={up.float().mean():.2f} fell={fell.float().mean():.2f} t_fall_p50={t_fall:.2f}s "
    f"reached_landing={reached_landing.float().mean():.2f} trunk_over={trunk_over.float().mean():.2f} "
    f"head_hits={head_hits.float().mean():.2f} t_over_p50={t_over_p50:.2f}s last_support_before_fall={supports} upright_end_support={end_support}"
)

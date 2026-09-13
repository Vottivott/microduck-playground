"""Render a floor-start climb of the left staircase with a checkpoint.

usage: uv run python scripts/render_staircase_demo.py <ckpt> <out_dir> [seconds] [azimuth] [effects 0/1] [task]
720p, no command arrow, shadow fix; sparks + dead motors on impact by default.
"""
import sys, torch
from dataclasses import asdict
from pathlib import Path
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner
from mjlab_microduck.video_effects import CrashEffects, configure_video_cfg, fix_render_shadows
import mjlab_microduck.tasks  # noqa

ck, out = sys.argv[1], sys.argv[2]
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
azimuth = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
effects_on = (sys.argv[5] if len(sys.argv) > 5 else "1") == "1"
task = sys.argv[6] if len(sys.argv) > 6 else "Mjlab-StaircaseLanding-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 1
p = cfg.events["reset_stair_ladder"].params
p["floor_spawn_prob"] = 1.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
for k in ("landing_spawn_prob", "landing_approach_prob", "top_spawn_prob"):
    if k in p: p[k] = 0.0
if "Demo" in task:
    cfg.viewer.distance = 0.9
cfg.commands["twist"].ranges.lin_vel_x = (0.03, 0.03)
cfg.episode_length_s = seconds + 1.0
configure_video_cfg(cfg); cfg.viewer.distance = 0.6; cfg.viewer.elevation = -12.0; cfg.viewer.azimuth = azimuth
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
fix_render_shadows(raw, light_dir=(-0.5, 0.2, -1.0))
effects = CrashEffects(raw, entity_names=("robot",)) if effects_on else None
steps = int(round(seconds / raw.step_dt))
env = VideoRecorder(raw, video_folder=Path(out), step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
st = raw._stair
from mjlab_microduck.tasks import mdp as m
best = -1
for step in range(steps):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
    if effects is not None: effects.step()
    c = m._stair_contacts(raw); best = max(best, int(c["foot_tread"][0].max()))
print("STAIR_VIDEO", out, "highest tread", best)

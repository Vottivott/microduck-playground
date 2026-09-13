import sys, math, torch, imageio, numpy as np
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
ck, out = sys.argv[1], sys.argv[2]; task = "Mjlab-Staircase-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 2; cfg.terminations.clear()
p = cfg.events["reset_stair_ladder"].params; p["floor_spawn_prob"] = 1.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
configure_video_cfg(cfg); cfg.viewer.distance = 0.55; cfg.viewer.elevation = -10.0; cfg.viewer.azimuth = 120.0
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
fix_render_shadows(raw, light_dir=(-0.5, 0.2, -1.0))
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
robot = raw.scene["robot"]; st = raw._stair; sites = m._stair_foot_sites(raw, robot)
frames = []
for step in range(450):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
    if step % 25 == 0:
        frames.append(raw.render())
        fz = robot.data.site_pos_w[0, sites, 2] - raw.scene.env_origins[0, 2]
        c = m._stair_contacts(raw)
        print("t=%.1f feet z %.3f %.3f support %s last %s body_touch %s" % (step/50, fz[0], fz[1], c["foot_support"][0].tolist(), st.foot_last_tread[0].tolist(), bool(c["body_touch"][0])))
tiles = [np.asarray(imageio.core.asarray(f)) for f in frames[:18]]
import PIL.Image as I
sheet = I.new("RGB", (640*3, 360*6))
for k, f in enumerate(tiles): sheet.paste(I.fromarray(f).resize((640, 360)), ((k%3)*640, (k//3)*360))
sheet.save(out); print("SHEET", out)

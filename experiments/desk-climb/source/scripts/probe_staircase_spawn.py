"""Settle test for the staircase task: spawn types held with HOME ctrl for 1 s;
report fall fractions per spawn type and dump one frame per type."""
import math, sys, torch, numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows
import mjlab_microduck.tasks  # noqa
out_dir = sys.argv[1]
cfg = load_env_cfg("Mjlab-Staircase-MicroDuck", play=True)
cfg.scene.num_envs = 256; cfg.terminations.clear(); cfg.viewer.distance = 1.2; cfg.viewer.elevation = -25.0; cfg.viewer.azimuth = 135.0
p = cfg.events["reset_stair_ladder"].params
p["floor_spawn_prob"] = 0.25; p["landing_spawn_prob"] = 0.34; p["swing_spawn_prob"] = 0.0; p["fixed_level"] = 2
configure_video_cfg(cfg)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
fix_render_shadows(env)
env.reset()
st = env._stair
robot = env.scene["robot"]
z0 = robot.data.root_link_pos_w[:, 2].clone()
frames = {}
import imageio
kinds = {"floor": st.spawn_on_floor.clone(), "landing_home": st.spawn_on_landing.clone(), "landing_stag": (st.start_tread == 7) & ~st.spawn_on_landing}
kinds["onto_landing"] = (st.start_tread == 6); kinds["stance"] = ~(kinds["floor"] | kinds["landing_home"] | kinds["landing_stag"] | kinds["onto_landing"])
print("counts", {k: int(v.sum()) for k, v in kinds.items()}, "flight1 stance", int((st.start_tread >= 8).sum()))
for step in range(50):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0"))
    if step in (0, 49):
        frames[step] = env.render()
tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
dz = robot.data.root_link_pos_w[:, 2] - z0
for k, m in kinds.items():
    if m.any():
        print("SETTLE %-8s n=%3d tilt>20deg=%.2f dropped>3cm=%.2f" % (k, int(m.sum()), float((tilt[m] > math.radians(20)).float().mean()), float((dz[m] < -0.03).float().mean())))
for step, f in frames.items():
    imageio.imwrite(f"{out_dir}/staircase_settle_{step}.png", f)
print("FRAMES", list(frames))

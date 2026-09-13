import math, sys, torch, imageio
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.video_effects import configure_video_cfg, fix_render_shadows
import mjlab_microduck.tasks  # noqa
out = sys.argv[1]
for tag, dist, elev, azim in (("close", 1.0, -20.0, 150.0), ("wide", 4.5, -35.0, 200.0)):
    cfg = load_env_cfg("Mjlab-StaircaseDemo-MicroDuck", play=True)
    cfg.scene.num_envs = 4; cfg.terminations.clear()
    configure_video_cfg(cfg); cfg.viewer.distance = dist; cfg.viewer.elevation = elev; cfg.viewer.azimuth = azim
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    fix_render_shadows(env, light_dir=(-0.4, 0.3, -1.0))
    env.reset()
    st = env._stair; robot = env.scene["robot"]
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    for step in range(50):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0"))
    f = env.render(); imageio.imwrite(f"{out}/demo_{tag}.png", f)
    dz = robot.data.root_link_pos_w[:, 2] - z0
    print("PROBE", tag, "drop", [round(v, 3) for v in dz.tolist()], "riser %.4f angle %.1f" % (st.riser[0], math.degrees(st.angle[0])), "tread127 top %.3f" % (st.tread_top[0, -1] - env.scene.env_origins[0, 2]))
    env.close()

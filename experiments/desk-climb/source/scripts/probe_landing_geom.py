import math, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
cfg = load_env_cfg("Mjlab-Staircase-MicroDuck", play=True)
cfg.scene.num_envs = 32; cfg.terminations.clear()
p = cfg.events["reset_stair_ladder"].params
p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 1.0; p["swing_spawn_prob"] = 0.0; p["fixed_level"] = 2; import os
if os.getenv("NOISE","0")=="0": p["position_noise"] = 0.0; p["yaw_noise_deg"] = 0.0; p["tilt_noise_deg"] = 0.0; p["joint_noise"] = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
st = env._stair; robot = env.scene["robot"]; o = env.scene.env_origins
root = robot.data.root_link_pos_w - o
sites = m._stair_foot_sites(env, robot); foot = robot.data.site_pos_w[:, sites, :] - o[:, None, :]
stag = ((st.start_tread == 7) & ~st.spawn_on_landing).nonzero().flatten()
print("staggered envs", stag.tolist()[:8])
i = int(stag[0]) if len(stag) else 0
print("start", int(st.start_tread[i]), "on_landing", bool(st.spawn_on_landing[i]), "riser %.4f angle %.1f flat %.3f x0 %.3f" % (st.riser[i], math.degrees(st.angle[i]), st.flat[i], st.x0[i]))
print("root", root[i].tolist(), "feet", foot[i].tolist())
jp = robot.data.joint_pos[i] - robot.data.default_joint_pos[i]; print("joint offsets", [round(v,3) for v in jp.tolist()])
for k in (6, 7, 8, 9, 15):
    c = st.tread_centre[i, k] - o[i]; print("tread", k, "centre", [round(v, 4) for v in c.tolist()], "top %.4f nose %.4f" % (st.tread_top[i, k] - o[i, 2], st.tread_nose_x[i, k] - o[i, 0]))
mocap = env.scene["tread_07"].data.root_link_pos_w[i] - o[i]; print("tread_07 mocap pos", [round(v, 4) for v in mocap.tolist()])
z_start = robot.data.root_link_pos_w[:, 2].clone()
for s in range(50):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0"))
root2 = robot.data.root_link_pos_w - o; foot2 = robot.data.site_pos_w[:, sites, :] - o[:, None, :]
print("after 0.5s root", root2[i].tolist(), "feet", foot2[i].tolist())
c = m._stair_contacts(env); print("contacts foot_support", c["foot_support"][i].tolist(), "foot_tread", c["foot_tread"][i].tolist(), "floor", c["foot_on_floor"][i].tolist())

drop = z_start - robot.data.root_link_pos_w[:, 2]
for e in stag.tolist()[:10]:
    print("ENV %d flat %.3f drop %.3f feet_z %s tread8top %.3f" % (e, st.flat[e], drop[e], [round(v,3) for v in foot2[e,:,2].tolist()], st.tread_top[e, 8]-o[e,2]))

import sys, math, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
ck = sys.argv[1]; task = "Mjlab-StaircaseLanding-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 64; cfg.terminations.clear()
cfg.commands["twist"].ranges.lin_vel_x = (0.03, 0.03)
p = cfg.events["reset_stair_ladder"].params
p["fixed_level"] = 0; p["level_mix_prob"] = 0.0; p["approach_swing_prob"] = 0.0
p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 1.0; p["max_start_tread"] = 7
for k in ("position_noise", "yaw_noise_deg", "tilt_noise_deg", "joint_noise"):
    if k in p: p[k] = 0.0
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
robot = raw.scene["robot"]; st = raw._stair; o = raw.scene.env_origins; sites = m._stair_foot_sites(raw, robot)
head_id = robot.body_names.index("yaw_roll_motion")
print("head body", robot.body_names[head_id])
i4 = (st.start_tread == 4).nonzero().flatten(); i5 = (st.start_tread == 5).nonzero().flatten()
print("envs k=4", len(i4), "k=5", len(i5))
i = int(i4[0]) if len(i4) else 0
for k in range(3, 9):
    c = st.tread_centre[i, k] - o[i]; print("tread %d centre %s top %.4f nose_x %.4f" % (k, [round(v, 4) for v in c.tolist()], st.tread_top[i, k] - o[i, 2], st.tread_nose_x[i, k] - o[i, 0]))
def snap(tag):
    root = robot.data.root_link_pos_w[i] - o[i]; head = robot.data.body_link_pos_w[i, head_id] - o[i]
    feet = robot.data.site_pos_w[i, sites, :] - o[i]
    g = robot.data.projected_gravity_b[i]; pitch = math.degrees(math.atan2(g[0], -g[2]))
    c = m._stair_contacts(raw)
    print("%s root x %.3f z %.3f | head x %.3f z %.3f | feet x %.3f/%.3f z %.3f/%.3f | tread %s | pitch %.0f | head_touch %s lean %s" % (
        tag, root[0], root[2], head[0], head[2], feet[0,0], feet[1,0], feet[0,2], feet[1,2], c["foot_tread"][i].tolist(), pitch, bool(c["head_touch"][i]), bool(c["lean_touch"][i])))
snap("t=0.00")
for step in range(1, 151):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
    if step % 10 == 0: snap("t=%.2f" % (step / 50))
# outcome over the k=4 and k=5 groups
c = m._stair_contacts(raw); tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
for name, ids in (("k=4", i4), ("k=5", i5)):
    if len(ids) == 0: continue
    up = (tilt[ids] < math.radians(60)).float().mean().item(); best = c["foot_tread"][ids].max(dim=1).values
    print(name, "upright %.2f" % up, "highest tread hist", torch.bincount(best.clamp_min(0), minlength=9).tolist())

import sys, math, torch, mujoco, collections
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
ck = sys.argv[1]; task = "Mjlab-Staircase-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 64; cfg.terminations.clear()
p = cfg.events["reset_stair_ladder"].params; p["floor_spawn_prob"] = 1.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
model = raw.sim.mj_model
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
landing = [g for g, n in enumerate(names) if n.startswith("tread_07/")]
hist = collections.Counter(); steps_touch = 0
for step in range(400):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
    if step < 100: continue
    count = int(raw.sim.data.nacon[0].item())
    pairs = raw.sim.data.contact.geom[:count].cpu(); dist = raw.sim.data.contact.dist[:count].cpu()
    for (g0, g1), d in zip(pairs.tolist(), dist.tolist()):
        if d > 0: continue
        if g0 in landing and names[g1].startswith("robot/"): hist[names[g1]] += 1
        elif g1 in landing and names[g0].startswith("robot/"): hist[names[g0]] += 1
print("ROBOT GEOMS TOUCHING THE LANDING BOX (steps 100-400, 64 envs):")
for n, c in hist.most_common(12): print("  %-40s %d" % (n, c))

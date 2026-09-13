"""Where does the policy stall on the two-flight staircase? Floor spawns, 12 s."""
import sys, math, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
ck = sys.argv[1]; task = "Mjlab-Staircase-MicroDuck"
MODE = sys.argv[2] if len(sys.argv) > 2 else "normal"
if MODE == "mask_landing":
    _orig = m._stair_foot_target_info
    def _patched(env, asset, min_rise=0.006):
        info = _orig(env, asset, min_rise)
        g = env._stair.geometry
        landing = torch.tensor([g.is_landing(i) for i in range(g.num_treads)] + [False], device=env.device)
        is_l = landing[info["index"].clamp_max(g.num_treads)]
        info = dict(info); info["valid"] = info["valid"] & ~is_l
        return info
    m._stair_foot_target_info = _patched
elif MODE == "fake_regular":
    _orig = m._stair_foot_target_info
    def _patched(env, asset, min_rise=0.006):
        info = _orig(env, asset, min_rise)
        g = env._stair.geometry; st = env._stair
        landing = torch.tensor([g.is_landing(i) for i in range(g.num_treads)] + [False], device=env.device)
        is_l = landing[info["index"].clamp_max(g.num_treads)]
        # pretend the landing is a regular tread: target = nose - 18mm at top+3mm, i.e. shift vec back by setback and down by target_up
        vec = info["vec"].clone()
        shift = torch.tensor([g.landing_setback_m + 0.5 * g.tread_depth_m, 0.0, g.landing_target_up_m - 0.003], device=env.device)
        vec = torch.where(is_l[:, :, None], vec - shift, vec)
        info = dict(info); info["vec"] = vec
        return info
    m._stair_foot_target_info = _patched
print("MODE", MODE)
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 128; cfg.terminations.clear()
p = cfg.events["reset_stair_ladder"].params; p["floor_spawn_prob"] = 1.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
robot = raw.scene["robot"]; st = raw._stair
for step in range(600):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
up = tilt < math.radians(45)
flt = st.foot_last_tread
info = m._stair_foot_target_info(raw, robot)
print("upright fraction %.2f" % up.float().mean())
import collections
print("foot_last_tread (upright envs) left:", sorted(collections.Counter(flt[up, 0].tolist()).items()))
print("foot_last_tread (upright envs) right:", sorted(collections.Counter(flt[up, 1].tolist()).items()))
print("target index left:", sorted(collections.Counter(info["index"][up, 0].tolist()).items()))
print("target index right:", sorted(collections.Counter(info["index"][up, 1].tolist()).items()))
vec = info["vec"][up]; print("target vec fwd/up mean per foot:", vec[:, :, 0].mean(0).tolist(), vec[:, :, 2].mean(0).tolist(), "lateral", vec[:, :, 1].abs().mean(0).tolist())
print("flat p50 %.3f lateral %.3f dyaw %.1f" % (st.flat.median(), st.lateral.abs().median(), math.degrees(st.dyaw.abs().median())))

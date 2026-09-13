"""Inspect the top-landing spawn on the corner staircase: where does the duck land?

MICRODUCK_STAIRCASE_DEMO=1 uv run python scripts/probe_staircase_top_spawn.py
"""
import sys, torch
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.robot import ladder as L

cfg = load_env_cfg("Mjlab-StaircaseDemo-MicroDuck", play=True); cfg.scene.num_envs = 4
p = cfg.events["reset_stair_ladder"].params
p["floor_spawn_prob"] = 0.0; p["fixed_level"] = 0; p["level_mix_prob"] = 0.0
for k in ("landing_spawn_prob", "landing_approach_prob"):
    if k in p: p[k] = 0.0
p["top_spawn_prob"] = 1.0
cfg.events.pop("push_robot", None)
raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
ck = sys.argv[1] if len(sys.argv) > 1 else None
policy = None
if ck:
    agent = load_rl_cfg("Mjlab-StaircaseDemo-MicroDuck")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = (load_runner_cls("Mjlab-StaircaseDemo-MicroDuck") or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
else:
    raw.reset()
st = raw._stair
robot = raw.scene["robot"]
act = torch.zeros(raw.num_envs, raw.action_manager.total_action_dim, device="cuda:0")
for step in range(16):
    root = robot.data.root_link_pos_w[0]
    feet = robot.data.site_pos_w[0, :, 2]
    c = m._stair_contacts(raw)
    top127 = st.tread_top[0, 127]; centre = st.tread_centre[0, 127]
    tf = raw.scene[L.TOP_FLOOR_ENTITY].data.root_link_pos_w[0] if L.TOP_FLOOR_ENTITY in raw.scene.entities else None
    ft = robot.data.site_pos_w[0, m._stair_foot_sites(raw, robot), 2]
    print(f"step {step}: trunk-lowfoot {float(root[2]-ft.min()):.3f} root z {root[2]:.3f} xy ({root[0]:.3f},{root[1]:.3f})  tread127 top {top127:.3f} centre ({centre[0]:.3f},{centre[1]:.3f},{centre[2]:.3f})  top_floor {None if tf is None else [round(float(v),3) for v in tf]}  foot_tread {c['foot_tread'][0].tolist()} support {c['foot_support'][0].tolist()} on_floor {c['foot_on_floor'][0].tolist()}  spawn_on_top {bool(st.spawn_on_top[0])}")
    terms = raw.termination_manager
    fired = [k for k in terms.active_terms if bool(terms.get_term(k)[0])]
    if fired: print("   terminated:", fired)
    if policy is not None:
        with torch.inference_mode():
            obs, *_ = env.step(policy(obs))
    else:
        raw.step(act)
print("TOPSPAWN_PROBE_DONE")

"""Trace the step onto the landing: stance spawn with the lower (right) foot on
tread 5 and the higher (left) foot on tread 6; log the right foot relative to
the landing nose, trunk pitch, supports, and the outcome after 3 s."""
import sys, math, torch, collections
from dataclasses import asdict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa
from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.robot import ladder as L
ck = sys.argv[1]; task = "Mjlab-Staircase-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = 128; cfg.terminations.clear()
p = cfg.events["reset_stair_ladder"].params
p["floor_spawn_prob"] = 0.0; p["landing_spawn_prob"] = 0.0; p["landing_approach_prob"] = 1.0; p["swing_spawn_prob"] = 0.0
p["fixed_level"] = 0; p["level_mix_prob"] = 0.0; p["max_start_tread"] = 7; p["approach_swing_prob"] = 0.0; p["path_spawn_frac"] = 0.0
agent = load_rl_cfg(task); raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0"); runner.load(ck, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0"); obs = env.get_observations()
robot = raw.scene["robot"]; st = raw._stair; sites = m._stair_foot_sites(raw, robot); o = raw.scene.env_origins
print("start treads", collections.Counter(st.start_tread.tolist()))
base = st.flight_base[:, 0]; yaw = st.flight_yaw[:, 0]
nose7 = st.tread_nose_x[:, 7]  # u of the landing nose
def foot_u(slot):
    fp = robot.data.site_pos_w[:, sites[slot], :2]
    u, v = L.to_flight_frame(fp, base, yaw); return u
maxlift = torch.zeros(128, device="cuda:0"); max_fwd = torch.full((128,), -1.0, device="cuda:0")
for step in range(150):
    with torch.inference_mode(): obs, *_ = env.step(policy(obs))
    rz = robot.data.site_pos_w[:, sites[1], 2] - o[:, 2]
    ru = foot_u(1) - nose7
    maxlift = torch.maximum(maxlift, rz); max_fwd = torch.maximum(max_fwd, ru)
    if step in (10, 25, 50, 75, 100, 149):
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        c = m._stair_contacts(raw)
        print("t=%.1f right foot: z p50 %.3f (landing top %.3f) u-nose p50 %+.3f | tilt p50 %.0f deg | right support %.2f on tread %s | left last %s" % (
            step/50, rz.median(), (st.tread_top[:, 7]-o[:, 2]).median(), ru.median(), math.degrees(tilt.median()),
            c["foot_support"][:, 1].float().mean(), collections.Counter(c["foot_tread"][:, 1].tolist()).most_common(3), collections.Counter(st.foot_last_tread[:, 0].tolist()).most_common(3)))
tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
up = tilt < math.radians(45)
print("OUTCOME upright %.2f | right foot reached landing (last tread 7) %.2f | max lift p50 %.3f p90 %.3f | max u-nose p50 %+.3f" % (
    up.float().mean(), (st.foot_last_tread[:, 1] == 7).float().mean(), maxlift.median(), maxlift.quantile(0.9), max_fwd.median()))

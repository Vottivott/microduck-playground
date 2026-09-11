"""Headless battery for a basketball checkpoint: survival and command tracking.

  MICRODUCK_BB_CMD_SCALE=2 uv run python scripts/eval_basketball_checkpoint.py <ckpt> [num_envs] [seconds] [out.json]
Play cfg (pushes on), fresh spawns; reports the fraction of envs that never
fell over the horizon, the mean time to the first fall, and the mean absolute
forward-velocity tracking error of the root while standing.
"""
import json, os, sys
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.basketball_distillation import load_actor, make_distillation_env_cfg
import mjlab_microduck.tasks  # noqa

ck = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 512; secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
out = sys.argv[4] if len(sys.argv) > 4 else None
task = "Mjlab-Basketball-MicroDuck"
checkpoint = torch.load(ck, map_location="cpu", weights_only=False)
cfg = (make_distillation_env_cfg(play=True, command_scale=float(os.getenv("MICRODUCK_BB_CMD_SCALE", "1")))
       if "student_state_dict" in checkpoint else load_env_cfg(task, play=True))
cfg.scene.num_envs = n; cfg.episode_length_s = secs + 1.0
raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
agent = load_rl_cfg(task); env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
obs = env.get_observations()
policy = load_actor(checkpoint, obs, device="cuda:0")
steps = int(round(secs / raw.step_dt))
fell_step = torch.full((n,), -1, dtype=torch.long, device="cuda:0")
err_sum = torch.zeros(n, device="cuda:0"); err_cnt = torch.zeros(n, device="cuda:0")
for i in range(steps):
    with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)
    fell = raw.termination_manager.get_term("fell")
    fell_step = torch.where((fell_step < 0) & fell, torch.full_like(fell_step, i), fell_step)
    alive = fell_step < 0
    cmd = raw.command_manager.get_command("twist")[:, 0]
    v = raw.scene["robot"].data.root_link_lin_vel_b[:, 0]
    err_sum += torch.where(alive, (cmd - v).abs(), torch.zeros_like(v)); err_cnt += alive.float()
surv = float((fell_step < 0).float().mean())
ttf = fell_step[fell_step >= 0].float().mean() * raw.step_dt if bool((fell_step >= 0).any()) else None
res = {"checkpoint": ck, "num_envs": n, "horizon_s": secs, "survival_fraction": surv,
       "mean_time_to_first_fall_s": (float(ttf) if ttf is not None else None),
       "mean_abs_forward_tracking_error_mps": float((err_sum / err_cnt.clamp_min(1)).mean())}
print("BB_EVAL", json.dumps(res))
if out:
    json.dump(res, open(out, "w"), indent=1)
env.close()

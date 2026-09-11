"""Seeded long-horizon basketball evaluation with pre-reset failure diagnostics.

Uses the same play config as eval_basketball_checkpoint.py. The termination wrapper
only records state; it leaves physics, resets, commands and termination decisions alone.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.basketball_distillation import load_actor
from mjlab_microduck.tasks import mdp
import mjlab_microduck.tasks  # noqa: F401


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('out')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--command-scale', type=float, default=2)
    p.add_argument('--blind', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg
    cfg = make_microduck_basketball_env_cfg(play=True, blind=a.blind, history=1)
    cfg.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    cfg.episode_length_s = a.seconds + 1
    cfg.commands['twist'].ranges.lin_vel_x = (-.15*a.command_scale, .15*a.command_scale)
    cfg.commands['twist'].ranges.lin_vel_y = (-.1*a.command_scale, .1*a.command_scale)
    cfg.commands['twist'].ranges.ang_vel_z = (-.5*a.command_scale, .5*a.command_scale)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    snapshot = {}
    original = cfg.terminations['fell'].func

    def record_fall(env, **kwargs):
        robot, ball = env.scene['robot'].data, env.scene['ball'].data
        rel = robot.root_link_pos_w - ball.root_link_pos_w
        tilt = mdp.wrestle_tilt(env, 'robot')
        offset = rel[:, :2].norm(dim=1)
        delta = env.action_manager.action - env.action_manager.prev_action
        snapshot['values'] = torch.cat((
            env.command_manager.get_command('twist').clone(),
            robot.root_link_lin_vel_b[:, :2].clone(),
            robot.root_link_ang_vel_b[:, 2:3].clone(),
            tilt[:, None], offset[:, None], rel[:, 2:3],
            ball.root_link_lin_vel_w[:, :2].norm(dim=1)[:, None],
            delta.square().mean(dim=1).sqrt()[:, None],
        ), dim=1)
        snapshot['causes'] = torch.stack((
            rel[:, 2] < kwargs['ball_radius'] + kwargs['min_height'],
            offset > kwargs['max_offset'],
            tilt > torch.deg2rad(torch.tensor(kwargs['fall_tilt_deg'], device=env.device)),
        ), dim=1)
        return original(env, **kwargs)

    cfg.terminations['fell'].func = record_fall
    raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=load_rl_cfg('Mjlab-Basketball-MicroDuck').clip_actions)
    policy = load_actor(checkpoint, env.get_observations(), device='cuda:0')
    # Model construction consumes RNG differently for MLP and LSTM. Reseed reset
    # after loading so this does not choose the subsequent disturbance sequence.
    raw.reset(seed=a.seed + 10000)
    obs = env.get_observations()
    assert obs['actor'].shape == (a.num_envs, 61)
    if a.blind:
        assert torch.count_nonzero(obs['actor'][:, 55:]) == 0
    n, dt = a.num_envs, raw.step_dt
    first = torch.full((n,), -1, device='cuda:0', dtype=torch.long)
    first_fell = first.clone()
    sums = torch.zeros((n, 8), device='cuda:0')
    counts = torch.zeros(n, device='cuda:0')
    terminal = torch.zeros((n, 11), device='cuda:0')
    causes = torch.zeros((n, 3), device='cuda:0', dtype=torch.bool)
    failure_terms = {}
    horizons = {}
    for i in range(round(a.seconds / dt)):
        alive = first < 0
        with torch.inference_mode():
            obs, _, done, _ = env.step(policy(obs))
            policy.reset(done)
        vals = snapshot['values']
        fell = raw.termination_manager.get_term('fell')
        first_fell[(first_fell < 0) & fell] = i
        ended = alive & done.bool()
        first[ended] = i
        terminal[ended] = vals[ended]
        causes[ended] = snapshot['causes'][ended]
        for name in raw.termination_manager.active_terms:
            term = raw.termination_manager.get_term(name)
            failure_terms[name] = failure_terms.get(name, 0) + (ended & term).sum()
        # All errors are in the robot frame and use the command that drove this step.
        metrics = torch.cat(((vals[:, :3] - vals[:, 3:6]).abs(), vals[:, 6:11]), dim=1)
        valid = alive & ~done.bool() & torch.isfinite(metrics).all(dim=1)
        sums += torch.where(valid[:, None], metrics, 0.)
        counts += valid.float()
        t = (i+1)*dt
        if any(abs(t-h)<dt/2 for h in [10, 20, 30, 60, 120]):
            horizons[str(round(t))] = {'survived': int((first<0).sum()), 'fraction': float((first<0).float().mean())}
            print('BB_LONG_PROGRESS', a.seed, round(t), horizons[str(round(t))], flush=True)
    labels = ['forward_mae_mps', 'lateral_mae_mps', 'yaw_mae_radps', 'tilt_rad',
              'offset_m', 'root_above_ball_m', 'ball_speed_mps', 'action_delta_rms']
    means = (sums.sum(dim=0) / counts.sum().clamp_min(1)).tolist()
    by_env = (sums / counts.clamp_min(1)[:, None]).mean(dim=0).tolist()
    first_cpu, vals_cpu, causes_cpu = first.cpu().tolist(), terminal.cpu().tolist(), causes.cpu().tolist()
    rows = [{'env': j, 'time_s': (step+1)*dt, 'command': vals_cpu[j][:3],
             'actual_twist': vals_cpu[j][3:6], 'tilt_rad': vals_cpu[j][6],
             'offset_m': vals_cpu[j][7], 'root_above_ball_m': vals_cpu[j][8],
             'ball_speed_mps': vals_cpu[j][9], 'action_delta_rms': vals_cpu[j][10],
             'causes': dict(zip(['low_height','offset','tilt'], causes_cpu[j]))}
            for j, step in enumerate(first_cpu) if step >= 0]
    result = dict(checkpoint=a.checkpoint, sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  seed=a.seed, reset_seed=a.seed+10000, blind=a.blind, command_scale=a.command_scale,
                  num_envs=n, horizon_s=a.seconds, survival=horizons,
                  push_velocity_range=cfg.events['push_robot'].params['velocity_range'],
                  push_interval_s=cfg.events['push_robot'].interval_range_s,
                  survival_fraction=float((first<0).float().mean()),
                  fell_only_survival_fraction=float((first_fell<0).float().mean()),
                  first_termination_counts={k:int(v) for k,v in failure_terms.items()},
                  alive_time_weighted_metrics=dict(zip(labels,means)),
                  per_environment_mean_metrics=dict(zip(labels,by_env)), failures=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print('BB_LONG_DONE', a.out, result['survival_fraction'], flush=True)
    env.close()


if __name__ == '__main__':
    main()

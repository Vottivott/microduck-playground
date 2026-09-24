"""Verified-state continuation of the public climb checkpoint, not exact replay."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--wall-friction', type=float, nargs=2)
    p.add_argument('--seed', type=int, default=91)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be a new directory; original checkpoints are never overwritten.')
    if a.num_envs < 1 or a.iterations < 1:
        p.error('Environment/iteration counts must be positive.')
    recipe = json.loads((Path(__file__).resolve().parents[1] / 'experiments/chimney-climb/resume.json').read_text())
    if sha256(a.bank) != recipe['bank_sha256']:
        p.error('Bank hash does not match the released 645-state bank.')
    # Avoid silently inheriting a rendering or unrelated experiment configuration.
    inherited = [k for k in os.environ if k.startswith('MICRODUCK_')]
    if inherited:
        p.error('Unset inherited MICRODUCK_* settings before using this pinned recipe: ' + ', '.join(inherited))
    settings = dict(recipe['environment'])
    if a.wall_friction:
        lo, hi = a.wall_friction
        if not 0 < lo <= hi:
            p.error('Friction must satisfy 0 < lower <= upper.')
        settings['MICRODUCK_CH_WALL_FRIC'] = f'{lo},{hi}'
    settings['MICRODUCK_CH_HANDOVER_BANK'] = str(a.bank.resolve())
    os.environ.update(settings)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.tasks.chimney_mdp import _handover_bank, _geom_ids

    bank = _handover_bank(str(a.bank.resolve()), 'cpu')
    for key, shape in [('qpos', (645, 21)), ('qvel', (645, 20)), ('origins', (645, 3))]:
        assert bank is not None and tuple(bank[key].shape) == shape
        assert torch.isfinite(bank[key]).all()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter', 'infos'):
        assert key in checkpoint, key
    task = 'Mjlab-Chimney-MicroDuck'
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    agent = load_rl_cfg(task)
    agent.seed = a.seed
    agent.logger = 'tensorboard'
    a.output.mkdir(parents=True)
    dump_yaml(a.output / 'params/env.yaml', asdict(cfg))
    dump_yaml(a.output / 'params/agent.yaml', asdict(agent))
    raw = ManagerBasedRlEnv(cfg, device='cuda:0')
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    friction = {}
    lo, hi = map(float, settings['MICRODUCK_CH_WALL_FRIC'].split(','))
    for wall in ('left', 'right'):
        values = raw.sim.model.geom_friction[:, _geom_ids(raw)[wall].long(), 0]
        assert torch.isfinite(values).all()
        assert float(values.min()) >= lo - 1e-6 and float(values.max()) <= hi + 1e-6
        friction[wall] = {'min': float(values.min()), 'max': float(values.max()),
                          'unique': int(values.unique().numel())}
    local_q = raw.sim.data.qpos.clone()
    local_q[:, :3] -= raw.scene.terrain.env_origins
    bank_q = bank['qpos'].clone().to(local_q.device)
    bank_q[:, :3] -= bank['origins'].to(local_q.device)
    matches = ((local_q[:, None] - bank_q[None]).abs().amax(dim=-1) < 1e-5).any(dim=1)
    initial_bank_states = int(matches.sum())
    runner = load_runner_cls(task)(env, asdict(agent), str(a.output), device='cuda:0')
    runner.load(str(a.checkpoint), map_location='cuda:0')
    # rsl_rl initializes the writer in learn(); this pre-training snapshot is earlier.
    if not hasattr(runner.logger, 'writer'):
        runner.logger.writer = None
    runner.save(str(a.output / 'resume_initial.pt'))
    initial = torch.load(a.output / 'resume_initial.pt', map_location='cpu', weights_only=False)

    def same(x, y):
        if torch.is_tensor(x):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                same(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for u, v in zip(x, y):
                same(u, v)
        else:
            assert x == y

    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter'):
        same(checkpoint[key], initial[key])
    same(checkpoint['infos']['env_state'], initial['infos']['env_state'])
    obs = env.get_observations()
    assert obs['actor'].shape == (a.num_envs, 61)
    assert raw.action_manager.total_action_dim == 14
    runner.current_learning_iteration = int(checkpoint['iter']) + 1
    runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
    assert torch.isfinite(raw.sim.data.qpos).all()
    assert torch.isfinite(raw.sim.data.qvel).all()
    runner.save(str(a.output / 'continued.pt'))
    report = dict(recipe=recipe, effective_environment=settings, seed=a.seed,
                  checkpoint_sha256=sha256(a.checkpoint), bank_sha256=sha256(a.bank),
                  original_release_checkpoint=sha256(a.checkpoint) == recipe['checkpoint_sha256'],
                  exact_initial_actor_critic_normalizer_optimizer_counter=True,
                  num_envs=a.num_envs, iterations=a.iterations, actor_obs=61, actions=14,
                  finite_final_state=True, final_counter=raw.common_step_counter,
                  sampled_wall_friction=friction, initial_bank_states=initial_bank_states,
                  restored_counter=checkpoint['infos']['env_state']['common_step_counter'],
                  note='Fresh simulator episodes/RNG; reconstructed recipe, not exact historical replay.')
    (a.output / 'resume-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    raw.close()
    print('CHIMNEY_RESUME_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

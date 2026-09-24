"""Resume a trusted released PPO checkpoint with an explicit continuation recipe.

This restores learning state, not the old simulator/RNG state. Run in a fresh
process with no inherited MICRODUCK_* overrides. Recipes document provenance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_recipe(path, profile):
    recipe = json.loads(Path(path).read_text())['profiles'][profile]
    assert recipe['task'].startswith('Mjlab-')
    assert len(recipe['checkpoint_sha256']) == 64
    assert recipe['provenance'] and recipe['limitations']
    assert all(k.startswith('MICRODUCK_') and isinstance(v, str)
               for k, v in recipe['environment'].items())
    return recipe


def same(x, y):
    import torch
    if torch.is_tensor(x):
        assert torch.equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            same(x[key], y[key])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            same(u, v)
    else:
        assert x == y


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--iterations', type=int, default=5, help='Additional updates, not total iteration target')
    p.add_argument('--seed', type=int)
    p.add_argument('--bank', type=Path, help='Optional recovered real-arrival bank for exit only; explicitly changes the reset distribution')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--descendant', action='store_true', help='Allow a new checkpoint from this same task/recipe')
    a = p.parse_args()
    recipe = read_recipe(a.recipe, a.profile)
    if a.output.exists() or a.num_envs < 1 or a.iterations < 1:
        p.error('Require a new output directory and positive environment/iteration counts.')
    inherited = [k for k in os.environ if k.startswith('MICRODUCK_')]
    if inherited:
        p.error('Unset inherited MICRODUCK_* settings: ' + ', '.join(inherited))
    checkpoint_hash = sha256(a.checkpoint)
    if not a.descendant and checkpoint_hash != recipe['checkpoint_sha256']:
        p.error('Wrong release checkpoint; --descendant is only for a later checkpoint of this same task.')
    os.environ.update(recipe['environment'])
    if a.bank:
        if a.profile != 'exit' or sha256(a.bank) != recipe.get('optional_bank_sha256'):
            p.error('Bank is only supported for exit and must match the published bank hash.')
        os.environ['MICRODUCK_EX_ARRIVAL_BANK'] = str(a.bank.resolve())
        os.environ['MICRODUCK_EX_ARRIVAL_PROB'] = '0.45'
    os.environ.setdefault('WANDB_MODE', 'disabled')
    seed = recipe['seed'] if a.seed is None else a.seed
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    import mjlab_microduck.tasks  # noqa: F401

    configure_torch_backends()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    if a.bank:
        bank = torch.load(a.bank, map_location='cpu', weights_only=True)
        for key, shape in [('qpos', (18, 21)), ('qvel', (18, 20)), ('origins', (18, 3))]:
            assert tuple(bank[key].shape) == shape and torch.isfinite(bank[key]).all()
        assert float(bank['platform_top']) == 3.0
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter', 'infos'):
        assert key in checkpoint, key
    assert checkpoint['optimizer_state_dict']['state'], 'Expected original populated optimizer state'
    cfg = load_env_cfg(recipe['task'])
    cfg.scene.num_envs = a.num_envs
    cfg.seed = seed
    agent = load_rl_cfg(recipe['task'])
    agent.seed = seed
    agent.logger = 'tensorboard'
    for key, value in recipe.get('algorithm', {}).items():
        assert hasattr(agent.algorithm, key), key
        setattr(agent.algorithm, key, value)
    a.output.mkdir(parents=True)
    (a.output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    dump_yaml(a.output / 'params/env.yaml', asdict(cfg))
    dump_yaml(a.output / 'params/agent.yaml', asdict(agent))
    raw = ManagerBasedRlEnv(cfg, device=a.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    try:
        runner = load_runner_cls(recipe['task'])(env, asdict(agent), str(a.output), device=a.device)
        runner.load(str(a.checkpoint), map_location=a.device)
        # Keep the adaptive scheduler's Python LR in sync with restored Adam.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
        if not hasattr(runner.logger, 'writer'):
            runner.logger.writer = None
        runner.save(str(a.output / 'resume_initial.pt'))
        initial = torch.load(a.output / 'resume_initial.pt', map_location='cpu', weights_only=True)
        for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter'):
            same(checkpoint[key], initial[key])
        same(checkpoint['infos']['env_state'], initial['infos']['env_state'])
        counter = checkpoint['infos']['env_state']['common_step_counter']
        # Apply restored-counter curriculum stages before collecting any PPO data.
        raw.reset(seed=seed)
        assert raw.common_step_counter == counter
        bank_matches = None
        if a.bank:
            from mjlab_microduck.robot import exit_stage
            q = raw.sim.data.qpos.detach().clone()
            q[:, :3] -= raw.scene.env_origins
            bq = bank['qpos'].to(q.device).clone()
            bq[:, :3] -= bank['origins'].to(q.device)
            bq[:, 2] += exit_stage.PLATFORM_TOP_M - float(bank['platform_top'])
            bank_matches = int(((q[:, None] - bq[None]).abs().amax(-1) < 1e-5).any(1).sum())
            if a.num_envs >= 64:
                assert bank_matches > 0, 'Optional bank was not applied to any reset'
        obs = env.get_observations()
        assert obs['actor'].shape == (a.num_envs, 61)
        assert raw.action_manager.total_action_dim == 14
        event_params = {name: raw.event_manager.get_term_cfg(name).params
                        for name in cfg.events if cfg.events[name] is not None}
        dump_yaml(a.output / 'params/events-after-restore.yaml', event_params)
        reward_weights = {name: raw.reward_manager.get_term_cfg(name).weight
                          for name in cfg.rewards if cfg.rewards[name] is not None}
        (a.output / 'params/rewards-after-restore.json').write_text(json.dumps(reward_weights, indent=2)+'\n')
        original_step = env.step
        checked_steps = 0

        def checked_step(actions):
            nonlocal checked_steps
            assert torch.isfinite(actions).all(), 'Nonfinite actions'
            result = original_step(actions)
            assert all(torch.isfinite(v).all() for v in result[0].values()), 'Nonfinite observations'
            assert torch.isfinite(result[1]).all(), 'Nonfinite rewards'
            assert torch.isfinite(raw.sim.data.qpos).all() and torch.isfinite(raw.sim.data.qvel).all()
            checked_steps += 1
            return result

        env.step = checked_step
        runner.current_learning_iteration = int(checkpoint['iter']) + 1
        runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
        runner.save(str(a.output / 'continued.pt'))
        assert raw.common_step_counter == counter + a.iterations * agent.num_steps_per_env
        assert checked_steps == a.iterations * agent.num_steps_per_env
        for group in (runner.alg.actor.state_dict(), runner.alg.critic.state_dict()):
            assert all(torch.isfinite(v).all() for v in group.values())
        # Compare float32 CPU inference to ONNX CPU, avoiding CUDA TF32 batch
        # rounding (training intentionally uses the configured CUDA backend).
        actor = runner.get_inference_policy(device='cpu')
        term = raw.action_manager.get_term('joint_pos')
        lo, hi = term.raw_action_bounds()
        lo, hi = lo.cpu(), hi.cpu()
        # Save random, zero and real post-training observations with expected actions.
        obs = env.get_observations().to('cpu')
        batches = [obs['actor'].detach().clone(), torch.zeros_like(obs['actor']),
                   torch.randn_like(obs['actor'])]
        inputs, expected = [], []
        with torch.inference_mode():
            for batch in batches:
                sample = obs.clone(); sample['actor'] = batch
                action = actor(sample)
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                inputs.append(batch.cpu().numpy())
                expected.append(torch.clamp(action, lo, hi).cpu().numpy())
        np.savez_compressed(a.output/'parity-inputs.npz', obs=np.concatenate(inputs), actions=np.concatenate(expected))
        report = dict(profile=a.profile, recipe_sha256=sha256(a.recipe), checkpoint_sha256=checkpoint_hash,
                      original_release_checkpoint=checkpoint_hash == recipe['checkpoint_sha256'],
                      optional_bank_sha256=sha256(a.bank) if a.bank else None,
                      initial_bank_matches=bank_matches,
                      effective_environment={k: v for k, v in os.environ.items() if k.startswith('MICRODUCK_')},
                      exact_actor_critic_normalizer_optimizer_counter_restore=True,
                      restored_counter=counter, final_counter=raw.common_step_counter,
                      additional_iterations=a.iterations, checked_steps=checked_steps,
                      num_envs=a.num_envs, seed=seed, actor_obs=61, actions=14,
                      finite_all_training_steps=True, restored_learning_rate=initial['optimizer_state_dict']['param_groups'][0]['lr'],
                      continuation_checkpoint_sha256=sha256(a.output/'continued.pt'),
                      note='Fresh simulator/RNG and reverse-curriculum episodes; see recipe provenance. Not bit-exact historical replay.')
    finally:
        raw.close()
    # The official exporter bakes observation normalization AND action bounds.
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root/'scripts/export.py'), recipe['task'],
                    '--checkpoint-file', str((a.output/'continued.pt').resolve()),
                    '--onnx-file', str((a.output/'policy.onnx').resolve()),
                    '--num-envs', '1', '--device', a.device], cwd=root, check=True)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(a.output/'policy.onnx'))
    session = ort.InferenceSession(str(a.output/'policy.onnx'), providers=['CPUExecutionProvider'])
    arrays = np.load(a.output/'parity-inputs.npz')
    actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: x[None]})[0] for x in arrays['obs']])
    error = float(np.abs(actual - arrays['actions']).max())
    assert actual.shape == arrays['actions'].shape and np.isfinite(actual).all()
    assert error < 1e-4, error
    report.update(normalized_export=True, max_abs_onnx_error=error, parity_observations=len(actual))
    (a.output/'resume-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('RELEASE_RESUME_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

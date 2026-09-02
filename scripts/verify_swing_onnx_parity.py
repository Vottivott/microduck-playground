"""Verify checkpoint and clipped ONNX actions along one swing rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--duration", type=float, default=36.0)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.atol <= 0.0:
        parser.error("--duration and --atol must be positive")

    model = onnx.load(args.onnx)
    clips = [node for node in model.graph.node if node.op_type == "Clip"]
    if not any(node.name == "action_clip" for node in clips):
        raise RuntimeError("ONNX graph has no baked action_clip node")

    configure_torch_backends()
    task_id = "Mjlab-SwingPump-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 1.0
    env_cfg.terminations.clear()
    agent_cfg = load_rl_cfg(task_id)
    if agent_cfg.clip_actions is None:
        raise RuntimeError("Swing task has no configured action clipping")

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint), map_location=args.device)
        checkpoint_policy = runner.get_inference_policy(device=args.device)
        session = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        obs = env.get_observations()
        max_abs_error = 0.0
        max_abs_onnx_action = 0.0
        steps = round(args.duration / raw.step_dt)
        for _ in range(steps):
            with torch.inference_mode():
                raw_action = checkpoint_policy(obs)
            actor_obs = obs if isinstance(obs, torch.Tensor) else obs["actor"]
            expected = torch.clamp(
                raw_action, -agent_cfg.clip_actions, agent_cfg.clip_actions
            ).detach().cpu().numpy()
            actual = session.run(
                None, {input_name: actor_obs.detach().cpu().numpy()}
            )[0]
            max_abs_error = max(
                max_abs_error, float(np.max(np.abs(expected - actual)))
            )
            max_abs_onnx_action = max(
                max_abs_onnx_action, float(np.max(np.abs(actual)))
            )
            if not np.allclose(expected, actual, rtol=0.0, atol=args.atol):
                raise RuntimeError(
                    f"Checkpoint/ONNX action mismatch: {max_abs_error:.6g} > "
                    f"{args.atol:.6g}"
                )
            obs, _, _, _ = env.step(raw_action)
    finally:
        raw.close()

    print(
        f"PASS steps={steps} seed={args.seed} max_abs_error={max_abs_error:.6g} "
        f"max_abs_onnx_action={max_abs_onnx_action:.6g} "
        f"clip_actions={agent_cfg.clip_actions:g}"
    )


if __name__ == "__main__":
    main()

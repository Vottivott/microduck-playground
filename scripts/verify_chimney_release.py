"""Check exported actors against trusted checkpoints and live task observations.

Usage: python scripts/verify_chimney_release.py PACKAGE OUTPUT.json
Loads pickle checkpoints: only use trusted release artifacts.
"""
import json
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
import mjlab_microduck.tasks  # noqa: F401

root, output = map(Path, sys.argv[1:3])
results = {}
for role, task, folder in (("climb", "Chimney", root), ("enter", "Approach", root / "enter"), ("exit", "Exit", root / "exit")):
    task_id = f"Mjlab-{task}-MicroDuck"
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 29
    raw = ManagerBasedRlEnv(cfg, device="cuda:0")
    agent = load_rl_cfg(task_id)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task_id)(env, asdict(agent), device="cuda:0")
    runner.load(str(folder / "checkpoint.pt"), map_location="cuda:0")
    actor = runner.get_inference_policy(device="cuda:0")
    term = raw.action_manager.get_term("joint_pos")
    lo, hi = term.raw_action_bounds()
    sess = ort.InferenceSession(str(folder / "policy.onnx"), providers=["CPUExecutionProvider"])
    onnx.checker.check_model(onnx.load(folder / "policy.onnx"))
    obs = env.get_observations()
    max_error = 0.
    rng = np.random.default_rng(29)
    with torch.inference_mode():
        for i in range(200):
            inp = obs.clone()
            if i < 100:
                x = np.zeros((1, 61), np.float32) if i == 0 else rng.normal(size=(1, 61)).astype(np.float32)
                inp["actor"] = torch.as_tensor(x, device="cuda:0")
            else:
                x = inp["actor"].cpu().numpy()
            expected = actor(inp)
            if agent.clip_actions is not None:
                expected = expected.clamp(-agent.clip_actions, agent.clip_actions)
            expected = torch.clamp(expected, lo, hi)
            actual = sess.run(None, {sess.get_inputs()[0].name: x})[0]
            assert actual.shape == (1, 14) and np.isfinite(actual).all()
            max_error = max(max_error, float(np.abs(actual - expected.cpu().numpy()).max()))
            if i >= 100:
                obs, _, _, _ = env.step(torch.as_tensor(actual, device="cuda:0"))
                assert torch.isfinite(raw.sim.data.qpos).all()
    assert max_error < 1e-4, (role, max_error)
    offset = term._offset[0].cpu().tolist()
    results[role] = {"task": task_id, "max_abs_action_error": max_error,
        "test_observations": 200, "live_steps": 100, "normalizer_baked": True,
        "joint_offset_rad": offset, "action_scale": float(term._scale),
        "action_bounds_lo": lo.cpu().tolist(), "action_bounds_hi": hi.cpu().tolist(),
        "joint_names": list(raw.scene["robot"].joint_names),
        "input_name": sess.get_inputs()[0].name, "output_name": sess.get_outputs()[0].name}
    raw.close()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, indent=2) + "\n")
print("RELEASE_PARITY_OK", results)

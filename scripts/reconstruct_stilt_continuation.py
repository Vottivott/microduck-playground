"""Build an actor-exact RSL-RL continuation checkpoint from a stilt ONNX.

The original stilt PPO snapshots were not retained, while the selected ONNX
actors were. This utility recovers every actor and observation-normalizer
tensor exactly, then combines them with a fresh compatible critic scaffold and
an empty optimizer state. The result is suitable for conservative continuation
training, but is deliberately not described as the original training state.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn.functional as F


NORMALIZER_EPS = 1.0e-2
ACTOR_KEYS = (
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
    "mlp.4.weight",
    "mlp.4.bias",
    "mlp.6.weight",
    "mlp.6.bias",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_initializers(path: Path) -> dict[str, torch.Tensor]:
    model = onnx.load(path)
    return {
        initializer.name: torch.from_numpy(
            np.array(numpy_helper.to_array(initializer), copy=True)
        )
        for initializer in model.graph.initializer
    }


def actor_mean(checkpoint: dict, observations: torch.Tensor) -> torch.Tensor:
    state = checkpoint["actor_state_dict"]
    value = (observations - state["obs_normalizer._mean"]) / (
        state["obs_normalizer._std"] + NORMALIZER_EPS
    )
    for index in (0, 2, 4):
        value = F.elu(
            F.linear(value, state[f"mlp.{index}.weight"], state[f"mlp.{index}.bias"])
        )
    return F.linear(value, state["mlp.6.weight"], state["mlp.6.bias"])


def verify_actor(checkpoint: dict, onnx_path: Path) -> float:
    rng = np.random.default_rng(417)
    observations = rng.normal(size=(32, 61)).astype(np.float32)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    expected = np.concatenate(
        [session.run(None, {input_name: row[None, :]})[0] for row in observations],
        axis=0,
    )
    with torch.inference_mode():
        actual = actor_mean(checkpoint, torch.from_numpy(observations)).numpy()
    error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=2.0e-5, atol=2.0e-5):
        raise RuntimeError(f"reconstructed actor differs from ONNX (max error {error})")
    return error


def build_checkpoint(
    onnx_path: Path,
    template_path: Path,
    iteration: int,
    learning_rate: float,
    exploration_std: float,
    num_steps_per_env: int,
    training_num_envs: int,
) -> dict:
    checkpoint = torch.load(template_path, map_location="cpu", weights_only=False)
    initializers = onnx_initializers(onnx_path)
    actor = checkpoint["actor_state_dict"]

    for key in ACTOR_KEYS:
        source = initializers[key].to(dtype=actor[key].dtype)
        if source.shape != actor[key].shape:
            raise ValueError(f"shape mismatch for {key}: {source.shape} != {actor[key].shape}")
        actor[key] = source

    actor["obs_normalizer._mean"] = initializers["obs_normalizer._mean"].to(
        dtype=actor["obs_normalizer._mean"].dtype
    )
    effective_divisor = initializers["onnx::Div_24"].to(
        dtype=actor["obs_normalizer._std"].dtype
    )
    std = effective_divisor - NORMALIZER_EPS
    if torch.any(std <= 0.0):
        raise ValueError("ONNX normalizer divisor is not greater than epsilon")
    actor["obs_normalizer._std"] = std
    actor["obs_normalizer._var"] = std.square()
    actor["obs_normalizer.count"] = torch.tensor(
        iteration * num_steps_per_env * training_num_envs, dtype=torch.long
    )
    actor["distribution.std_param"].fill_(exploration_std)

    optimizer = checkpoint["optimizer_state_dict"]
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate

    checkpoint["iter"] = iteration
    checkpoint["infos"] = {
        "env_state": {"common_step_counter": iteration * num_steps_per_env},
        "release_continuation": {
            "kind": "actor-exact-warm-start",
            "actor_source": onnx_path.name,
            "actor_source_sha256": sha256(onnx_path),
            "critic": "fresh-compatible-scaffold",
            "optimizer_moments_reset": True,
            "learning_rate": learning_rate,
            "exploration_std": exploration_std,
            "normalizer_epsilon": NORMALIZER_EPS,
            "normalizer_count_reconstructed": True,
            "training_num_envs_assumption": training_num_envs,
            "num_steps_per_env": num_steps_per_env,
        },
    }
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--exploration-std", type=float, default=0.1)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--training-num-envs", type=int, default=2048)
    args = parser.parse_args()
    if args.iteration <= 0:
        parser.error("--iteration must be positive")
    if args.learning_rate <= 0.0 or args.exploration_std <= 0.0:
        parser.error("learning rate and exploration std must be positive")

    checkpoint = build_checkpoint(
        args.onnx,
        args.template,
        args.iteration,
        args.learning_rate,
        args.exploration_std,
        args.num_steps_per_env,
        args.training_num_envs,
    )
    max_error = verify_actor(checkpoint, args.onnx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"WROTE {args.output} sha256={sha256(args.output)} "
        f"onnx_actor_max_abs_error={max_error:.3g}"
    )


if __name__ == "__main__":
    main()

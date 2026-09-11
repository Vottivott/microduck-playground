"""Prepare an existing RSL-RL checkpoint for conservative fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--exploration-std", type=float, default=0.35)
    parser.add_argument(
        "--selected-exploration-std",
        type=float,
        default=None,
        help="Optional std for only --exploration-std-indices.",
    )
    parser.add_argument(
        "--exploration-std-indices",
        type=int,
        nargs="*",
        default=(),
        help="Action dimensions receiving --selected-exploration-std.",
    )
    parser.add_argument("--keep-moments", action="store_true")
    parser.add_argument(
        "--reset-actor-observation-indices",
        type=int,
        nargs="*",
        default=(),
        help="Normalizer columns whose semantics changed; reset mean/variance.",
    )
    parser.add_argument(
        "--reset-actor-observation-std",
        type=float,
        default=1.0,
        help="Fixed initial std used for reset observation columns.",
    )
    parser.add_argument(
        "--zero-actor-input-columns",
        type=int,
        nargs="*",
        default=(),
        help="First actor-layer input columns to zero for exact-policy continuity.",
    )
    parser.add_argument(
        "--reset-critic-observation-indices",
        type=int,
        nargs="*",
        default=(),
        help="Critic normalizer columns whose semantics changed.",
    )
    parser.add_argument(
        "--reset-critic-observation-std",
        type=float,
        default=1.0,
        help="Fixed initial std used for reset critic observation columns.",
    )
    parser.add_argument(
        "--zero-critic-input-columns",
        type=int,
        nargs="*",
        default=(),
        help="First critic-layer input columns to zero for value continuity.",
    )
    args = parser.parse_args()

    if not 0.0 < args.exploration_std <= 1.0:
        parser.error("--exploration-std must be in (0, 1]")
    if args.selected_exploration_std is not None and not (
        0.0 < args.selected_exploration_std <= 1.0
    ):
        parser.error("--selected-exploration-std must be in (0, 1]")
    if args.exploration_std_indices and args.selected_exploration_std is None:
        parser.error(
            "--exploration-std-indices requires --selected-exploration-std"
        )
    if args.reset_actor_observation_std <= 0.0:
        parser.error("--reset-actor-observation-std must be positive")
    if args.reset_critic_observation_std <= 0.0:
        parser.error("--reset-critic-observation-std must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    std_param = checkpoint["actor_state_dict"]["distribution.std_param"]
    old_std_range = [float(std_param.min()), float(std_param.max())]
    std_param.fill_(args.exploration_std)
    selected_std_indices = sorted(set(args.exploration_std_indices))
    if any(index < 0 or index >= std_param.shape[-1] for index in selected_std_indices):
        parser.error("--exploration-std-indices contains an out-of-range action")
    if selected_std_indices:
        std_param[..., selected_std_indices] = args.selected_exploration_std

    reset_indices = sorted(set(args.reset_actor_observation_indices))
    actor_state = checkpoint["actor_state_dict"]
    normalizer_mean = actor_state["obs_normalizer._mean"]
    if any(index < 0 or index >= normalizer_mean.shape[-1] for index in reset_indices):
        parser.error(
            "--reset-actor-observation-indices contains an out-of-range column"
        )
    for key, value in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", args.reset_actor_observation_std**2),
        ("obs_normalizer._std", args.reset_actor_observation_std),
    ):
        actor_state[key][..., reset_indices] = value

    zero_columns = sorted(set(args.zero_actor_input_columns))
    first_layer = actor_state["mlp.0.weight"]
    if any(index < 0 or index >= first_layer.shape[1] for index in zero_columns):
        parser.error("--zero-actor-input-columns contains an out-of-range column")
    first_layer[:, zero_columns] = 0.0

    reset_critic_indices = sorted(set(args.reset_critic_observation_indices))
    critic_state = checkpoint["critic_state_dict"]
    critic_normalizer_mean = critic_state["obs_normalizer._mean"]
    if any(
        index < 0 or index >= critic_normalizer_mean.shape[-1]
        for index in reset_critic_indices
    ):
        parser.error(
            "--reset-critic-observation-indices contains an out-of-range column"
        )
    for key, value in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", args.reset_critic_observation_std**2),
        ("obs_normalizer._std", args.reset_critic_observation_std),
    ):
        critic_state[key][..., reset_critic_indices] = value

    zero_critic_columns = sorted(set(args.zero_critic_input_columns))
    critic_first_layer = critic_state["mlp.0.weight"]
    if any(
        index < 0 or index >= critic_first_layer.shape[1]
        for index in zero_critic_columns
    ):
        parser.error("--zero-critic-input-columns contains an out-of-range column")
    critic_first_layer[:, zero_critic_columns] = 0.0

    optimizer = checkpoint["optimizer_state_dict"]
    old_rates = [group.get("lr") for group in optimizer["param_groups"]]
    for group in optimizer["param_groups"]:
        group["lr"] = args.learning_rate
        group["initial_lr"] = args.learning_rate
    if not args.keep_moments:
        optimizer["state"] = {}

    checkpoint.setdefault("infos", {})["conservative_finetune"] = {
        "source": str(args.checkpoint),
        "old_learning_rates": old_rates,
        "learning_rate": args.learning_rate,
        "old_exploration_std_range": old_std_range,
        "exploration_std": args.exploration_std,
        "selected_exploration_std": args.selected_exploration_std,
        "exploration_std_indices": selected_std_indices,
        "optimizer_moments_reset": not args.keep_moments,
        "reset_actor_observation_indices": reset_indices,
        "reset_actor_observation_std": args.reset_actor_observation_std,
        "zero_actor_input_columns": zero_columns,
        "reset_critic_observation_indices": reset_critic_indices,
        "reset_critic_observation_std": args.reset_critic_observation_std,
        "zero_critic_input_columns": zero_critic_columns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"WROTE {args.output} old_lr={old_rates} new_lr={args.learning_rate} "
        f"old_std={old_std_range} new_std={args.exploration_std} "
        f"selected_std={args.selected_exploration_std}@{selected_std_indices} "
        f"moments_reset={not args.keep_moments} "
        f"reset_actor_obs={reset_indices}@{args.reset_actor_observation_std} "
        f"zero_actor_inputs={zero_columns} "
        f"reset_critic_obs={reset_critic_indices}@{args.reset_critic_observation_std} "
        f"zero_critic_inputs={zero_critic_columns} sha256={digest}"
    )


if __name__ == "__main__":
    main()

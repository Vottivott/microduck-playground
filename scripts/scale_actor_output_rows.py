"""Scale selected final actor-output rows in an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--action-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    weight = actor["mlp.6.weight"]
    bias = actor["mlp.6.bias"]
    indices = sorted(set(args.action_indices))
    if any(index < 0 or index >= weight.shape[0] for index in indices):
        parser.error("--action-indices contains an out-of-range action")

    weight[indices] *= args.factor
    bias[indices] *= args.factor
    checkpoint.setdefault("infos", {})["actor_output_row_scaling"] = {
        "source": str(args.checkpoint),
        "factor": args.factor,
        "action_indices": indices,
        "critic_and_optimizer": "copied_from_source",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"WROTE {args.output} factor={args.factor} rows={indices} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()

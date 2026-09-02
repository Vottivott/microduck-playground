"""Interpolate selected actor output rows between two RSL-RL checkpoints.

This is intentionally strict: every actor tensor other than the selected rows
of the final linear layer must be bitwise identical.  The critic and optimizer
are copied from the source checkpoint because the resulting file is primarily
an inference/evaluation artifact; use ``retune_checkpoint_optimizer.py`` before
resuming training from it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


FINAL_WEIGHT = "mlp.6.weight"
FINAL_BIAS = "mlp.6.bias"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--action-indices", type=int, nargs="+", required=True)
    parser.add_argument(
        "--allow-distribution-std-difference",
        action="store_true",
        help=(
            "Allow target exploration std to differ. The output still copies "
            "the source std; only selected deterministic actor-mean rows blend."
        ),
    )
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    target = torch.load(args.target, map_location="cpu", weights_only=False)
    source_actor = source["actor_state_dict"]
    target_actor = target["actor_state_dict"]
    if source_actor.keys() != target_actor.keys():
        parser.error("actor state-dict keys differ")

    weight = source_actor[FINAL_WEIGHT]
    indices = sorted(set(args.action_indices))
    if any(index < 0 or index >= weight.shape[0] for index in indices):
        parser.error("--action-indices contains an out-of-range action")
    selected = torch.zeros(weight.shape[0], dtype=torch.bool)
    selected[indices] = True

    for key, source_value in source_actor.items():
        target_value = target_actor[key]
        if source_value.shape != target_value.shape:
            parser.error(f"shape differs for actor tensor {key}")
        if key == FINAL_WEIGHT:
            if not torch.equal(source_value[~selected], target_value[~selected]):
                parser.error("unselected final-weight rows differ")
        elif key == FINAL_BIAS:
            if not torch.equal(source_value[~selected], target_value[~selected]):
                parser.error("unselected final-bias rows differ")
        elif (
            key == "distribution.std_param"
            and args.allow_distribution_std_difference
        ):
            continue
        elif not torch.equal(source_value, target_value):
            parser.error(f"frozen actor tensor differs: {key}")

    output = copy.deepcopy(source)
    output_actor = output["actor_state_dict"]
    for key in (FINAL_WEIGHT, FINAL_BIAS):
        source_value = source_actor[key]
        target_value = target_actor[key]
        output_actor[key][selected] = source_value[selected] + args.alpha * (
            target_value[selected] - source_value[selected]
        )
    output.setdefault("infos", {})["actor_output_interpolation"] = {
        "source": str(args.source),
        "target": str(args.target),
        "alpha": args.alpha,
        "action_indices": indices,
        "target_distribution_std_ignored": bool(
            args.allow_distribution_std_difference
        ),
        "critic_and_optimizer": "copied_from_source",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"WROTE {args.output} alpha={args.alpha} rows={indices} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()

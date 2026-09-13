"""Turn a locomotion checkpoint into a fresh-iteration ladder initializer.

Copies the actor/critic/normalizer weights unchanged, resets the iteration
and environment step counters so ladder curricula start from stage 0, and
clears Adam moments (the loss landscape is a different task).

    uv run python scripts/reinit_ladder_checkpoint.py <in.pt> <out.pt>
"""

import sys

import torch


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print("loaded keys:", sorted(ckpt.keys()))
    print("actor obs width:", ckpt["actor_state_dict"]["mlp.0.weight"].shape)
    ckpt["iter"] = 0
    env_state = ckpt.get("infos", {}).get("env_state", {}) if isinstance(ckpt.get("infos"), dict) else {}
    if "common_step_counter" in env_state:
        env_state["common_step_counter"] = 0
    opt = ckpt.get("optimizer_state_dict")
    if isinstance(opt, dict):
        for state in opt.get("state", {}).values():
            for key, value in state.items():
                if torch.is_tensor(value) and key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step"}:
                    value.zero_()
    torch.save(ckpt, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()

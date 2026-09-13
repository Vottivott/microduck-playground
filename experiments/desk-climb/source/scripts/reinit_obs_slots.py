"""Neutralize selected observation inputs in an rsl_rl checkpoint.

For each given actor-observation index: the running normalizer gets mean 0
and std 1, and the first MLP layer's weights reading that input are zeroed
(actor and critic; critic indices are the same because the critic
observation starts with the same 61 values).  Used when an observation slot
changes meaning between runs (e.g. a gait clock replacing a near-constant
padding), so a resumed policy is unaffected until it learns the new signal.

    uv run python scripts/reinit_obs_slots.py <in.pt> <out.pt> --indices 49 50
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    for name in ("actor_state_dict", "critic_state_dict"):
        state = ckpt[name]
        weight = state["mlp.0.weight"]
        for index in args.indices:
            if not 0 <= index < weight.shape[1]:
                raise ValueError(f"index {index} outside {name} input width {weight.shape[1]}")
            weight[:, index].zero_()
            for key in list(state):
                if key.endswith("obs_normalizer._mean"):
                    state[key][..., index] = 0.0
                elif key.endswith("obs_normalizer._var"):
                    state[key][..., index] = 1.0
                elif key.endswith("obs_normalizer._std"):
                    state[key][..., index] = 1.0
        print(name, "normalizer keys:", [k for k in state if "normalizer" in k])
    torch.save(ckpt, args.dst)
    print("wrote", args.dst)


if __name__ == "__main__":
    main()

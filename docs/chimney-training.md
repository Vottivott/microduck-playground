# Continuing the chimney climbing policy

This guide continues the **main climb** checkpoint (w17, iteration 99,250), not
the enter/exit companions or whole video chain. All required data is public.

## Provenance and limitations

The checkpoint includes actor/critic, observation normalizers, Adam state,
iteration and environment counter. The helper compares these after loading.
Simulator episodes, RNG and rollout buffers restart fresh.

The original **w17 launch record was not recovered**. `resume.json` uses explicit
settings from the preceding w16 launcher with the current public task's remaining
defaults. This is a reproducible continuation recipe, **not exact historical w17
settings or bitwise replay**. The public source keeps contact-sensor capacity 500;
historical w17 could drop matches with its smaller buffer. We do not restore that bug.

The recovered bank has 645 ground-level approach handover states: `qpos` (645,21),
`qvel` (645,20), `origins` (645,3). Count, shape and date match the w16 launcher
description, but no original run-side hash survived to independently prove identity.
It is not the elevated exit-arrival bank. HF `training/handover-provenance.json`
documents the hash and provenance limits.

## Install and download

On a supported NVIDIA CUDA machine, from the repository root:

```bash
uv sync --locked
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "HannesVonEssen/microduck-chimney-climb",
    revision="e5b09f566bc958fc9d7c1529d085e9d98fa9ea37",
    allow_patterns=["checkpoint.pt", "training/*"],
    local_dir="policies/chimney-training", token=False,
)
PY
```

Only load trusted pickle checkpoints. Unset inherited `MICRODUCK_*` shell settings;
the helper rejects them rather than silently inheriting another experiment's recipe.
It checks the bank hash, dimensions and finite values and requires a new output path.

## Smoke test and continuation

Always start with the 64-environment, five-iteration smoke test:

```bash
uv run python scripts/resume_chimney.py \
  --checkpoint policies/chimney-training/checkpoint.pt \
  --bank policies/chimney-training/training/handover.pt \
  --output logs/chimney-resume-smoke --num-envs 64 --iterations 5
```

After it passes, start a separate continuation, for example:

```bash
uv run python scripts/resume_chimney.py \
  --checkpoint policies/chimney-training/checkpoint.pt \
  --bank policies/chimney-training/training/handover.pt \
  --output logs/chimney-resume-run1 --num-envs 4096 --iterations 1000
```

Iterations are **additional PPO updates**, not a target total. For a later resume,
use your new `continued.pt` and another new output directory. The helper restores
normalizer/optimizer/counter and advances the iteration label past the checkpoint.
Outputs include effective `params/env.yaml`, `params/agent.yaml`, initial restored
state, TensorBoard logs and `resume-validation.json`. Inspect penalty signs, NaN
terminations, braced height and rollouts. A smoke test is not long-term learning
validation or a hardware-safety claim. No new trained policy replaces the original.

## Randomization and resets

| Setting | Continuation recipe |
|---|---|
| Wall friction | **0.5–1.1**, sampled at startup through separate wall events |
| Foot friction | **0.7–1.1**, inherited task event |
| Episode / wall height | **40 s / 10 m** |
| Mid-corridor spawn heights | **0.25–3.0 m** |
| Ground / upright reset knobs | **0.45 / 0.6**, upright conditional on ground reset |
| Handover-bank probability | **0.40**, applied last, replacing selected poses |
| Upright yaw / x / joint-velocity noise knobs | **0.35 rad / 0.03 m / 0.4 rad/s** |

Width is not pinned to the video's 12.5 cm: the task's default width range is used.
BAM, inherited randomization, observation noise, delays, rewards and PPO defaults
come from the public source; the resolved configuration is saved for every run.
No action filter is added.

The earlier w9 friction experiment used **0.4–1.1**. Append
`--wall-friction 0.4 1.1` to explicitly choose that wider range. This is a changed
recipe, not evidence that w17 used it. The bare task defaults to fixed wall
friction 0.9; the helper sets the randomized range explicitly.

Export new checkpoints with `scripts/export.py Mjlab-Chimney-MicroDuck
--checkpoint-file ... --onnx-file ... --num-envs 1`, preserving the normalizer
and bounded actions. See [policy contract and chain limitations](chimney-climb.md).

## Verification (2026-09-24)

A fresh locked installation on one H200 downloaded checkpoint and bank from the
public HF revision above without authentication. The 64-environment/five-iteration
continuation restored actor, critic, normalizers, Adam state and counter exactly;
the counter advanced from 2,382,312 to 2,382,432. Both walls had 64 distinct
friction samples in the requested range, and 22 initial states matched the bank.
All five logged NaN termination rates were zero and chimney penalty signs were
non-positive. The chimney/approach/exit/resume regression suite passed 78 tests.
These checks establish a working continuation path, not equivalence to an
unrecovered historical launch or a new robustness result.

# Continue the released parkour policies

Both HF repos contain original full PPO checkpoints: actor, critic,
observation normalizers, populated Adam state and environment step counter.
Use the source revision in each repo's `training/source.json`. Install with
`uv sync --locked` on Linux/CUDA. MuJoCo also needs system EGL or OSMesa libraries;
the validation uses libosmesa6 and `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.
Only load trusted checkpoints.

| Profile | HF repo | Stored iteration |
| --- | --- | ---: |
| `long-jump` | `HannesVonEssen/microduck-long-jump` | 15,250 |
| `backflip` | `HannesVonEssen/microduck-backflip` | 12,000 |

```bash
hf download HannesVonEssen/microduck-long-jump checkpoint.pt --local-dir policies/long-jump
hf download HannesVonEssen/microduck-backflip checkpoint.pt --local-dir policies/backflip

# From the source root, without inherited MICRODUCK_* environment settings:
uv run python scripts/resume_release.py \
  --recipe experiments/parkour/resume.json --profile long-jump \
  --checkpoint policies/long-jump/checkpoint.pt \
  --output logs/long-jump-smoke --num-envs 64 --iterations 5
uv run python scripts/resume_release.py \
  --recipe experiments/parkour/resume.json --profile backflip \
  --checkpoint policies/backflip/checkpoint.pt \
  --output logs/backflip-smoke --num-envs 64 --iterations 5
```

After the smoke passes and its logs are inspected, choose a new output directory
and e.g. `--num-envs 4096 --iterations 250`. Iterations are **additional PPO
updates**, not a total target. For a subsequent `continued.pt`, add
`--descendant` while keeping the same task/source/recipe. This explicitly
relaxes the released-checkpoint hash guard; it does not infer another task's
observation semantics. Output directories must be new. Original release assets
are never replaced by the smoke or your continuation.

## What is verified and saved

`scripts/resume_release.py` compares a pre-training snapshot against the input
checkpoint's actor, critic, normalizers, Adam, iteration and counter. It applies
curricula at the **restored counter before collecting data**, synchronizes the
adaptive LR to restored Adam, and checks every step's observations, actions,
rewards and physical state for finiteness. It saves resolved config, effective
event parameters and reward weights, `resume_initial.pt`, `continued.pt`,
TensorBoard logs and `resume-validation.json`. The mandatory normalized
`scripts/export.py` export is checked against live, zero and random observations.

All recipes use procedural reverse-curriculum resets; no external motion/reset
bank, private local path or unshipped teacher is required to continue these
checkpoints. PPO state restoration is **not** restoration of old simulator,
RNG, actuator buffers or per-environment state. A smoke pass is not a behavior,
performance, robustness or hardware-safety result.

## Training settings are not render profiles

- **Long jump:** p1 launcher and recorded YAML were recovered. Entropy is 0.005;
  gap/drop are not pinned to the preview. The current source's range curriculum
  resumes at the saved counter (late range: gap 0.15–0.40 m, drop 0.15–0.35 m).
  This source includes later research changes, so it is a supported continuation
  rather than an exact reconstruction of p1's original implementation.
- **Backflip:** f21 launcher and recorded YAML were recovered: 0.7–0.9 m drop,
  entropy0.003, takeoff weight45, minimum takeoff vz0, curriculum shift3000.
  The initial midflip probability0.25 becomes **0.15** at the restored mature
  stage. The preview's fixed0.8m profile is not the training recipe. Current
  published source remains the implementation boundary, not a promise of exact
  historical replay.

Recipe provenance/limitations are machine-readable in `resume.json`; recovered
run YAML and hashes accompany the applicable HF training packages. Changes to
curricula/rewards beyond these recipes are new experiments and should be logged.
The training source's contact budgets include publication fixes.

**Simulation only. The landing may damage or break the real robot.** After
continuation, evaluate complete physical maneuvers and resets with the correct
command slots and action bounds, not reward totals alone. The original montage's
exact checkpoint/seed mapping remains unverified; training documentation does
not change that limitation.

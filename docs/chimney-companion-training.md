# Continue training chimney enter and exit

Full PPO checkpoints for **a11 enter (25,500)** and **x19 exit (73,750)** are
public: actor, critic, normalizers, populated Adam state and environment counter.
Use the source revision in HF `training/companions-source.json` and run
`uv sync --locked` on Linux with CUDA. Install system EGL or OSMesa libraries;
validation uses `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` with libosmesa6.
Only load trusted checkpoints. Main climb has a separate [recipe](chimney-training.md).

```bash
hf download HannesVonEssen/microduck-chimney-climb \
  --include 'enter/checkpoint.pt' 'exit/checkpoint.pt' 'training/*' \
  --local-dir policies/chimney

# From the playground root, with no inherited MICRODUCK_* overrides:
uv run python scripts/resume_release.py \
  --recipe experiments/chimney-climb/companions.json --profile enter \
  --checkpoint policies/chimney/enter/checkpoint.pt \
  --output logs/enter-smoke --num-envs 64 --iterations 5
uv run python scripts/resume_release.py \
  --recipe experiments/chimney-climb/companions.json --profile exit \
  --checkpoint policies/chimney/exit/checkpoint.pt \
  --output logs/exit-smoke --num-envs 64 --iterations 5
```

After inspecting the smoke results, use a new output directory and e.g.
`--num-envs 4096 --iterations 250`. Iterations mean **additional updates**.
For a later `continued.pt`, add `--descendant`, retaining the same task/source/
recipe. This relaxes the original-release hash guard only. Original policies
and checkpoints are never overwritten.

The helper compares the restored actor/critic/normalizers/Adam/counter; resets
episodes **after** restoring the counter to apply curriculum stages; starts the
adaptive scheduler from Adam's restored learning rate; checks every training
step for finite observations/actions/rewards/physical state; saves resolved
configs and post-restore event/reward settings; exports through `scripts/export.py`;
and checks normalized ONNX parity on live, zero and random observations.
Outputs include `resume_initial.pt`, `continued.pt`, `policy.onnx`,
`resume-validation.json`, TensorBoard logs and `params/`. A successful smoke
validates continuation plumbing, not improved behavior or hardware safety.

## Provenance and limits

Enter's recorded a11 agent/env YAML was recovered from the verified archive:
seed 137, 12.5 cm slot, 8 s episodes, dash weight 150, and 35% frontier / 15%
nearly-done / 50% outside starts. Current public source/contact implementation
is used, not a byte-identical historical source snapshot.

Exit's recovered `launch_x19.sh` records seed 151, a 1.2 m platform, 45 cm
overhang, 40 cm open extension, 7 cm clearance, 37.5 cm wall above the platform,
10 s episodes and primary reward weights. Current public source includes later
posture shaping/termination changes. This continues x19 on that explicit task,
**not its exact historical reward implementation**. Training geometry differs
from the 3 m, thinner presentation geometry.

**x19 did not require an arrival bank**: procedural starts came first; later
x20 experiments added bank training. Baseline exit uses 40% frontier, 20%
nearly-done, 40% procedural wedged starts and no external bank. Wall friction is
explicitly fixed at 0.9; main climb's randomized recipe is not silently copied.

## Optional real-arrival experiment

HF `training/exit-arrivals.pt` contains 18 recovered states. Count and tilt agree
with the historical filtered-bank description, but no contemporaneous hash
proves historical file identity. See `training/exit-arrivals-provenance.json`.
This is **not x19's training dataset**.

```bash
uv run python scripts/resume_release.py \
  --recipe experiments/chimney-climb/companions.json --profile exit \
  --checkpoint policies/chimney/exit/checkpoint.pt \
  --bank policies/chimney/training/exit-arrivals.pt \
  --output logs/exit-arrival-smoke --num-envs 64 --iterations 5
```

This opts into a **new** reset distribution: 45% bank replacements after
procedural resets. Positions rebase from the bank's 3 m platform to the recipe's
1.2 m platform. Hash, shapes and finiteness are checked. Preserved data is
qpos/qvel/origins/platform height only; action history resets to zero and hidden
actuator/solver/DR state is fresh. Do not infer live-handoff success or independent
robustness from this small selected bank or its smoke result.

Evaluate [the continuous chain](chimney-climb.md) separately, including command
semantics, offsets, action history and geometry. Fresh simulator/RNG state means
none of these recipes promises bit-identical historical replay.

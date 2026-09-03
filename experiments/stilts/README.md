# Stilt walking

This track learns locomotion while explicitly changing both support shape and
height. The showcased policy uses green 10 cm replacement sole-and-stilt
parts with blend 0.50 support: a rounded 17 × 22 mm tip. The clean 10-second
[`media/preview.mp4`](media/preview.mp4) shows the complete controlled policy
rollout. It completed without fall, reset, or auxiliary body contact. This is
a simulation milestone, not a printable hardware recommendation.

## Reproduce and extend

- Task: `Mjlab-Stilt-Flat-MicroDuck`;
- policy: PPO through `rsl_rl`;
- controls: 50 Hz joint-position actions;
- morphology: explicit contact mesh, mass, inertia, friction, and tip sensors;
- curriculum axes: support blend first, then height.

The source task is
[`src/mjlab_microduck/tasks/microduck_stilt_env_cfg.py`](../../src/mjlab_microduck/tasks/microduck_stilt_env_cfg.py),
and the compiled morphology is
[`src/mjlab_microduck/robot/stilt_constants.py`](../../src/mjlab_microduck/robot/stilt_constants.py).
See [`TRAINING.md`](TRAINING.md) for the executed curriculum and continuation
instructions, [`../../docs/stilt_training_plan.md`](../../docs/stilt_training_plan.md)
for the original plan, and
[`../../hardware/stilts`](../../hardware/stilts/README.md) for the parametric
hardware.

## Released policies

All released heights are collected in one
[`HannesVonEssen/microduck-stilts`](https://huggingface.co/HannesVonEssen/microduck-stilts)
model repository. Every height directory contains its own `policy.onnx`, full
playable video, continuation checkpoint, manifest, and checksum.
The policies share the 61D actor observation and 14D action contract, but each
is specialized to its listed height.

| Height | Training checkpoint | Model |
|---:|---:|---|
| 10 cm | 2,200 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/10cm) |
| 15 cm | 2,400 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/15cm) |
| 20 cm | 2,600 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/20cm) |
| 25 cm | 2,800 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/25cm) |
| 50 cm | 3,400 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/50cm) |
| 1.0 m | 4,000 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/100cm) |
| 1.4 m | 4,300 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/140cm) |
| 2.0 m | 6,500 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/200cm) |

Matching left, right, and paired STL geometry is indexed in the
[`hardware/stilts` README](../../hardware/stilts/README.md#released-policy-geometry).
These are simulation policies, not hardware candidates. Extreme simulated
heights do not validate a monolithic printed part, the mounting interface, or
the stock robot structure.

The machine-readable
[`eval/released_rollouts.json`](eval/released_rollouts.json) records the exact
checkpoint, seed, duration, reset state, root trajectory envelope, and artifact
hash for every released height. Each record is one controlled demonstration;
it should not be read as a multi-seed robustness claim.

## 10 cm printed-mass audit

The training morphology uses the explicit law `12 g + 1 g/cm` per stilt, so
the released 10 cm policy was trained with 22 g per stilt (44 g per pair). The
prototype slicer estimate is 58 g per pair, or approximately 29 g per stilt:
32% heavier than training nominal.

As a first sensitivity check, the released actor was evaluated unchanged with
29 g per stilt over 64 randomized environments for 10 seconds at the trained
0.15 m/s command. All environments survived; mean body-forward speed was
0.141 m/s and median maximum tilt was 3.28°. The exact result is saved in
[`eval/10cm_29g_seed123.json`](eval/10cm_29g_seed123.json). This small
simulation battery is encouraging, but it does not turn the policy into a
hardware candidate. A broader mass-randomized continuation and tethered,
low-speed hardware validation are still required.

Reproduce the audit with:

```bash
MICRODUCK_STILT_HEIGHT_CM=10 \
MICRODUCK_STILT_BLEND=0.5 \
MICRODUCK_STILT_MASS_KG=0.029 \
uv run python scripts/evaluate_running_checkpoint.py \
  --checkpoint-file checkpoint.pt \
  --task-id Mjlab-Stilt-Flat-MicroDuck \
  --speed 0.15 --num-envs 64 --duration-s 10 --warmup-s 1 \
  --seed 123 --output-file experiments/stilts/eval/10cm_29g_seed123.json
```

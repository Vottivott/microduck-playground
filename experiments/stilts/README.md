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
playable video, GIF preview, continuation checkpoint, manifest, and checksum.
The policies share the 61D actor observation and 14D action contract, but each
is specialized to its listed height.

| Height | Training checkpoint | Model |
|---:|---:|---|
| 10 cm | 2,200 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/100mm) |
| 15 cm | 2,400 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/150mm) |
| 20 cm | 2,600 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/200mm) |
| 25 cm | 2,800 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/250mm) |
| 50 cm | 3,400 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/500mm) |
| 1.0 m | 4,000 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/1000mm) |
| 1.4 m | 4,300 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/1400mm) |
| 2.0 m | 6,500 | [Artifacts](https://huggingface.co/HannesVonEssen/microduck-stilts/tree/main/2000mm) |

Matching left, right, and paired STL geometry is indexed in the
[`hardware/stilts` README](../../hardware/stilts/README.md#released-policy-geometry).
These are simulation policies, not hardware candidates. Extreme simulated
heights do not validate a monolithic printed part, the mounting interface, or
the stock robot structure.

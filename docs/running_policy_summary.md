# MicroDuck max-speed running policy

This document records the speed curriculum, robustification, and release
selection for the iteration-12,195 simulation-only hardware candidate. It has
not been validated on a physical robot.

## Source provenance

- Upstream base: `d424a0c899f6b33cbd3daeb279913134349c0b63`
- Task: `Mjlab-Running-Flat-MicroDuck`
- Running environment: `src/mjlab_microduck/tasks/microduck_running_env_cfg.py`
- Evaluator: `scripts/evaluate_running_checkpoint.py`
- Actor observation layout: 61D
- Action layout: 14D
- Algorithm: PPO (`rsl_rl`)

The pre-release development hashes were consolidated into the initial public
release and are intentionally not presented as public commits. Development
after that release uses ordinary commits.

## Speed frontier

The speed-focused branch resumed iteration 8,749 and trained for 3,000 updates.
Its winning iteration-11,748 checkpoint used:

```text
MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.2
MICRODUCK_RUNNING_SPEED_CAP=2.4
MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=-0.10
MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL=750
MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT=5.0
MICRODUCK_RUNNING_ENABLE_SYMMETRY=0
MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK=0
```

Six continuations compared 2.0, 2.1, and 2.2 m/s targets; standard and slower
stage pacing; and a stronger forward-progress reward. The 2.2 m/s target needed
the slower 750-update pacing. Faster pacing and the stronger reward both
underperformed.

At a 2.20 m/s command, iteration 11,748 reached 1.6873 m/s mean body-forward
speed and 98.83% survival across 1,024 randomized environments for 10 seconds.
Commands from 2.30 through 2.50 m/s did not increase realized speed, which
remained around 1.68 m/s. This checkpoint is preserved on Hugging Face under
`lineage/iteration-11748/`.

## Robustification

Iteration 11,748 was continued for three fixed stages on the ordinary robot
model. All stages used 4,096 environments, seed 42, 3–6 second push intervals,
and the ordinary 0.7–1.3 friction range.

| Stage | Updates | Planar velocity push | Trunk CoM | Head CoM | Initial pitch/roll |
|---|---:|---:|---:|---:|---:|
| A | 100 | ±0.03 m/s | ±3 mm | ±3 mm | ±1° |
| B | 150 | ±0.06 m/s | ±5 mm | ±5 mm | ±1.5° |
| C | 200 | ±0.10 m/s | ±8 mm | ±6 mm | ±2° |

The resulting iteration-12,195 checkpoint was selected. A separately
backlash-trained continuation was evaluated but performed worse, so it was not
used. The selected ordinary-model policy was instead cross-evaluated unchanged
on the backlash model.

## Final evaluation

All batteries used a 2.20 m/s command, a 1-second warm-up, and a 10-second
measurement horizon.

| Evaluation | Environments | Body-forward speed | Survival |
|---|---:|---:|---:|
| Ordinary model, nominal | 512 | 1.651 m/s | 99.22% |
| Ordinary model, push/CoM/tilt stress | 512 | 1.635 m/s | 98.83% |
| Backlash model, nominal | 256 | 1.636 m/s | 98.83% |
| Backlash model, push/CoM/tilt stress | 256 | 1.612 m/s | 98.44% |
| High grip plus full stress | 512 | 1.642 m/s | 96.68% |

The held-out stress case used ±0.10 m/s pushes, ±10 mm trunk CoM, ±6 mm head
CoM, and ±2° initial pitch/roll. The high-grip case additionally widened foot
friction to 0.7–1.8. Note that the held-out trunk-CoM range is wider than the
±8 mm used in the final training stage.

Robustification reduced nominal body-forward speed by about 2.1% relative to
iteration 11,748. The substantially broader evaluation performance makes
iteration 12,195 the more defensible public default. It is still not a claim of
hardware readiness: mean heading error is 31.33° nominal and 43.81° in the
backlash-plus-stress battery, with large lateral displacement.

The walking baseline samples forward commands only up to 0.40 m/s. The robust
candidate's nominal body-forward speed is approximately 4.1 times that
configured ceiling; this is not a matched measured walking-policy comparison.

## Artifact versions

[`HannesVonEssen/microduck-running`](https://huggingface.co/HannesVonEssen/microduck-running)
contains:

- root `policy.onnx`, `checkpoint.pt`, and `media/preview.mp4`: iteration 12,195;
- `lineage/iteration-11748/`: the speed-focused parent;
- `legacy/iteration-8749/`: the earlier exact-video release;
- `media/run-into-mat.mp4`: a scripted edit driven by iteration 8,749.

The root ONNX accepts `[1, 61]`, returns `[1, 14]`, and includes the observation
normalizer. Running training did not clip actor outputs, so no output clamp is
required. A direct checkpoint-to-ONNX comparison over zero and deterministic
random inputs measured a maximum absolute action difference of 7.63e-6.

The full machine-readable metrics, artifact hashes, and stage parameters are in
[`../experiments/running/eval/released_12195.json`](../experiments/running/eval/released_12195.json).

Do not deploy this directly to hardware without an ONNX runtime rehearsal,
support rig, conservative command ramp, current and temperature monitoring,
and an emergency stop.

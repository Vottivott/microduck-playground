# Microduck max-speed running policy

This document records the simulation result selected on 2026-08-30 and the
artifacts needed to reproduce or deploy it. The policy has not yet been
validated on hardware.

## Source provenance

- Base commit: `d424a0c899f6b33cbd3daeb279913134349c0b63`
- Public experiment implementation: initial playground release `7a75253`
- Running environment: `src/mjlab_microduck/tasks/microduck_running_env_cfg.py`
- Evaluator: `scripts/evaluate_running_checkpoint.py`

The pre-release development hashes for the continuation controls and
higher-speed curriculum were consolidated into the initial public release and
are intentionally not presented as public commits.
- Task: `Mjlab-Running-Flat-MicroDuck`
- Actor observation layout: 61D
- Action layout: 14D
- Training backend: PPO (`rsl_rl`) with parallel simulated environments
- Winning branch: `highspeed2-max22-slow-r1`
- Winning checkpoint: iteration 11,748
- Resumed from: `sweep-speed-r1`, iteration 8,749
- Continuation length: 3,000 iterations

The winning continuation used:

```text
MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.2
MICRODUCK_RUNNING_SPEED_CAP=2.4
MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=-0.10
MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL=750
MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT=5.0
MICRODUCK_RUNNING_ENABLE_SYMMETRY=0
MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK=0
```

Six continuations were trained concurrently from the same checkpoint and seed.
They compared 2.0, 2.1, and 2.2 m/s targets; standard and slower stage pacing;
and a stronger forward-progress reward. The 2.2 m/s target required slower
750-iteration pacing. Faster pacing and the stronger reward both underperformed.

## Selected evaluation

The recommended operating point is a 2.20 m/s forward command. Across 1,024
randomized environments evaluated for 10 seconds:

- survival fraction: 0.9883
- mean body-forward speed: 1.6873 m/s
- mean displacement speed along the initial heading: 1.3490 m/s
- mean absolute heading error: 26.85 degrees
- flight fraction: 0.6602

On the identical 2.20 m/s, 1,024-by-10-second battery, the previous iteration
8,749 policy achieved 1.6414 m/s body-forward speed, 1.1843 m/s initial-heading
progress, and 0.7666 survival. The new policy improves those by 2.8%, 13.9%,
and 22.2 percentage points respectively.

Commands from 2.30 through 2.50 m/s did not increase realized speed. The policy
remained stable, but body speed stayed near 1.68 m/s and net progress near
1.40 m/s in the 512-by-8-second boundary battery. Further improvement again
requires another trained stage rather than a larger deployment command.

The velocity task used as the walking baseline samples forward commands only
up to 0.40 m/s. The selected running policy is approximately 4.2 times that
configured ceiling by body-forward speed, or 3.4 times by net initial-heading
progress. This is not a matched measured walking-policy comparison.

## Artifact storage

The public deployment package at
[`HannesVonEssen/microduck-running`](https://huggingface.co/HannesVonEssen/microduck-running)
contains the earlier iteration-8,749 checkpoint used in the run-into-mat video,
its clean rollout, manifest, and checksums. The later iteration-11,748 result
documented here remains a research result rather than the model currently
published there.

Preserved iteration-11,748 source assets:

- `microduck-running-highspeed2-final-model-11748.pt`: resumable PPO checkpoint
- `microduck-running-highspeed2-final-model-11748.onnx`: deployment policy with the
  observation normalizer baked in
- `microduck-running-highspeed2-final-close-2.20mps.mp4`: close-up 10-second rollout
- `microduck-running-highspeed2-evaluations-11748.tar.gz`: milestone, terminal,
  operating-limit, and matched-baseline evaluation JSON
- `SHA256SUMS`: integrity hashes for all release assets

The iteration-8,749 package is intentionally retained because it corresponds
exactly to the published demonstration. The later checkpoint is the stronger
simulation result, but should be released as a distinct version if selected
for deployment experiments.

The ONNX artifact was checked with `onnx.checker`, then run using ONNX Runtime
on CPU. It accepts `[1, 61]`, returns finite `[1, 14]` actions, and contains the
baked observation-normalizer mean.

## Verification

- Full CPU test suite passed at the selected release revision (run the command
  below for the current count):

  ```bash
  uv run --with pytest pytest -q
  ```
- Smoke training: 64 environments, 5 iterations, 61D observations, no NaNs
- Milestone comparisons: all six branches at iterations 9,000, 9,750, 10,250,
  10,750, and 11,250
- Terminal comparison: 256 randomized environments for 8 seconds
- Final head-to-head: 512 randomized environments for 8 seconds, followed by
  1,024 randomized environments for 10 seconds for the two finalists
- Operating-limit probes: 512 randomized environments for 8 seconds at 2.30,
  2.40, and 2.50 m/s commands

Do not deploy this directly to hardware without the usual ONNX runtime rehearsal,
support rig, conservative command ramp, and current/temperature monitoring.

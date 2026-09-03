# Fast running

The default release is the iteration-12,195 robustified simulation candidate.
It was continued from the faster iteration-11,748 policy through progressively
wider velocity pushes, trunk/head centre-of-mass offsets, and initial tilt. Its
clean 10-second [`media/preview.mp4`](media/preview.mp4) and the root ONNX on
[`HannesVonEssen/microduck-running`](https://huggingface.co/HannesVonEssen/microduck-running)
are this same checkpoint.

At a 2.20 m/s command, its 512-environment nominal battery measured 1.651 m/s
mean body-forward speed and 99.22% survival. The most directly relevant
backlash-plus-disturbance battery measured 1.612 m/s and 98.44% survival. This
is a **simulation-only hardware candidate**, not a hardware-validated policy.
Heading and lateral drift remain substantial.

The Hugging Face repository is deliberately versioned:

- root `policy.onnx` and `checkpoint.pt`: robust iteration 12,195;
- `lineage/iteration-11748/`: the speed-focused parent used to start
  robustification;
- `legacy/iteration-8749/`: the earlier controller shown in the run-into-mat
  edit.

The complete PPO checkpoints contain the actor, critic, optimizer, observation
normalizers, and curriculum counter. PyTorch checkpoints use pickle internally;
load them only from a repository and revision you trust.

## Evaluation

All reported batteries ran for 10 seconds after a 1-second warm-up at a
2.20 m/s forward command.

| Evaluation | Environments | Body-forward speed | Survival |
|---|---:|---:|---:|
| Ordinary model, nominal | 512 | 1.651 m/s | 99.22% |
| Ordinary model, push/CoM/tilt stress | 512 | 1.635 m/s | 98.83% |
| Backlash model, nominal | 256 | 1.636 m/s | 98.83% |
| Backlash model, push/CoM/tilt stress | 256 | 1.612 m/s | 98.44% |
| High grip plus full stress | 512 | 1.642 m/s | 96.68% |

The stress battery applies ±0.10 m/s planar velocity pushes every 3–6 seconds,
±10 mm trunk CoM offsets, ±6 mm head CoM offsets, and ±2° initial pitch and
roll. The high-grip case widens foot friction from the ordinary 0.7–1.3 range
to 0.7–1.8. The candidate was trained with the ordinary actuator model and then
cross-evaluated on the backlash model; a separately backlash-trained
continuation performed worse and was not selected.

The compact, machine-readable release record is
[`eval/released_12195.json`](eval/released_12195.json). The previous exact-video
record remains at [`eval/released_8749.json`](eval/released_8749.json).

## Continue training

Download the root `checkpoint.pt`, then place it in the standard RSL-RL layout:

```bash
mkdir -p logs/rsl_rl/running/release-12195
cp checkpoint.pt logs/rsl_rl/running/release-12195/model_12195.pt

MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.2 \
MICRODUCK_RUNNING_SPEED_CAP=2.4 \
MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL=750 \
MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=-0.10 \
MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT=5.0 \
MICRODUCK_RUNNING_ROBUST_PUSH_MPS=0.10 \
MICRODUCK_RUNNING_ROBUST_TRUNK_COM_M=0.008 \
MICRODUCK_RUNNING_ROBUST_HEAD_COM_M=0.006 \
MICRODUCK_RUNNING_ROBUST_INITIAL_TILT_DEG=2.0 \
uv run train Mjlab-Running-Flat-MicroDuck \
  --agent.resume True \
  --agent.load-run release-12195 \
  --agent.load-checkpoint model_12195.pt \
  --agent.max-iterations 100
```

The selected policy was reached from iteration 11,748 with these fixed stages:

| Stage | Updates | Push | Trunk CoM | Head CoM | Initial tilt |
|---|---:|---:|---:|---:|---:|
| A | 100 | ±0.03 m/s | ±3 mm | ±3 mm | ±1° |
| B | 150 | ±0.06 m/s | ±5 mm | ±5 mm | ±1.5° |
| C | 200 | ±0.10 m/s | ±8 mm | ±6 mm | ±2° |

All stages used 4,096 environments, seed 42, 3–6 second push intervals, and
the ordinary 0.7–1.3 friction range. See
[`../../docs/running_policy_summary.md`](../../docs/running_policy_summary.md)
for the earlier speed curriculum and the full selection rationale.

## Policy contract

- input: float32 `[1, 61]`;
- output: float32 `[1, 14]`;
- control rate: 50 Hz;
- action scale: 1.0 around the MicroDuck HOME joint pose;
- entry pose: standing;
- hardware modifications: none;
- observation normalizer: embedded in ONNX;
- output clipping: none (training also used unclipped actor output).

Checkpoint-to-ONNX parity was checked on zero and deterministic random
observations; the maximum absolute action difference was 7.63e-6.

Do not begin hardware testing at the maximum simulated command. Use a support
rig, conservative command ramp, current and temperature monitoring, and an
emergency stop.

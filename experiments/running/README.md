# Fast running

The later iteration-11,748 continuation shown in the clean 10-second
[`media/preview.mp4`](media/preview.mp4) reaches 1.687 m/s mean body-forward
speed in its randomized simulation battery—about four times the 0.4 m/s
walking-training ceiling. It is a documented simulation result, not the
downloadable policy linked below and not a hardware-robustified release.

The ONNX package for iteration 8,749—the approximately 1.607 m/s checkpoint
used in the run-into-mat demonstration—its manifest, checksums, runtime contract, and sim-to-real
boundary are published separately at
[`HannesVonEssen/microduck-running`](https://huggingface.co/HannesVonEssen/microduck-running).
The clean preview here and the metrics below document iteration 11,748. The
two checkpoints are deliberately identified rather than presented as the same
artifact. No hardware-robustified running policy is currently published.
Training code and continuation controls remain in this repository.

The Hugging Face package includes the complete iteration-8,749 PPO
`checkpoint.pt` (actor, critic, optimizer, observation normalizers, and
curriculum counter), so that exact released line can be resumed rather than
merely inferred through ONNX. PyTorch checkpoints use pickle internally; load
them only from a repository and revision you trust.

To continue it, place the file in the standard RSL-RL run layout:

```bash
mkdir -p logs/rsl_rl/running/release-8749
cp checkpoint.pt logs/rsl_rl/running/release-8749/model_8749.pt

uv run train Mjlab-Running-Flat-MicroDuck \
  --agent.resume True \
  --agent.load-run release-8749 \
  --agent.load-checkpoint model_8749.pt \
  --agent.max-iterations 100
```

## Selected evaluation

At the 2.20 m/s forward command, the 1,024-environment, 10-second randomized
battery produced:

- 98.83% survival;
- 1.6873 m/s mean body-forward speed;
- 1.3490 m/s mean displacement speed along the initial heading;
- 26.85° mean absolute heading error;
- 66.02% flight fraction.

Commands above 2.20 m/s did not improve realized speed. Heading control is a
known weakness, and these figures are simulation measurements rather than
hardware results. See [`../../docs/running_policy_summary.md`](../../docs/running_policy_summary.md)
for the complete training and evaluation record.

## Policy contract

- input: float32 `[1, 61]`;
- output: float32 `[1, 14]`;
- control rate: 50 Hz;
- action scale: 1.0 around the MicroDuck HOME joint pose;
- entry pose: standing;
- hardware modifications: none.

Do not begin hardware testing at the maximum simulated command. Use a support
rig, conservative command ramp, current and temperature monitoring, and an
emergency stop.

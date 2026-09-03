# Fast running

The clean 10-second [`media/preview.mp4`](media/preview.mp4) and downloadable
ONNX are the same iteration-8,749 policy. Its 256-environment randomized
simulation battery measured 1.607 m/s mean body-forward speed at a 1.80 m/s
command—about four times the walking task's 0.4 m/s command ceiling. This is a
simulation result, not a hardware-robustified release.

The ONNX package, manifest, checksums, runtime contract, run-into-mat edit, and
sim-to-real boundary are published at
[`HannesVonEssen/microduck-running`](https://huggingface.co/HannesVonEssen/microduck-running).
No hardware-robustified running policy is currently published. Training code
and continuation controls remain in this repository. The later iteration-11,748
research continuation is documented separately in
[`../../docs/running_policy_summary.md`](../../docs/running_policy_summary.md).

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

At a 1.80 m/s forward command, the released checkpoint's 256-environment,
8-second randomized battery produced:

- 99.22% survival;
- 1.6073 m/s mean body-forward speed;
- 1.3277 m/s mean displacement speed along the initial heading;
- 25.22° mean absolute heading error;
- 65.72% flight fraction.

At a 2.00 m/s extrapolated command, mean body-forward speed rose only to
1.6452 m/s while survival fell to 93.36%. Heading control is a known weakness,
and these figures are simulation measurements rather than hardware results.
The compact released-policy record is
[`eval/released_8749.json`](eval/released_8749.json).

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

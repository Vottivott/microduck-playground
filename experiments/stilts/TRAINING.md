# Stilt training and continuation

## Executed curriculum

The released actors form one seed-72 continuation lineage. Support was first
narrowed at a fixed 2 cm height, then the selected blend-0.50 tip was raised
in small stages. Each stage loaded the preceding selected checkpoint; height
and support area were never made harder in the same transition.

| Iteration | Height | Blend | Role |
|---:|---:|---:|---|
| 400 | 2 cm | 0.00 | platform bootstrap |
| 500 | 2 cm | 0.25 | support transition |
| 600 | 2 cm | 0.50 | selected 17 × 22 mm tip |
| 700–800 | 3–4 cm | 0.50 | early height ramp |
| 2,000 | 5 cm | 0.50 | gait consolidation |
| 2,100–2,800 | 7.5–25 cm | 0.50 | 2.5 cm height steps |
| 2,900–3,400 | 27.5–50 cm | 0.50 | widening height steps |
| 3,500–4,200 | 55 cm–1.2 m | 0.50 | simulation extension |
| 4,300 | 1.4 m | 0.50 | released milestone |
| 4,600–6,000 | 1.5–1.9 m | 0.50 | tall-stilt bridge |
| 6,500 | 2.0 m | 0.50 | released simulation ceiling |

The individually released milestones are 10 cm/2,200, 15 cm/2,400,
20 cm/2,600, 25 cm/2,800, 50 cm/3,400, 1.0 m/4,000, 1.4 m/4,300, and
2.0 m/6,500 (height / iteration). The 3.0 m clip was an unchanged-policy
zero-shot failure and is not a trained or released policy.

## What the Hugging Face checkpoints contain

The selected ONNX files were retained, but the original stilt PPO critic and
optimizer snapshots were not. Each `checkpoint.pt` in
[`HannesVonEssen/microduck-stilts`](https://huggingface.co/HannesVonEssen/microduck-stilts)
therefore provides:

- actor weights and actor observation normalization reconstructed exactly from
  the selected ONNX;
- a shape-compatible fresh critic scaffold;
- cleared optimizer moments with a conservative `1e-5` learning rate;
- exploration standard deviation reset to `0.1`; and
- the released iteration and curriculum counter.

These are actor-exact continuation warm starts, not byte-identical original PPO
training states. The distinction is recorded inside each checkpoint under
`infos.release_continuation` and in its manifest. Reconstruct and independently
verify one with `scripts/reconstruct_stilt_continuation.py`.

PyTorch checkpoints use pickle internally. Load them only from a repository and
revision you trust.

## Continue one released height

Download one height directory, place its checkpoint in the ordinary RSL-RL
run layout, and keep the matching morphology fixed:

```bash
mkdir -p logs/rsl_rl/stilt_locomotion/release-25cm
cp checkpoint.pt logs/rsl_rl/stilt_locomotion/release-25cm/model_2800.pt

MICRODUCK_STILT_HEIGHT_CM=25 MICRODUCK_STILT_BLEND=0.5 \
  uv run train Mjlab-Stilt-Flat-MicroDuck \
    --agent.resume True \
    --agent.load-run release-25cm \
    --agent.load-checkpoint model_2800.pt \
    --agent.max-iterations 100
```

Because the critic and optimizer are fresh, begin with a short run and inspect
value loss, termination rate, contact identity, and rollout video before
raising the learning rate. To continue the height curriculum, first stabilize
the downloaded morphology; then change only
`MICRODUCK_STILT_HEIGHT_CM`, leaving blend at `0.5`.

## Start from scratch

```bash
MICRODUCK_STILT_HEIGHT_CM=2 MICRODUCK_STILT_BLEND=0 \
  uv run train Mjlab-Stilt-Flat-MicroDuck \
    --env.scene.num-envs 64 \
    --agent.max-iterations 5
```

For a full recreation, promote checkpoints through the executed table above
and apply the gates in [`../../docs/stilt_training_plan.md`](../../docs/stilt_training_plan.md).

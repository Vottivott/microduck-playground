# Continue training released policies

The source repository contains the environments, training and evaluation
tools, hardware geometry, and locked Python dependency graph. Learned artifacts
live on Hugging Face. Always start with a small smoke continuation, inspect its
rollout and losses, then restart from the downloaded checkpoint for a longer
experiment.

```bash
git clone https://github.com/Vottivott/microduck-playground.git
cd microduck-playground
uv sync --locked
```

PyTorch checkpoints use pickle internally. Download and load them only from a
repository and revision you trust.

## Continuation fidelity

| Policy | Hugging Face artifact | Preserved training state | Guide |
|---|---|---|---|
| Running | root `checkpoint.pt` | actor, critic, optimizer, normalizers, curriculum counter | [`experiments/running/README.md`](experiments/running/README.md#continue-training) |
| Basketball | root `checkpoint.pt` | recurrent actor, critic, Adam state, normalizers, curriculum counter | [`experiments/basketball/README.md`](experiments/basketball/README.md#download-and-continue-training) |
| Stilts | `<height>/checkpoint.pt` | exact actor and normalizer; fresh critic and optimizer | [`experiments/stilts/TRAINING.md`](experiments/stilts/TRAINING.md#continue-one-released-height) |
| Swing | root `checkpoint.pt` | exact selected actor; source-endpoint critic and empty optimizer | [`experiments/swing/TRAINING.md`](experiments/swing/TRAINING.md#continue-from-the-released-policy) |

Running and basketball are full PPO resumes, although simulated episode and
random-number-generator state are not restored bit-for-bit. Stilt and swing
checkpoints are explicitly labeled actor-exact warm starts; do not describe
them as original optimizer-state continuations.

## Download examples

```bash
hf download HannesVonEssen/microduck-running \
  checkpoint.pt policy.onnx config.json manifest.json SHA256SUMS \
  --local-dir artifacts/running

hf download HannesVonEssen/microduck-basketball \
  checkpoint.pt policy.onnx config.json manifest.json SHA256SUMS \
  --local-dir artifacts/basketball

# One complete stilt specialization from the shared eight-height repository.
hf download HannesVonEssen/microduck-stilts \
  25cm/checkpoint.pt 25cm/policy.onnx 25cm/manifest.json \
  config.json index.json SHA256SUMS \
  --local-dir artifacts/stilts

hf download HannesVonEssen/microduck-swing \
  checkpoint.pt policy.onnx config.json manifest.json SHA256SUMS \
  --local-dir artifacts/swing
```

The stilt height in the environment must match the downloaded specialization.
The swing continuation belongs to the alpha-0.50 release lineage represented
by the published 173.20° seed-27 video. Its release source deliberately stops
at that selected actor.

## Reproduction boundary

The experiment guides record the executed stages and selection rationale.
Re-running stochastic PPO is not expected to produce byte-identical weights
across different accelerator, driver, PyTorch, or MuJoCo Warp versions. Record
the Git commit, Hugging Face revision, seed, morphology, and physical/evaluation
gates when publishing a continuation.

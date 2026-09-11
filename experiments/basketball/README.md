# Blind basketball balance

Microduck balances on a free-rolling size-7 basketball and follows velocity commands
using proprioception and a **one-layer LSTM**, with **no ball state in the actor inputs**.
The released checkpoint is **b11, iteration 6,999**.
This is an **experimental policy for hardware testing**:
simulation and export checks passed; onboard timing and real-robot behavior are untested.

[ONNX, full training checkpoint and configuration on Hugging Face](https://huggingface.co/HannesVonEssen/microduck-basketball) ·
[Training source](https://github.com/Vottivott/microduck-playground) · [LSTM runtime support: PR #231](https://github.com/pollen-robotics/microduck/pull/231)

## Video and measured performance

[Full 30-second simulation video](media/preview.mp4), seed 0, with a free ball,
velocity commands and disturbances. The selected take runs without a termination.

Matched evaluation: 1,024 environments per seed (101, 202, 303), 3,072 trials per
policy, 60 seconds, first fall counts as failure. Play disturbances reset base
horizontal velocity within ±0.09 m/s every 0.5–1 seconds, without angular pushes.
Commands span ±0.30 m/s forward, ±0.20 m/s lateral and ±1.0 rad/s yaw.

| Policy | Survive 20 s | Survive 60 s | Yaw tracking MAE, rad/s | Action-change RMS |
|---|---:|---:|---:|---:|
| **b11_6999 blind LSTM** | **98.86%** | **97.01%** | **1.2618** | **0.23384** |
| b2_6250 ball-state MLP reference | 99.67% | 99.12% | 0.7574 | 0.13657 |

b11 survives 2,980/3,072 trials; the reference survives 3,045/3,072. Survival is
closer to the reference, but turning accuracy and action smoothness remain worse.
Tracking/action metrics exclude samples at and after first failure, so they are
conditioned on survival. Results do not establish equal movement quality or transfer.
Per-seed results and the export receipt are in `eval/` and `validation.json`.

## Architecture and deployment contract

The normalized 61D observation feeds LSTM(256), then an ELU MLP(512, 256, 128),
then 14 actions. The actor has about 624,000 parameters. The usual policy feeds
its observation directly into the MLP. Recurrent memory lets this actor use past
proprioception to help estimate otherwise hidden contact/ball dynamics.
The privileged training critic can observe the ball; it is absent from ONNX.

| Observation indices | Input |
|---|---|
| 0:3 | body gyroscope |
| 3:6 | projected gravity |
| 6:20 | joint positions in the usual runtime convention |
| 20:34 | joint velocities |
| 34:48 | previous 14 policy actions |
| 48:51 | forward, lateral and yaw velocity commands |
| 51:55 | zero head-command padding |
| 55:61 | zero body-command padding; **no ball state** |

Use the standard Microduck joint order, observation scaling and action-to-joint
mapping. Observation normalization is already baked into `policy.onnx`.
The float32 ONNX interface is:

- Inputs: `obs [1,61]`, `h_in [1,1,256]`, `c_in [1,1,256]`.
- Outputs: `actions [1,14]`, `h_out [1,1,256]`, `c_out [1,1,256]`.
- Run at **50 Hz**. Carry both output states into the next step; initialize them
  to zero and reset on activation, policy switches and recovery/reset boundaries.
  Retain memory during ordinary command changes.
- Publish/install with **`model_api: 2`**. The existing feedforward-only loader
  needs the recurrent support in [runtime PR #231](https://github.com/pollen-robotics/microduck/pull/231), or an equivalent loader.
- Force head/body command slots to zero. No camera or ball tracker is required.
- Training used **no action low-pass filter**. Disable optional runtime action
  filters for matched deployment; this is separate from the action-rate reward.
- Start with the robot supported upright on the ball apex, as in training.
  The policy does not climb onto the ball or recover itself from the floor.

The action-rate reward coefficient was −0.2. “Smoothing” here means penalizing
squared action changes during learning, not filtering outputs afterward.
b11 continuation trained commands ±0.15 m/s forward, ±0.10 m/s lateral and
±0.50 rad/s yaw. The wider commands above are an evaluation stress test.
`[0,0,0]` is the balance-in-place command.

40-step PyTorch/ONNX comparison with a reset: maximum absolute action error
1.43e-6. An 80-step production Rust/Python ONNX comparison with a reset matched
exactly on the development host. Neither check measures board latency. Run the
runtime PR's `policy-rehearsal` example on the robot computer to check the 20 ms
control budget before motor deployment.

## Download and continue training

The Hugging Face repo contains `policy.onnx`, **`checkpoint.pt`** (actor, critic,
normalizers, Adam optimizer state and iteration/curriculum counter), recorded
`params/agent.yaml` and `params/env.yaml`, `source.tar.gz`, evaluation, video,
manifest and checksums. The YAML files are records; the Python continuation
factory is the executable configuration. Source is also in this GitHub repo. For an exact release checkout, use the
`training.commit` in the Hugging Face manifest, or unpack `source.tar.gz` into
a new directory and run `uv sync --locked` there.
Resuming restores learned/optimizer state, but starts new simulated episodes;
the checkpoint is not a bitwise snapshot of every environment and RNG state.

```bash
git clone https://github.com/Vottivott/microduck-playground
cd microduck-playground
uv sync --locked
# Authenticate with your Hugging Face account while the model repo is private.
uv run python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download('HannesVonEssen/microduck-basketball', local_dir='artifacts/basketball')
PYCODE
cd artifacts/basketball
sha256sum -c SHA256SUMS
cd ../..

# Required smoke test before a longer continuation.
uv run python scripts/finetune_basketball.py artifacts/basketball/checkpoint.pt \
  --run-name basketball-resume-smoke --num-envs 64 --iterations 5 \
  --learning-rate 2e-5 --push-interval-s 1.5 3 --save-interval 5

# Continue the released recipe from b11; this is a new experiment.
uv run python scripts/finetune_basketball.py artifacts/basketball/checkpoint.pt \
  --run-name basketball-resume --num-envs 4096 --iterations 500 \
  --learning-rate 2e-5 --action-rate-weight -0.2 --command-scale 1 \
  --episode-seconds 10 --seed 42 --push-interval-s 1.5 3 --save-interval 125
```

The runner keeps the actor blind and the ball free, restores Adam's moments,
then explicitly reinstalls the fixed 2e-5 learning rate. This avoids the saved
optimizer silently overwriting the requested rate. It infers the LSTM dimensions
from the checkpoint. No teacher or distillation is used in b11.

Export each new checkpoint through the normalized exporter, then check parity:

```bash
MICRODUCK_BB_BLIND=1 MICRODUCK_BB_HISTORY=1 uv run python scripts/export.py \
  Mjlab-Basketball-MicroDuck --checkpoint-file PATH/TO/model_XXXX.pt \
  --onnx-file output.onnx
uv run python scripts/verify_basketball_onnx_parity.py PATH/TO/model_XXXX.pt output.onnx
uv run python scripts/eval_basketball_long.py PATH/TO/model_XXXX.pt result.json \
  --seed 101 --seconds 60 --num-envs 1024 --command-scale 2 --blind
```

Repeat evaluation with seeds 202 and 303 and inspect a video before selecting a
replacement. See [TRAINING.md](TRAINING.md) for the experiment lineage.

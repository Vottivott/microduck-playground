# Continue training

The full checkpoints are hosted on Hugging Face; this GitHub directory contains the source snapshot, reset banks, geometry, recordings and locked dependencies. Use a CUDA machine for MuJoCo Warp; keep evaluation seeds out of training.

```bash
cd experiments/desk-climb/source
uv sync --frozen
uv run ../training/download_models.py

# Mandatory small smoke test before a larger continuation.
uv run ../training/run.py climber --envs 64 --iterations 5 --out /tmp/desk-climber-smoke
uv run ../training/run.py getup --envs 64 --iterations 5 --out /tmp/desk-getup-smoke

# Explicit, bounded continuations from the released PPO/Adam states.
uv run ../training/run.py climber --envs 1024 --iterations 250 --seed 233 --out logs/desk-climber
uv run ../training/run.py getup --envs 1024 --iterations 128 --out logs/desk-getup

# Frozen full-sequence evaluation, with the recorded simulator supervisor.
uv run ../training/run.py evaluate --envs 64 --seconds 60 --seed 19923 --out logs/desk-eval
```

The climber recipe uses90% floor starts and10% upper-ladder starts, LR5e-6 and the original0.74m recovery termination threshold. Although the physical table is0.66m high, the selected checkpoint is the **unchanged-height control**. Preserve that distinction when reproducing its training. It resumes actor, critic and Adam exactly; environment counters start afresh as recorded.

The warm get-up checkpoint continues with frozen actor normalization, LR1e-6 and the original98-state training bank. Its separate validation bank contains154 states. Bank resets restore kinematic state and previous actions, but not historical BAM/sag/domain-randomization buffers; evaluate the live full sequence separately. The original official actor checkpoint, ONNX and actor reconstruction provenance are also supplied.

After any training, use the preserved official exporter (normalization is mandatory):

```bash
uv run python -c "import mjlab_microduck.tasks; from scripts.export import run_export, ExportConfig; run_export('Mjlab-DeskRecoveryBlind-MicroDuck', ExportConfig(checkpoint_file='logs/desk-climber/final.pt', onnx_file='logs/desk-climber/policy.onnx', num_envs=1, device='cuda:0'))"
```

Repeat CPU normalized export/reset checks and evaluate physical support, standing duration, falls, drift and finite trajectories before adopting a candidate. Reward or a selected video alone does not establish improvement. The package includes the selected original models so experimental continuations need not overwrite them.

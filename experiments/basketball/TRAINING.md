# Basketball training lineage

The release is b11, iteration 6999, a blind recurrent PPO actor. Earlier b1/b2
ball-state policies are comparison references, not the released model.

| Stage | Change | Observation / outcome |
|---|---|---|
| b2_6250 | Original ball-state MLP | Strong simulation reference, requires ball tracking |
| b4 / b5 | Blind single frame / five-frame history | Near-zero 10-second survival |
| b7_5999 | Blind ten-frame history | 44.6% at 10 seconds in the early slow-command battery |
| b6_5999 | Blind LSTM256 | 84.4% at 10 seconds in that battery; 61.9% at 20 seconds with wider commands |
| b8 | Teacher/student distillation pilot | Failed to preserve balance; not in the released lineage |
| b9 | PPO from b6_4250, free ball, action-rate weight −0.2 | b9_6500 selected by matched 60-second validation |
| b10 | b9_6500 + fixed LR 2e-5, unchanged 3–6 s push interval | Final seed404: 92.19% 60-second survival |
| b11 | Same source/LR, pushes every 1.5–3 s | Final seed404: 97.85%; matched seeds101/202/303: 97.01% |

b9 continued for 3,000 iterations; final7249 regressed, so validation selected6500.
b10 and b11 each resumed the full source checkpoint and trained 500 iterations,
4096 environments, seed42, saving every125 iterations. b11 changed recovery
practice, retaining the reward stack, unfiltered actions, 10-second episodes,
command scale1 and fixed action-rate weight−0.2. Both completed without NaNs.
64-environment, five-iteration smoke and normalized export checks passed before
these continuations. Final b11 had 92 first falls versus 343 for b9_6500 in the
matched 3072-trial battery (73.2% fewer); b2 had27.

The earlier short-battery numbers used a different evaluation protocol; they
must not be directly compared with the final seeded 60-second battery. The
release README and eval JSONs define that protocol. A no-push diagnostic on
b9_7000 survived99.32% for60 seconds, versus87.50% with pushes on the same seed,
motivating b11's increased recovery practice. Steering and motion quality remain
behind the ball-state reference even when balance is good.

Source checkpoint SHA256 (b9_6500):
`1e1ab1645aee340e721ff5383eb36e0547411dac7c8ab4ce71ebbd650a87ce37`.
Released checkpoint SHA256 (b11_6999):
`57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41`.

The public source port keeps the original basketball calculations and shared
walking configuration, while isolating the visual colourway from the unrelated
tug task. The default actor is now blind; the old reference remains available
with explicit `blind=False`. Use the continuation factory, not generic fresh
training defaults, to continue b11. Recorded YAML files describe the executed
run; source and lockfile provide the runnable implementation.

Artifacts and continuation instructions:
[Hugging Face](https://huggingface.co/HannesVonEssen/microduck-basketball) ·
[Experiment](README.md) ·
[Recurrent runtime PR #231](https://github.com/pollen-robotics/microduck/pull/231).

## Release validation (2026-09-07)

A fresh `uv sync --locked`, 16 CPU tests, and a 64-environment five-iteration
resume from the packaged b11 checkpoint passed. The port preserves the original
basketball finite-reward/±10 guard; it applies only to the basketball reward
stack. A normalized re-export matched PyTorch within 1.43e-6 over40 steps
including reset. A separate 1024-environment seed101/60-second rollout using
the public source survived991/1024 (96.78%), versus988/1024 in the original
battery. This repeat is reported separately; it is not added to the original
three-seed aggregate. Physics rollouts are not promised to reproduce bitwise.

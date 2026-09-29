# Raised-platform long jump

<video src="../media/long-jump.mp4" controls muted playsinline width="720"></video>

[Watch the long-jump clip](../media/long-jump.mp4).

[Policy and checkpoint](https://huggingface.co/HannesVonEssen/microduck-long-jump)
· [Parkour overview and reproduction](../README.md)
· [Continue training](../TRAINING.md)

This is `Mjlab-PlatformJump-MicroDuck`, not the flat-ground
`Mjlab-LongJump-MicroDuck` foundation. The robot starts near the edge of block A
and jumps over a gap to lower block B. The release profile pins a **30 cm gap
and 25 cm drop**; B is 10 cm above the floor. Floor contacts indicate a pit
failure. These are reproduction-profile dimensions, not measurements inferred
from the edited video.

The actor receives `[gap / 0.3, drop / 0.3, 0, 0, 0, 0]` in its six body-command
slots. It does not perceive or estimate the platforms. Use the exact joint
offsets and bounded-action convention in the HF `config.json`.

Selected checkpoint: p1, stored iteration 15,250. Its historical filename
`platformjump_p1_model_15000.pt` is misleading. No extra get-up policy is
included or switched in by the reproduction script. Landing and stabilization
are the jump policy's responsibility.

**Simulation only; not yet validated on hardware.** See the overview's montage-provenance caveat.

## Fresh release check — 2026-09-29

Fresh default-seed CPU render: crosses the gap, lands and remains standing on the target through 12 seconds. This single rollout is not a robustness estimate. See [verification](../VERIFICATION.md).

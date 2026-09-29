# Raised-platform backflip

<video src="../media/backflip-tight-v2.mp4" controls muted playsinline width="720"></video>

[Watch the backflip clip](../media/backflip-tight-v2.mp4).

**⚠️ Landings may break the robot.** Simulation only; not yet validated on hardware. Successful simulated recovery does not establish a safe physical landing.

[Policy and checkpoint](https://huggingface.co/HannesVonEssen/microduck-backflip)
· [Parkour overview and reproduction](../README.md)
· [Continue training](../TRAINING.md)

`Mjlab-Flip-MicroDuck`, retained f21 iteration-12,000 checkpoint. The release
profile specifies a **0.8 m drop measured from platform top to mat top**, not
to the floor. The mat is 0.44342 m thick, with a 1.90154 m × 0.605 m footprint.

The blue mat is not merely decorative: collision uses a soft-contact box with
MuJoCo `solref=(0.03, 1)` and `solimp=(0.9, 0.95, 0.01, 0.5, 2)`. This is a
numerical contact model, **not a calibrated physical crash mat or a model of
servo/structure damage**. A rigid-floor landing is outside the demonstrated
setup. The reproduction profile uses the collision box as its visible mat;
the sculpted visual mesh in the montage is not required for the physics.

The actor receives `[(drop_height - 0.7) / 0.2, direction_sign, 0, 0, 0, 0]`
in the body-command block. The backflip direction sign is set by the task.
Preparatory bounces are part of the learned behavior. This is not a
standing-height backflip, and there is no separate get-up policy in the
release reproduction script. Play starts on the platform, not in mid-air.

See the overview for exact commands, the observation/action contract and the
montage-provenance caveat. Earlier flip success figures were affected by a
mid-air-reset bug; they are not quoted as validation of this release.

## Fresh release check — 2026-09-29

Fresh default-seed CPU render: completes the backward rotation and recovers to standing, but later drifts off the mat and falls at 7.62 seconds. This is not a stable idle policy. This single rollout is not a robustness estimate. See [verification](../VERIFICATION.md).

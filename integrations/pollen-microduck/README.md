# Pollen runtime adapter for the swing policy

The released swing actor keeps the standard 61D MicroDuck observation shape,
but repurposes command slots `48..51` for a deployable swing-plane cue. The
official runtime does not currently construct that cue, so loading the ONNX
alone is not enough.

[`swing_plane_cue.rs`](swing_plane_cue.rs) is a dependency-free reference
adapter tested against the quaternion convention in
[`pollen-robotics/microduck@2c61dcc`](https://github.com/pollen-robotics/microduck/tree/2c61dcc1f03440541cdc0729f7a375b2a9ea3005): scalar-first `[w, x, y, z]`,
mapping trunk coordinates into world coordinates. Pinning the revision makes
the integration target auditable if the upstream controller changes.

## Integration

1. Copy `swing_plane_cue.rs` into `robotd/src/` and declare the module.
2. Wait until `SflpDecoder::ready()` is true **and the occupied swing is still
   at its bottom rest pose**, then call `SwingPlaneCue::capture(imu.quat)` once.
   Refuse to arm if capture returns `None`.
3. On every 50 Hz swing-policy tick, start from `Command::default()` and set:

   ```rust
   swing_command.twist = swing_plane_cue.command_twist(sensors.imu.quat);
   ```

   Pass `swing_command` to `Observation::build`. This writes exactly
   `[0, body_y.x, body_y.z]` into observation slots `48..51`; the other ten
   command slots remain zero.
4. Use the raw ONNX result as `last_action` for the next observation. The
   released graph already clamps its output to `[-1, 1]`.
5. Convert each clamped action to a joint target as
   `HOME + 0.7 * action`. Disable the official controller's head and leg
   low-pass filters for this skill: training used no post-policy filter.
6. Clear the cue anchor, previous action, and previous target whenever the
   skill is disarmed. Require a new stationary capture before rearming.

The following skeleton shows the policy-specific part of the controller path:

```rust
let mut command = Command::default();
command.twist = swing_plane_cue.command_twist(sensors.imu.quat);

let observation = Observation::build(
    &sensors.imu,
    &sensors.positions,
    &sensors.velocities,
    &DEFAULT_POSITION,
    &last_action,
    &command,
);
let action = policy.infer(&observation, Net::Swing)?;
last_action = action;

let offsets = Observation::scatter_action(&action);
let targets = std::array::from_fn(|joint| DEFAULT_POSITION[joint] + 0.7 * offsets[joint]);
```

`Net::Swing` is illustrative—the upstream revision has no swing skill variant.
The policy-selection, arming, motor limits, fall handling, and emergency-stop
path must be added by the integrator. This reference adapter has not been
merged into Pollen's runtime and has not been validated on hardware.

## Test independently

```bash
rustc --edition=2024 --test \
  integrations/pollen-microduck/swing_plane_cue.rs \
  -o /tmp/swing-plane-cue-test
/tmp/swing-plane-cue-test
```

The tests cover stationary capture, sagittal pitch invariance, yaw/roll signs,
world-frame invariance, and invalid IMU samples.

# Chimney climb release

Three 50 Hz policies for a MicroDuck entering a narrow vertical corridor,
climbing by bracing between the walls, and exiting onto a platform.
The main artifact is the **climb** policy. Enter and exit are companion skills.
This is simulation research, not hardware-validated autonomous climbing.

Models and the final user-selected video:
[`HannesVonEssen/microduck-chimney-climb`](https://huggingface.co/HannesVonEssen/microduck-chimney-climb).

## Artifacts and conventions

| Role | HF checkpoint | ONNX | Task |
|---|---|---|---|
| Main climb (w17) | `checkpoint.pt` | `policy.onnx` | `Mjlab-Chimney-MicroDuck` |
| Enter (a11) | `enter/checkpoint.pt` | `enter/policy.onnx` | `Mjlab-Approach-MicroDuck` |
| Exit (x19) | `exit/checkpoint.pt` | `exit/policy.onnx` | `Mjlab-Exit-MicroDuck` |

All take float32 `[1,61]` and return float32 `[1,14]`. Observation normalization
and joint-travel output bounds are baked into the ONNX by `scripts/export.py`.
No action smoothing is added. Joint targets are `offset + action * 1.0`;
**climb and exit use BRACE, not HOME**, while enter uses the standing offset.
Exact offsets, ordering and bounds are in each HF `config.json`.
Feed the bounded action back as the previous-action observation.

The first 48 observations are angular velocity(3), gravity(3), joint positions
relative to the active offset(14), joint velocities(14), previous actions(14).
The remaining13 are twist(3), head command(4), body command(6). Twist and head
commands are zero. The body command differs by policy:

- Climb: `[(width_m-.13)/.02, 0, 0, 0, 0, 0]`.
- Enter: `[dx_body, dy_body, sin(yaw), cos(yaw), (width_m-.13)/.02, 0]`,
  with dx/dy pointing to the corridor center, yaw relative to the across-slot
  heading. Requires corridor-relative localization, not only proprioception.
- Exit: `[goal_y-root_y, goal_z-root_z, goal_x-root_x, 0,
  (width_m-.13)/.02, 0]` in the training corridor frame. Requires geometry and
  relative position. `goal_z=platform_top+.115`.

The exit was trained toward positive y. The video uses it toward negative y
by reflecting observations, absolute joint targets, and action history;
`scripts/reproduce_chimney_chain.py` contains that adapter and the handoffs.
Do not simply hot-swap these files in an unmodified walking runtime.

## Reproduce the continuous chain

Use Linux with an NVIDIA GPU for mjlab/Warp. Install the project with `uv sync`.
Download the public release, then:

```bash
hf download HannesVonEssen/microduck-chimney-climb --local-dir policies/chimney
# Fetch the exact official alpha_stand revision from the release manifest.
uv run python scripts/fetch_chimney_getup.py policies/chimney/manifest.json policies/chimney/official_alpha_stand.onnx

MICRODUCK_EX_WIDTH=.125 MICRODUCK_EX_PLATFORM_TOP=3 \
MICRODUCK_CH_WIDTH=.125 MICRODUCK_AP_WIDTH=.125 MICRODUCK_CH_WALL_H=6 \
FULL_SECONDS=36 FULL_RES=96 FULL_SEED=29 SWEEP=1 \
GETUP_POLICY=policies/chimney/official_alpha_stand.onnx \
RECOVERY_CONTRACT=official RECOVERY_HISTORY=pose RECOVERY_Y=.60 \
uv run python scripts/reproduce_chimney_chain.py \
  policies/chimney/enter/checkpoint.pt policies/chimney/checkpoint.pt \
  policies/chimney/exit/checkpoint.pt chain.mp4
```

Use `MUJOCO_GL=egl` on a graphics-capable headless GPU or
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` with system OSMesa installed.
The diagnostic still constructs a renderer at96px for consistent initialization,
but saves `chain_trajectory.npz`, not a video, with `SWEEP=1`.
Rendering the saved trajectory (`REPLAY=...`, omit `SWEEP`) does not rerun physics.
`REPLAY_STRIDE=1 FULL_RES=720` renders native50fps; the selected camera uses the
six-second final orbit. The published MP4 is the user's supplied final edit,
not a freshly re-encoded render.

The official recovery actor is Pollen's **unchanged `alpha_stand.onnx`**, not an
additional policy trained here. Handoff initializes receiver action history
from its current pose relative to HOME, preserving physical state, contacts,
actuator state, and velocity. There is no teleport/reset to a standing pose.

The demonstration uses a12.5cm gap, a3m platform,6cm walls/platform thickness,
50cm platform width,45cm wall overlap and40cm open platform, walls ending40cm
above the platform. Geometry is specified in the reproduction script; the
training exit environment's default geometry is deliberately preserved.

## Verification and limitations

The selected seed29 chain switched enter→climb at1.74s, climb→exit at25.08s,
and exit→official recovery at27.80s. In the original36s recording, both feet
remain on the platform for the last3s, maximum tilt is0.725degrees and maximum
root speed is0.00632m/s. Two of four exploratory seeds completed the chain;
this is a selected successful rollout, not a measured robust success rate.
HF `eval/` includes the original frame/contact audit and ONNX parity checks.

A fresh seed29 run from this publication port reached the exit but did not
complete recovery. The command above reruns the controller sequence; it does
not guarantee the selected outcome from the seed alone. Replay the included
saved trajectory to inspect the exact demonstrated physical motion.

Retained modeling includes BAM actuator dynamics, delays, observation noise,
mass/friction/CoM randomization, bounded targets, and augmented robot collision
hulls. Real wall friction/compliance, collision geometry tolerances, current
and thermal loading, corridor-relative localization, and policy transitions
remain unvalidated. These are ordinary, not backlash-trained, policies.
Use a catch/support rig and remote torque-off for any future hardware work;
the3m demonstration is not a safe initial hardware experiment.

Export another checkpoint through `scripts/export.py --checkpoint-file ...`
with the matching task and `--num-envs 1`; do not hand-convert the actor.
`scripts/verify_chimney_release.py PACKAGE OUTPUT.json` compares each exported
actor to its bounded checkpoint actions on random and live observations.
PyTorch checkpoints use pickle: load only trusted files/revisions.

The long-jump modules are included solely because they supply the collision-
covered robot and base task configuration; this branch does not register or
publish a new long-jump policy. Historical task-specific MDP modules are kept
intact to preserve the trained behavior.

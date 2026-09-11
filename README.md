# Microduck Playground

[![CI](https://github.com/Vottivott/microduck-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/Vottivott/microduck-playground/actions/workflows/ci.yml)

<img src="docs/media/playground.png" align="right" width="320" alt="Watercolor illustration of Microduck pumping on its swing">

Reproducible reinforcement-learning experiments, policy demonstrations, and
printable hardware add-ons for
[Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck).

This is an independent experimental continuation of
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl),
not an official Pollen Robotics release. The playground is maintained as one
curated snapshot commit so the visible history stays focused on the release.
Its upstream base is
[`d424a0c`](https://github.com/pollen-robotics/microduck_rl/commit/d424a0c899f6b33cbd3daeb279913134349c0b63),
with attribution and source details retained in the repository. Add the
upstream project as a Git remote when preparing focused contributions.

<br clear="right">

## Experiments

Animated GIF previews play directly in the table. Click one—or use its explicit
play link—to open the browser-ready, silent MP4.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Preview</th>
      <th>Result and artifacts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Self-pumped swing</strong></td>
      <td>
        <a href="experiments/swing/media/alpha050_seed27.mp4?raw=1">
          <img src="experiments/swing/media/preview.gif" width="280" alt="Animated preview of Microduck pumping itself on a swing">
        </a>
      </td>
      <td>
        Starts still and reaches a 173.20° strict full span.<br>
        <a href="experiments/swing/media/alpha050_seed27.mp4?raw=1">Play MP4</a> ·
        <a href="experiments/swing/README.md">Experiment</a> ·
        <a href="integrations/pollen-microduck/README.md">Runtime adapter</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-swing">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Blind basketball balance</strong></td>
      <td><a href="experiments/basketball/media/preview.mp4?raw=1"><img src="experiments/basketball/media/preview.gif" width="280" alt="Animated preview of Microduck balancing and moving on a basketball"></a></td>
      <td>LSTM policy with no ball-state input: 97.01% survival over 60 seconds in 3,072 simulation trials. Experimental hardware-test candidate; hardware untested.<br>
      <a href="experiments/basketball/media/preview.mp4?raw=1">Play MP4</a> ·
      <a href="experiments/basketball/README.md">Experiment and training</a> ·
      <a href="https://huggingface.co/HannesVonEssen/microduck-basketball">ONNX and checkpoint</a> ·
      <a href="https://github.com/pollen-robotics/microduck/pull/231">LSTM runtime PR</a></td>
    </tr>
    <tr>
      <td><strong>Fast running</strong></td>
      <td>
        <a href="experiments/running/media/preview.mp4?raw=1">
          <img src="experiments/running/media/preview.gif" width="280" alt="Animated preview of Microduck running on flat ground">
        </a>
      </td>
      <td>
        Robustified iteration-12,195 simulation candidate: 1.651 m/s nominal,
        and 1.612 m/s under backlash plus disturbance stress.<br>
        <a href="experiments/running/media/preview.mp4?raw=1">Play MP4</a> ·
        <a href="experiments/running/README.md">Experiment</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-running">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Stilt walking</strong></td>
      <td>
        <a href="experiments/stilts/media/preview.mp4?raw=1">
          <img src="experiments/stilts/media/preview.gif" width="280" alt="Animated preview of Microduck walking on 10 cm stilts">
        </a>
      </td>
      <td>
        Eight height-specific blend-0.50 policies: 10, 15, 20, 25, 50,
        100, 140, and 200 cm stilts (10 cm shown).<br>
        <a href="experiments/stilts/media/preview.mp4?raw=1">Play MP4</a> ·
        <a href="experiments/stilts/README.md">Experiment</a> ·
        <a href="hardware/stilts/README.md">Hardware</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-stilts">Policies and videos</a>
      </td>
    </tr>
  </tbody>
</table>

Each preview is a direct simulation demonstration of the policy linked in its
row. Compact machine-readable evaluation records live beside each experiment.

## Hardware galleries

The retained swing seat keeps the battery centered without occupying the
head-and-leg pumping corridors. It includes compliant locating pads, a padded
strap, and a removable buckle. The source generators, printable millimetre
meshes, MuJoCo collision hulls, and clearance reports are under
[`hardware/swing-seat`](hardware/swing-seat/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/swing-seat/renders/seat_front.png" width="300" alt="Retained swing seat, front view"><br><sub>Front</sub></td>
    <td align="center"><img src="hardware/swing-seat/renders/seat_three_quarter.png" width="300" alt="Retained swing seat, three-quarter view"><br><sub>Three-quarter</sub></td>
    <td align="center"><img src="hardware/swing-seat/renders/seat_side.png" width="300" alt="Retained swing seat, side view"><br><sub>Side</sub></td>
  </tr>
</table>

The stilt system replaces the removable soles and preserves explicit tip
contact geometry. The gallery uses the demonstrated green 10 cm blend-0.50
configuration. Parametric generators and printable meshes are under
[`hardware/stilts`](hardware/stilts/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/stilts/renders/stilts_front.png" width="230" alt="Microduck green stilts, front view"><br><sub>Front</sub></td>
    <td align="center"><img src="hardware/stilts/renders/stilts_three_quarter.png" width="230" alt="Microduck green stilts, three-quarter view"><br><sub>Three-quarter</sub></td>
    <td align="center"><img src="hardware/stilts/renders/stilts_side.png" width="230" alt="Microduck green stilts, side view"><br><sub>Side</sub></td>
    <td align="center"><img src="hardware/stilts/renders/printed_stilt.jpg" width="230" alt="Green 3D-printed Microduck replacement sole and stilt prototype"><br><sub>3D-printed prototype</sub></td>
  </tr>
</table>

## Quick start

A CUDA GPU and [`uv`](https://docs.astral.sh/uv/) are recommended. Training
uses MuJoCo Warp through `mjlab`.

See [`CONTINUING_TRAINING.md`](CONTINUING_TRAINING.md) for the artifact to
download, the level of training state preserved, and the exact continuation
guide for each released policy.

```bash
git clone https://github.com/Vottivott/microduck-playground
cd microduck-playground
uv sync

# Cheap configuration/training smoke test first.
uv run train Mjlab-SwingPump-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5

# Full swing training configuration.
uv run train Mjlab-SwingPump-MicroDuck \
  --env.scene.num-envs 4096
```

The default swing run stops at the preserved 3,600-update endpoint. The
released actor is then selected by the documented alpha-0.50 endpoint
interpolation in [`experiments/swing/TRAINING.md`](experiments/swing/TRAINING.md).

## Repository layout

```text
experiments/
  basketball/            blind LSTM policy, evaluation, continuation guide
  running/               clean policy preview and result summary
  stilts/                policy index, executed curriculum, continuation guide
  swing/                 selected checkpoints, evaluation, media, methodology
hardware/
  stilts/                parametric stilt generator and printable meshes
  swing-seat/            retained-seat generator, meshes, clearance reports
src/mjlab_microduck/     tasks, robot models, actuator model, rewards
scripts/                 evaluation, export, rendering, and selection tools
integrations/            policy-specific deployment adapters
tests/                   CPU configuration and invariant tests
docs/                    supporting research and training notes
```

## Scope and safety

These are simulation experiments, not hardware safety certifications. The
swing model simulates two elastic tension-only cords and randomized actuator
and sensor dynamics, but real cord knots, frame flex, textile contact, servo
temperature, and assembly tolerances remain. Extreme-height stilts require an
engineered load path and fall protection. Use a safety tether, current limits,
an emergency stop, a clear exclusion zone, and conservative incremental tests.

## Upstream and contributions

To compare against or prepare a focused pull request for Pollen's project:

```bash
git remote add upstream https://github.com/pollen-robotics/microduck_rl.git
git fetch upstream
```

## License

Software is licensed under Apache-2.0; see [`LICENSE`](LICENSE). As in the
upstream project, 3D hardware design files are licensed under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International; see
[`LICENSE-HARDWARE`](LICENSE-HARDWARE). Third-party Microduck assets retain
their original attribution and terms. See [`NOTICE`](NOTICE) and the
hardware-specific READMEs.

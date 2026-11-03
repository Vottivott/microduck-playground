# Microduck Playground

[![CI](https://github.com/Vottivott/microduck-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/Vottivott/microduck-playground/actions/workflows/ci.yml)

<img src="docs/media/playground.png" align="right" width="320" alt="Watercolor illustration of Microduck pumping on its swing">

Reproducible reinforcement-learning experiments, policy demonstrations, and
printable hardware add-ons for
[Pollen Robotics' Microduck](https://github.com/pollen-robotics/microduck).

This is an independent experimental continuation of
[`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl),
not an official Pollen Robotics release. For the initial public release, all
playground-specific work is consolidated into one commit on top of upstream
commit [`d424a0c`](https://github.com/pollen-robotics/microduck_rl/commit/d424a0c899f6b33cbd3daeb279913134349c0b63),
preserving the original project history and attribution. Development after the
release uses ordinary commits. The upstream project can be added as a Git
remote when preparing focused contributions.

<br clear="right">

## Experiments

Animated previews play directly in the table. Click one—or use its explicit
full-video link—to open the complete video.

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
        <a href="experiments/swing/media/alpha050_seed27.mp4">
          <img src="experiments/swing/media/preview.gif" width="280" alt="Animated preview of Microduck pumping itself on a swing">
        </a>
      </td>
      <td>
        Starts still and reaches a 173.20° strict full span.<br>
        <a href="experiments/swing/media/alpha050_seed27.mp4">Full video</a> ·
        <a href="experiments/swing/README.md">Experiment</a> ·
        <a href="integrations/pollen-microduck/README.md">Runtime adapter</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-swing">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Blind basketball balance</strong></td>
      <td><a href="experiments/basketball/media/preview.mp4"><img src="experiments/basketball/media/preview.gif" width="280" alt="Microduck balancing and moving on a basketball"></a></td>
      <td>LSTM policy with no ball-state input: 97.01% survival over 60 seconds in 3,072 simulation trials. Experimental hardware-test candidate; hardware untested.<br>
      <a href="experiments/basketball/media/preview.mp4">Full video</a> ·
      <a href="experiments/basketball/README.md">Experiment and training</a> ·
      <a href="https://huggingface.co/HannesVonEssen/microduck-basketball">ONNX and checkpoint</a> ·
      <a href="https://github.com/pollen-robotics/microduck/pull/231">LSTM runtime PR</a></td>
    </tr>
    <tr>
      <td><strong>Fast running</strong></td>
      <td>
        <a href="experiments/running/media/preview.mp4">
          <img src="experiments/running/media/preview.gif" width="280" alt="Animated preview of Microduck running on flat ground">
        </a>
      </td>
      <td>
        Robustified iteration-12,195 simulation candidate: 1.651 m/s nominal,
        and 1.612 m/s under backlash plus disturbance stress.<br>
        <a href="experiments/running/media/preview.mp4">Full video</a> ·
        <a href="experiments/running/README.md">Experiment</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-running">ONNX on Hugging Face</a>
      </td>
    </tr>
    <tr>
      <td><strong>Stilt walking</strong></td>
      <td>
        <a href="experiments/stilts/media/preview.mp4">
          <img src="experiments/stilts/media/preview.gif" width="280" alt="Animated preview of Microduck walking on green 10 cm stilts">
        </a>
      </td>
      <td>
        Blend-0.50 policies for 10, 15, 20, 25, and 50 cm, plus
        1.0, 1.4, and 2.0 m simulation stilts (10 cm shown).<br>
        <a href="experiments/stilts/media/preview.mp4">Full video</a> ·
        <a href="experiments/stilts/README.md">Experiment</a> ·
        <a href="hardware/stilts/README.md">Hardware</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-stilts">Policies and videos</a>
      </td>
    </tr>
  <tr>
    <td><strong>Climbing</strong></td>
    <td><a href="experiments/desk-climb/media/preview.mp4"><img src="experiments/desk-climb/media/preview.gif" width="280" alt="Cream Microduck climbing a modular ladder onto a desk and standing up"></a></td>
    <td>Climb, land, recover.<br>
    <a href="experiments/desk-climb/README.md">Experiment and training</a> · <a href="hardware/ladder/README.md">Ladder and mounts</a> · <a href="https://huggingface.co/HannesVonEssen/microduck-climb">Models</a></td>
  </tr>
    <tr>
      <td><strong>Chimney climbing</strong></td>
      <td>
        <a href="experiments/chimney-climb/media/preview.mp4">
          <img src="experiments/chimney-climb/media/preview.gif" width="280" alt="Animated preview of Microduck entering a narrow corridor, climbing to a platform, exiting and standing up">
        </a>
      </td>
      <td>
        Braced climbing through a 12.5 cm gap to a 3 m platform.
        Selected simulation rollout; hardware untested.<br>
        <a href="experiments/chimney-climb/media/preview.mp4">Full video</a> ·
        <a href="experiments/chimney-climb/README.md">Experiment and training</a> ·
        <a href="https://huggingface.co/HannesVonEssen/microduck-chimney-climb">Climb, enter and exit policies</a>
      </td>
    </tr>
    <tr><td><strong>Raised-platform long jump</strong></td><td><a href="experiments/parkour/media/long-jump.mp4"><img src="experiments/parkour/media/long-jump.gif" width="280" alt="Raised-platform long jump in simulation"></a></td><td>Simulation only; not yet validated on hardware.<br><a href="experiments/parkour/long-jump/README.md">Details and reproduction</a> · <a href="https://huggingface.co/HannesVonEssen/microduck-long-jump">Policy and training checkpoint</a></td></tr>
    <tr><td><strong>Raised-platform backflip</strong></td><td><a href="experiments/parkour/media/backflip-tight-v2.mp4"><img src="experiments/parkour/media/backflip-tight-v2.gif" width="280" alt="Raised-platform backflip in simulation"></a></td><td>Experimental simulation policy; hardware unvalidated. ⚠️ Landings may break the robot.<br><a href="experiments/parkour/backflip/README.md">Details and reproduction</a> · <a href="https://huggingface.co/HannesVonEssen/microduck-backflip">Policy and training checkpoint</a></td></tr>
    <tr>
      <td><strong>Self-charging dock</strong></td>
      <td>
        <a href="hardware/microdock/media/preview.mp4">
          <img src="hardware/microdock/media/preview.gif" width="280" alt="Animated preview of Microduck turning around and backing into a printed dock until the magnetic tip in its USB-C port meets the charging adapter">
        </a>
      </td>
      <td>
        Backs into a fully printed dock that connects a magnetic charger to its own USB-C port:
        30/30 docked in simulation with the official walking policy. Not yet printed or tried on hardware.<br>
        <a href="https://x.com/HannesVonEssen/status/2117688155600007338">Full video</a> ·
        <a href="hardware/microdock/README.md">Hardware and simulation</a> ·
        <a href="hardware/microdock/print/microdock_P1S.3mf">Bambu P1S project (3 plates)</a>
        (<a href="hardware/microdock/print/README.md">how to print</a>)
      </td>
    </tr>
  </tbody>
</table>

Previews illustrate simulation behaviors. The jump/backflip excerpts are selected research footage; see their documented fresh-checkpoint verification and original footage provenance limits. Compact machine-readable evaluation records live beside each experiment.

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

The modular climbing ladder has pin-free stacking joints, a central spine, a
matching floor base and printed tightening screws for the table mounts.
Printable models, generators, assembly instructions and the full gallery are under
[`hardware/ladder`](hardware/ladder/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/ladder/renders/friction-exploded.png" width="300" alt="Stackable ladder modules and friction-fit joints"><br><sub>Stacking joints</sub></td>
    <td align="center"><img src="hardware/ladder/renders/friction-floor-assembled.png" width="300" alt="Modular ladder floor base"><br><sub>Floor base</sub></td>
    <td align="center"><img src="hardware/ladder/renders/friction-top-module.png" width="300" alt="Top module with threaded table mounts"><br><sub>Table mounts</sub></td>
  </tr>
</table>

The MicroDock charging dock prints without screws or inserts. Microduck backs in
until the magnetic tip in its own USB-C port meets a magnetic adapter in a
floating hood, which lines the robot up and rides clear of the power button.
The generator, printable meshes, docking simulation and renders are under
[`hardware/microdock`](hardware/microdock/README.md), with a ready-to-print Bambu Studio project for a Bambu Lab
P1S in [`hardware/microdock/print`](hardware/microdock/print/README.md).

<table>
  <tr>
    <td align="center"><img src="hardware/microdock/renders/microdock_three_quarter.png" width="300" alt="MicroDock assembled: base plate, board with AprilTag and floating hood"><br><sub>Assembled</sub></td>
    <td align="center"><img src="hardware/microdock/renders/microdock_parts.png" width="300" alt="MicroDock printed parts pulled apart: base, board, tag, hood and keeper"><br><sub>Printed parts</sub></td>
    <td align="center"><img src="hardware/microdock/renders/microdock_docked.png" width="300" alt="Microduck docked, with the hood resting on its back"><br><sub>Docked</sub></td>
  </tr>
</table>

## Quick start

A CUDA GPU and [`uv`](https://docs.astral.sh/uv/) are recommended. Training
uses MuJoCo Warp through `mjlab`.

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

## Repository layout

```text
experiments/
  parkour/               raised-platform long jump and backflip
  basketball/            blind LSTM policy, evaluation, continuation guide
  running/               clean policy preview and result summary
  stilts/                policy index, executed curriculum, continuation guide
  swing/                 selected checkpoints, evaluation, media, methodology
hardware/
  microdock/             printable charging dock, docking simulation, renders
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

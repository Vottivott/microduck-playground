# MicroDock v2: the seated dock

The [MicroDock](../microdock/README.md), but Microduck **sits down to charge**. It walks up to the dock's AprilTag and
turns around, as before. Then it sits with the official sit policy and **butt-walks backward** until the magnetic tip in
its USB-C port meets the adapter in the dock's floating hood. It stops when USB power appears and rests seated while it
charges.

Why sitting is better while it charges:
- **Low power:** seated, the legs carry almost no weight, so the servos draw little. A standing duck holds its weight
  with its legs and keeps balancing the whole time, and all of that comes out of the charge.
- **No heat or wear:** a duck that charges for hours doesn't spend them holding a stance under load.
- **Safe:** a seated duck can't fall over while it rests.

**Two new parts:**
- **The board:** it holds the hood 64 mm lower, at the seated duck's height, and lets it float further sideways and
  down.
- **The hood:** it holds the adapter 2.3 mm lower.

The base, keeper and tag are the MicroDock's.

| Assembled | Printed parts | Docked |
|---|---|---|
| ![](renders/microdock_three_quarter.png) | ![](renders/microdock_parts.png) | ![](renders/microdock_docked.png) |

[![Microduck sitting down and butt-walking back into the MicroDock v2 in simulation](media/preview.gif)](media/preview.mp4)

**Status:** designed and tested in simulation only, like the MicroDock; not yet printed or tried on hardware.
- **The butt walk:** a new policy, trained for this dock in simulation (see [`training/`](training/README.md)). It has
  not run on a robot yet.
- **The docking behaviour:** tag detection, the turn, the sit and stopping on USB power are scripted in the simulation,
  not implemented on the robot.

## How docking works

1. **Find the dock and turn around:** as on the MicroDock. The duck walks toward the tag and turns in place about 200 mm
   from the board. It turns 6.6° past straight, because the sit turns it back by about that much.
2. **Sit:** the official `alpha_sitstand` policy sits the duck down. It moves back about 63 mm while sitting.
3. **Butt-walk back:** the butt-walk policy scoots backward on its seat at about 2.5 cm/s, holding its heading, for the
   last 6–9 cm. On the way in, as on the MicroDock:
   - the hood's flared side walls line the duck up;
   - the hood rides up onto its back, floating up to 20 mm sideways;
   - the magnetic tip meets the adapter, and the magnet closes the last millimetre.
4. **Rest:** when USB power appears, it stops (the all-zero command) and sits still while it charges.

The duck backs in leaning about 2–3° forward and settles at about 7°. The v2 hood holds the adapter 2.3 mm lower than the
MicroDock's, which keeps the tip on it in both.

## Simulation result

- **Policies:** official `velstand.onnx` (walk) and `alpha_sitstand.onnx` (sit), unmodified, and `butt_walk.onnx`, all on
  the BAM XL330 actuator model (`microduck_rl` develop `bae39f59`).
- **Starts:** 380–460 mm out, ±40 mm sideways, heading 180 ± 12°. The tag estimate is the true position plus noise,
  σ = 5 mm along, 3 mm across and 1°.
- **Seat friction:** 0.45, a plastic shell on PLA. The butt walk trained with 0.3–1.0.

| Batch | Docked (USB connected and holding while it rests) | Hood touched the power button |
|---|---|---|
| [`results/seated30.json`](results/seated30.json) (seeds 41, 42, 43 × 10) | **30 / 30** | 0 runs |
| [`results/seat_friction.json`](results/seat_friction.json): seat friction 0.3 / 0.8 (10 each) | 9 / 10, 10 / 10 | 0 runs |

- **Accuracy:** every docked run ended within 0.2 mm of the adapter along the push, 0.9 mm sideways and 0.8 mm
  vertically.
- **Spread after the sit:** the duck sat −18 to +14 mm off-centre and −6° to +8.5° off; the flares and the floating hood
  absorbed it.
- **The miss:** at friction 0.3, the duck sat 30 mm off-centre, past what the hood can catch.

## Parts and printing

Print the MicroDock's base, keeper and tag ([`../microdock/stl`](../microdock/stl)) and the two v2 parts:
[`stl/microdock_v2_board.stl`](stl/microdock_v2_board.stl) and [`stl/microdock_v2_hood.stl`](stl/microdock_v2_hood.stl).
If you already printed a MicroDock, print only these two.

For a Bambu Lab P1S, [`print/microdock_v2_P1S.3mf`](print/microdock_v2_P1S.3mf) is a ready project with all three plates.
It is set up like the MicroDock's ([how to print](../microdock/print/README.md)), with the v2 board and hood on plate 2.
Bambu Studio's estimates:

| Plate | Time | PLA |
|---|---|---|
| 1 | 2 h 9 min | 101 g |
| 2 | 3 h 1 min | 130 g |
| 3 | 22 min | 9 g |

That's about 5 h 32 min and 240 g in all.

The bill of materials and the assembly are the MicroDock's ([BOM](../microdock/README.md#bill-of-materials),
[assembly](../microdock/README.md#assembly)), with the v2 board and hood.

### What the simulation leaves out

The simulation holds the dock still and moves the hood on two sliders, so these were checked separately:
- **The keeper:** it keeps the hood on the board. The hood leans forward on its pegs, and undocking pulls it forward
  through the magnet. `keeper_fit` in `../microdock/source/cad.py` checks every hood position (`build.py --check`
  runs it): at least 274 mm² of the keeper always bears on the board. Two filament pins lock it on the pegs.
- **Re-centring:** the V-seats under the pegs (about 10°) are too shallow to centre the hood against PLA's friction,
  so after an off-centre docking it stays where it was left. With the simulator's centring spring removed and the hood
  starting off-centre (A_1400, z0 62): ±10 mm docked 30/30 each, −20 mm 29/30, +20 mm 27/30.
- **Holding the dock in place:** backing in pushes the dock back by up to about 3 N (peaks to 6 N) and scooting out
  pulls it forward by up to about 3 N, with the duck's weight only partly on the plate. A free-standing 240 g dock
  would slide on a smooth floor: stand it against a wall and fix the base down (assembly step 8).
- **The cable loop:** as the hood floats, its cable exit moves 39 to 66 mm from the board's clip. Leave about 11 cm.
- **The hood's weight:** the print with its adapter and cable head is about 45 g, the simulator's hood 28 g. With
  45 g it still docked 30/30.

## Reproduce

The parts: the board, the hood, the print plates, and the checks that the cable head fits the hood and its catch and
that the other parts are the MicroDock's.

```bash
uv run hardware/microdock-v2/source/build.py --check
uv run hardware/microdock-v2/source/build.py --bambu /path/to/bambu-studio --check   # also the Bambu Studio project
```

The simulation needs the same checkout and official policies as the MicroDock's. The butt walk is in `policy/`.

```bash
export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
uv run python hardware/microdock-v2/sim/docking_seated.py --seeds 41,42,43 --n 10 --out results.json
# the showcased run, then the preview video and the gallery (the MicroDock's renderers)
uv run python hardware/microdock-v2/sim/docking_seated.py --n 1 --seed 0 --noise 0 --start 450,15,178 --hold 4 --save run.pkl
MUJOCO_GL=egl uv run --with manifold3d python hardware/microdock/sim/render_preview.py run.pkl preview.mp4 --end 0.06,0.0,0.085,0.62
MUJOCO_GL=egl uv run --with manifold3d python hardware/microdock/sim/render_gallery.py run.pkl hardware/microdock-v2/renders
```

## Files

- `source/seated.py`: the seated parameters, the MicroDock's with four changes:
  - the hood lower;
  - more sideways float;
  - more drop;
  - the adapter lower in the hood.
- `source/build.py`: the board, the hood, the print plates and the Bambu Studio project.
- `stl/`, `print/`: the v2 board and hood, and the P1S plates with them.
- `sim/docking_seated.py`: the docking trials (walk, turn, sit, butt walk, connect, rest).
- `sim/bw_eval.py`: the butt walk's command battery (speed, turning, stillness, tilt, head).
- `policy/butt_walk.onnx`: the butt walk (61 inputs, 14 outputs, the observation normalizer baked in).
- `training/`: the environment it was trained in, and how.
- `results/`, `media/`, `renders/`: the trial results, the preview and the gallery.

## License

As the MicroDock:
- **Hardware:** the printable meshes in `stl/` and `print/` are 3D hardware design files, licensed
  [CC BY-NC-SA 4.0](../../LICENSE-HARDWARE).
- **Software:** the scripts, the training environment and the policy are under the repository's
  [Apache-2.0](../../LICENSE) license.

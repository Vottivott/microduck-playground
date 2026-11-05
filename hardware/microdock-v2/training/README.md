# The butt walks: seated locomotion on command

A butt walk makes a seated Microduck scoot on its seat on command. The MicroDock v2 uses two of them:
- **`../policy/butt_walk.onnx` (docking):** backs the duck into the dock after it sits down, and holds it still there.
- **`../policy/butt_walk_undock.onnx` (undocking):** scoots it forward out of the dock against the magnet, for the
  official sit policy to stand it up.

An experimental third, `../policy/experimental/butt_walk_dock_undock.onnx`, does both (see
[below](#experimental-one-policy-for-both)).

Each is an ordinary member of the policy family:
- **Inputs:** the 61D observation (48 proprioception + the 13D command block). The twist slot is `[vx, vy, wz]` in the
  trunk frame; the head pose slot works as in the other policies; the body pose slot is zero.
- **Outputs:** 14 unfiltered actions on the BAM XL330 model, at 50 Hz.
- **Hand-over:** each starts from the pose the official `alpha_sitstand` policy sits in, so the runtime can hand over
  to it after a sit, and back.

## The environment

`microduck_butt_walk_env_cfg.py` builds on the sitstand environment of
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (develop, `bae39f59`). That gives it the
same robot model, seated contact settings, domain randomization, observation noise and delays. Every episode starts
seated.

**Command:** the velocity twist, biased backward: vx −0.10 to +0.06 m/s, wz ±0.3 rad/s, and an exact zero for 20% of the
commands.

**Rewards:**
- **Tracking:** Gaussian velocity and turn-rate tracking, plus a linear, capped progress term along vx and wz. At
  docking speeds (~0.05 m/s) the Gaussians are flat near zero motion: a duck that never moves earns most of them, and
  the first runs did exactly that. Progress pays for any motion the right way, and nothing beyond the commanded speed.
- **Seated band:** a height Gaussian at 65 mm, an L1 cost above 80 mm, and termination above 95 mm. Standing up and
  walking is the easy way to move.
- **Upright:** a Gaussian on the trunk's tilt. It keeps the seat square and repeatable, so the dock's hood, which meets
  the trunk in the trunk frame, lines the magnetic tip up with the adapter.
- **Head up:** the head-pose tracking from sitstand, plus a linear price on the error and termination beyond 0.8 rad.
  Without them, the policy bowed its head ~85° forward to put its weight on its feet and unload the seat.
- **Still at zero:** stillness at the zero command, which is both how it stops at the dock and how it rests there.

**Seat friction:** the seat slides on the floor all the time, and a plastic shell on PLA or wood is ~0.3–0.5. The
environment names the seat's geoms (the export leaves them unnamed) and randomizes their friction at every reset. It
starts slippery (0.05–0.25) and ramps to the full range by iteration 1500.

**Options:** environment variables switch these on; all are off by default. The settings of each policy are below.
- **`MICRODUCK_BW_HIP_LIMIT`:** a cost on the hip rolls and yaws within 0.15 rad of their hard stops.
  - Unpriced, the first policy parked its hips on their stops, with the servos pushing at their current limit even
    sitting still: about 18 W of motor heat in the CPU rehearsal.
  - With the cost, the hips stay off their stops and the duck sits level, 61–64 mm high.
  - That is why the dock holds the hood at `z0` 62.
- **`MICRODUCK_BW_UNDOCK`:** this share of the episodes starts docked at the MicroDock:
  - the magnet holds the port;
  - the hood rests over the back. It is a ceiling, and a stop that holds the heading until the duck is about 20 mm
    out (`MICRODUCK_BW_UNDOCK_YAW_K`).
  - The command is forward for the first 5 s.
  - The magnet's pull is drawn per episode from a range that grows over training (`MICRODUCK_BW_UNDOCK_F`,
    `iteration:lo-hi,...` in newtons).
  - `MICRODUCK_BW_UNDOCK_PROGRESS` pays for net progress out of the dock. It is potential-based, and a terminated
    episode pays it back, so lunging out and falling earns nothing.
  - `MICRODUCK_BW_UNDOCK_CRITIC=1` shows the magnet to the critic only; the policy has to feel it.
- **`MICRODUCK_BW_STRAIGHT`, `MICRODUCK_BW_YAW_L1`:** a share of straight commands (wz 0), and an L1 cost on the
  turn-rate error. Without them, a policy that also undocks spun while backing up (0.3–0.65 rad/s at wz 0) and missed
  the dock.
- **`MICRODUCK_BW_CURRICULUM_CAP`:** drops the inherited sitstand curricula from this iteration on. They harden at
  1500 (head-pose range, pushes, action rate, rise speed), and every butt-walk run collapsed there.
- **`MICRODUCK_BW_SEAT_MU`, `MICRODUCK_BW_HEAD_TERM`:** the seat friction range (default 0.3–1.0) and the head
  termination (rad).

## Train it

Apply `butt_walk.patch` to a checkout of microduck_rl at `bae39f59`. It adds:
- the environment;
- its helpers in `mdp.py` (readable here as `mdp_butt_walk.py`);
- the straight-command option of `VelocityCommandCommandOnly`;
- the task's registration (`Mjlab-ButtWalk-Flat-MicroDuck`);
- its config tests.

```bash
git -C microduck_rl checkout bae39f59 && git -C microduck_rl apply /path/to/hardware/microdock-v2/training/butt_walk.patch
cd microduck_rl
uv run train Mjlab-ButtWalk-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5     # smoke test
MICRODUCK_BW_HIP_LIMIT=3 uv run train Mjlab-ButtWalk-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 2000
uv run scripts/export.py Mjlab-ButtWalk-Flat-MicroDuck --checkpoint-file logs/rsl_rl/microduck_butt_walk/<run>/model_1400.pt --onnx-file butt_walk.onnx
```

## The released policies

All three were trained from scratch with 4096 environments on one H200, at about 30 iterations a minute. Each was picked
from its run's checkpoints with the docking simulation (`../sim`).

### Docking: `butt_walk.onnx`

Iteration 1400 of a 2000-iteration run with `MICRODUCK_BW_HIP_LIMIT=3`. The later checkpoints regressed after the
inherited curricula hardened at 1500.

`../sim/bw_eval.py` runs a command battery ([`../results/butt_walk_commands.json`](../results/butt_walk_commands.json)) in
the CPU MuJoCo docking scene:
- BAM actuators and the runtime's observation pipeline;
- seat friction 0.45;
- before each command, the official sitstand policy sits the duck down and hands over.

| Command | What it does |
|---|---|
| vx −0.05 m/s | backs up at 0.040 m/s |
| vx −0.08 m/s | backs up at 0.046 m/s |
| vx −0.05 m/s, wz ±0.2 rad/s | turns at +0.46 / −0.57 rad/s |
| all zero | sits still |
| vx +0.05 m/s | scoots forward at 0.043 m/s |

- **Trunk:** level, about 2° forward while it backs up.
- **Head:** at the commanded pose, its lowest point 80 mm above the floor.
- **Docking:** with the MicroDock v2 it docked 30/30 (see the dock's [README](../README.md)).
- **The magnet:** it can't break free of the 4 N magnet; that is the undocking policy's job.

### Undocking: `butt_walk_undock.onnx`

Iteration 1100 of a run where 30 % of the episodes start docked:

```
MICRODUCK_BW_HIP_LIMIT=3 MICRODUCK_BW_SEAT_MU=0.25,0.7 MICRODUCK_BW_HEAD_TERM=1.2
MICRODUCK_BW_UNDOCK=0.3 MICRODUCK_BW_UNDOCK_PROGRESS=1.0 MICRODUCK_BW_UNDOCK_CRITIC=1
MICRODUCK_BW_UNDOCK_F=0:0.5-1.5,400:0.5-2.5,800:1-3.5,1200:1.5-4.5,1600:2-6
```

It was trained before the hood's yaw stop existed: add `MICRODUCK_BW_UNDOCK_YAW_K=0` to train it again.

With the MicroDock v2 at a 4 N magnet it undocked 30/30:
- it breaks free 1.06 s into the scoot (median);
- it is 70 mm out after about 2 s;
- then the official policies stand it up and walk it off.

It only breaks free with the training's latencies in the simulation, as `../sim/latency.py` adds them: it pushes in the
rhythm the delays set. Without them it gets free once in ten.

### Experimental: one policy for both

`../policy/experimental/butt_walk_dock_undock.onnx` both backs in and scoots out; only the command's sign differs. It is
iteration 2699 of a 2700-iteration run with the undocking settings above, plus:

```
MICRODUCK_BW_CURRICULUM_CAP=1499 MICRODUCK_BW_STRAIGHT=0.3 MICRODUCK_BW_YAW_L1=1.0
```

The hood's yaw stop is on, as by default.

With the training's latencies it docked and undocked 30/30 at 4 N and 10/10 at 6 N
([`../results/experimental_dock_undock.json`](../results/experimental_dock_undock.json)). It backs up straight (turn-rate
drift 0.003 rad/s at wz 0).

It is experimental because:
- the training environment's own check still reports most undocking episodes ending early, which contradicts these
  results and is not understood yet;
- it was evaluated less than the two released policies.

One policy for both would let a duck that docked crooked pull forward and try again without switching policies.

These are simulation results: none of the butt walks has run on a robot. Before trying one, check these on a real unit:
- whether the seat slides on your floor (friction 0.3–1.0 was trained);
- the hand-over from the sit;
- for the undocking, the real magnet's pull and the robot's latency.

# The butt walk: seated locomotion on command

`../policy/butt_walk.onnx` makes a seated Microduck scoot on its seat: backward, forward, turning, and still at the
all-zero command. The MicroDock v2 uses it to back into the dock after sitting down.

It is an ordinary member of the policy family:
- **Inputs:** the 61D observation (48 proprioception + the 13D command block). The twist slot is `[vx, vy, wz]` in the
  trunk frame; the head pose slot works as in the other policies; the body pose slot is zero.
- **Outputs:** 14 unfiltered actions on the BAM XL330 model, at 50 Hz.
- **Hand-over:** it starts from the pose the official `alpha_sitstand` policy sits in, so the runtime can hand over to it
  after a sit, and back.

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
  the trunk in the trunk frame, lines the magnetic tip up with the adapter. Linear tilt costs, toward upright or toward
  a fixed lean, were tried and dropped: the stroke rocks the trunk to unload the seat, and pulling on its tilt killed it.
- **Head up:** the head-pose tracking from sitstand, plus a linear price on the error and termination beyond 0.8 rad.
  Without them, the policy bowed its head ~85° forward to put its weight on its feet and unload the seat, with the camera
  an inch off the floor.
- **Still at zero:** stillness at the zero command, which is both how it stops at the dock and how it rests there.

**Seat friction:** the model's collision geoms default to friction 1.0, but the seat slides on the floor all the time, and
a plastic shell on PLA or wood is ~0.3–0.5. The environment names the seat's geoms (the export leaves them unnamed) and
randomizes their friction at every reset. It starts slippery (0.05–0.25) and ramps to 0.3–1.0 by iteration 1500.

## Train it

Add the environment to a checkout of microduck_rl at `bae39f59`:
- `microduck_butt_walk_env_cfg.py` goes to `src/mjlab_microduck/tasks/`;
- the helpers in `mdp_butt_walk.py` go at the end of `src/mjlab_microduck/tasks/mdp.py`;
- register the task in `src/mjlab_microduck/tasks/__init__.py`:

```python
from .microduck_butt_walk_env_cfg import make_microduck_butt_walk_env_cfg, MicroduckButtWalkRlCfg

register_mjlab_task(
    task_id="Mjlab-ButtWalk-Flat-MicroDuck",
    env_cfg=make_microduck_butt_walk_env_cfg(),
    play_env_cfg=make_microduck_butt_walk_env_cfg(play=True),
    rl_cfg=MicroduckButtWalkRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

Then:

```bash
uv run train Mjlab-ButtWalk-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5     # smoke test
uv run train Mjlab-ButtWalk-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 1400
uv run scripts/export.py Mjlab-ButtWalk-Flat-MicroDuck --checkpoint-file logs/rsl_rl/microduck_butt_walk/<run>/model_1400.pt --onnx-file butt_walk.onnx
```

## The released policy

`../policy/butt_walk.onnx` is iteration 1400 of that run: 4096 environments on one H200, about 40 minutes. It was picked
over later checkpoints and other variants by `../sim/bw_eval.py` and the docking trials.

`../sim/bw_eval.py` runs a command battery ([`../results/butt_walk_commands.json`](../results/butt_walk_commands.json)) in
the CPU MuJoCo docking scene:
- BAM actuators and the runtime's observation pipeline;
- seat friction 0.45;
- before each command, the official sitstand policy sits the duck down and hands over.

| Command | What it does |
|---|---|
| vx −0.05 m/s | backs up at 0.025 m/s |
| vx −0.08 m/s | backs up at 0.032 m/s |
| vx −0.05 m/s, wz ±0.2 rad/s | turns at +0.55 / −0.47 rad/s, faster than asked |
| all zero | sits still |
| vx +0.05 m/s | barely moves (+0.005 m/s); the dock only needs backward |

- **Head:** at the commanded pose, its lowest point 72 mm above the floor.
- **Trunk:** about 2–3° forward while it backs up, about 7–8° at rest. The v2 hood holds the adapter 2.3 mm lower,
  which keeps the magnetic tip on it in both.
- **Docking:** with the MicroDock v2 it docked 30/30 (see the dock's [README](../README.md)).

This is a simulation result: the butt walk has not run on a robot. Before trying it, check two things on a real unit:
whether the seat slides on your floor (friction 0.3–1.0 was trained), and the hand-over from the sit.

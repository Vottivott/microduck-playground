# Swing training and selection notes

## Reproducible baseline

The environment task is `Mjlab-SwingPump-MicroDuck`. It starts every simulated
robot at the bottom of the arc with zero root and joint velocity. Positive
reward emphasizes new bidirectional angle frontiers and late-horizon height;
costs enforce planarity, cord geometry, attachment alignment, action rate, and
joint limits. Exact mechanism state is privileged to the critic and rewards,
never to the deployment actor.

The PPO actor/critic use hidden dimensions 512/256/128 with ELU activation and
observation normalization. The actor keeps the standard MicroDuck 61D contract
and produces 14 joint-position actions at 50 Hz.

## Executed training stages

The released policy was reached by progressively making physical validity part
of the objective, rather than maximizing angle first and filtering afterward:

1. A conservative still-start pumping policy established repeatable amplitude.
2. Planarity, attachment alignment, and tension-only cord constraints were
   increased in stages through iterations 2,000, 2,500, and 3,000.
3. Exploration variance was reduced while lateral position and velocity,
   roll/yaw kinetics, and both cord envelopes were screened together.
4. Invalid-geometry termination was replaced by full-horizon violation debt so
   a brief excursion could not be hidden by a reset.
5. Bounded predictive-severity terms and planar hip correction improved the
   late-horizon envelope without giving the actor privileged mechanism state.
6. A cord-aware sagittal-only continuation and final-layer correction produced
   the two preserved selection endpoints.
7. The final actor-layer interpolation sweep below selected alpha 0.50.

The stage boundaries are encoded directly in
`make_microduck_swing_env_cfg()` and its curricula; the physical audit script
reports the exact hard gates used for promotion.

## Selected checkpoint construction

The published checkpoint is a measured interpolation frontier, not the final
checkpoint of one long run:

1. `frontier_source_3500.pt` was the prepared, desaturated 36-second source
   policy derived from the robust head-2% frontier.
2. A 36-second continuation targeted late-horizon planar/cord validity and was
   early-stopped at `planar_target_3600.pt`; later checkpoints regressed.
3. Only the final actor layer differed, so exact all-row alphas 0.25, 0.50, and
   0.75 were screened on deliberately difficult seeds.
4. Alpha 0.50 won the hard screen and then the complete 100-seed, 36-second
   audit with 71 strict passes.
5. Finer alphas 0.40/0.45/0.50/0.55/0.60 confirmed 0.50 was the local winner.

Recreate the interpolation:

```bash
uv run python scripts/interpolate_actor_checkpoints.py \
  experiments/swing/checkpoints/frontier_source_3500.pt \
  experiments/swing/checkpoints/planar_target_3600.pt \
  /tmp/alpha050.pt \
  --alpha 0.5 \
  --action-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13
```

The checkpoint serialization also embeds source path provenance, so a relocated
recreation can have a different file SHA-256. The script strictly verifies the
frozen actor tensors and unselected rows; compare the recreated
`actor_state_dict` tensor-by-tensor with `alpha050.pt` before evaluating.

## Continue from the released policy

The Hugging Face release includes `checkpoint.pt`, identical to the selected
`alpha050.pt` here. Its actor is the exact released actor. Its critic comes from
the alpha-0.50 source endpoint rather than being interpolated, and its optimizer
moments are intentionally empty at a conservative `1e-7` learning rate with
exploration standard deviation `0.02`.

```bash
mkdir -p logs/rsl_rl/microduck_swing/release-alpha050
cp checkpoint.pt logs/rsl_rl/microduck_swing/release-alpha050/model_3500.pt

uv run train Mjlab-SwingPump-MicroDuck \
  --agent.resume True \
  --agent.load-run release-alpha050 \
  --agent.load-checkpoint model_3500.pt \
  --agent.max-iterations 25
```

Begin with a short continuation and re-run the full physical audit. PyTorch
checkpoints use pickle internally; load them only from a repository and
revision you trust.

## Release boundary

This history stops at the alpha-0.50 policy used in the published 173.20°
video. Later exploratory work on directional/single-sided release, intentional
falls, crash rendering, shortened cords, and full loops is intentionally not
included in this repository or model release.

## Reproducibility boundary

The code, seed-specific evaluator, selected weights, endpoints, and exact
selection transformation are reproducible. Training a bit-identical PPO model
from scratch is not promised across CUDA, MuJoCo Warp, PyTorch, driver, and
parallel-reduction changes. Reproduction should therefore report both the
software commit and physical-gate metrics, not only total reward or angle.

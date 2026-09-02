# Microduck tall hybrid-stilt curriculum

The selected hardware-direction target is locomotion on the blend-0.50 support,
up to 25 cm below each original sole. A separate simulation-only extension
pushes the same shape toward 2.0 m to probe the learned controller's transfer
ceiling. Its rounded tip is 17 × 22 mm: visibly stilt-like and materially
narrower than the 22 × 32 mm bootstrap platform, while retaining the small-boot
silhouette chosen after comparing rollout videos. Shape and height are
deliberately trained as independent difficulty axes. A checkpoint never
encounters a simultaneous height jump and support-area reduction.

## Morphology representation

`Mjlab-Stilt-Flat-MicroDuck` compiles one morphology selected by:

```bash
MICRODUCK_STILT_HEIGHT_MM=20 MICRODUCK_STILT_BLEND=0 \
  uv run train Mjlab-Stilt-Flat-MicroDuck
```

`blend=0` is the 22 × 32 mm platform, the selected `blend=0.50` target has a
17 × 22 mm rounded tip, and `blend=1` is the 12 mm circular research peg. The
intermediate dimensions match the cartridge progression. The generated MuJoCo
loft is the contacting geometry; the old sole collision is disabled, the
terrain-height/contact sensors use sites at the actual stilt tips, and the reset
height includes the selected stilt length. Each stilt contributes an explicit
stage-dependent mass and inertia.

The actor observation remains the standard 61D hot-swap layout. Morphology is
not exposed as an extra observation because each deployment policy targets a
known attachment and changing the observation contract would prevent reuse of
the existing runtime.

## Training stages

1. **Bootstrap balance and gait:** 2 cm, blend 0. Start with 25% exact-zero
   commands and modest walking commands. No pushes; minimal action-rate cost.
2. **Narrow the support at fixed height:** 2 cm at blends 0.25, then 0.50.
   Resume each stage from the previous winning checkpoint. Blends 0.75 and 1.00
   were trained successfully as research comparisons, but are not the chosen
   hardware direction.
3. **Grow the selected hybrid without changing its tip:** blend 0.50 at 2, 3,
   4, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, then 25 cm. Insert a midpoint if a
   transfer stage loses the behavior immediately.
4. **Simulation-only height extension:** after 25 cm succeeds, continue at
   27.5, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90 cm, then 1.0, 1.1, 1.2,
   1.4, 1.6, 1.8, and 2.0 m. Insert smaller midpoints whenever a jump
   does not recover monotonically.
5. **Consolidate the final morphology:** after the chosen stopping height exists, widen the
   velocity/yaw commands, introduce small pushes, then randomize friction,
   stilt mass, mounting tilt, and height around the manufactured value.
6. **Sim2real rehearsal:** export with `scripts/export.py`, run the policy on an
   all-collision deployment model with the exact attachment mass and measured
   geometry, then begin tethered hardware tests at a much shorter height.

## Promotion gate

A stage advances only after a 1,024-environment, 10-second evaluation shows:

- at least 90% of episodes remain upright for the full horizon;
- the commanded forward/standing buckets both work (no perpetual stepping to
  survive zero command);
- ground contacts come from the stilt geoms, not the original sole or another
  body part;
- velocity tracking improves rather than total reward rising only through
  regularizers;
- every weighted penalty logged to W&B is non-positive; and
- a close rollout video looks like controlled stepping rather than hopping,
  scraping, or exploiting collision geometry.

Interesting successes and new failure modes should be rendered close to the
robot for visual inspection; routine checkpoints do not need a full render.

## Expected risk beyond 25 cm

The 25 cm geometry roughly doubles the robot's total standing height; the
50 cm simulation milestone roughly triples it, while the 1.0 and 2.0 m
experiments push the support tip to about four and eight times the robot's own
body height below its feet. Each small contact consequently has an extreme
moment arm. The extension is a controller research target, not a direct
instruction to print and mount the CAD part. Hardware requires an engineered
metal/composite structure, measured inertia, a stronger mounting interface,
current and fall limits, a soft exclusion zone, and an overhead tether.

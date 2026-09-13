# Pin-free friction-fit validation

Geometric checks only. No physical insertion-force, pull-out-force, wear or load testing has been performed.

- 98 STL exports reload finite, watertight, consistently wound and positive-volume. Manufacturing parts each contain one solid.
- Evaluated-geometry SHA-256: `8de61699397e327238ab115a6f2370943c672d67a8a0469c1c785f8a732274d0`.
- Threaded clamp parts match the shared current `clamps/` set. No securing pin parts or locking pockets are generated.

## friction-fit

- 23 manufacturing parts including fit coupons; maximum packed axis 230.438 mm.
- Socket/tread intersection: 0.000000 mm³. Tread-union difference: 0.005398 mm³.
- Maximum individual tread volume missing from its assigned whole module: 0.000912 mm³.
- Maximum sampled unintended module docking overlap: 0.000391 mm³; floor docking: 0.000008 mm³.
- Intentional rib interference per assembled joint/component: 0.754–4.526 mm³. This is confined to the rib envelopes, not unexplained body collision.
- Floor shoe/crossbar solids do not overlap.

## friction-fit-3step

- 27 manufacturing parts including fit coupons; maximum packed axis 228.191 mm.
- Socket/tread intersection: 0.000000 mm³. Tread-union difference: 0.005398 mm³.
- Maximum individual tread volume missing from its assigned whole module: 0.000912 mm³.
- Maximum sampled unintended module docking overlap: 0.000574 mm³; floor docking: 0.000008 mm³.
- Intentional rib interference per assembled joint/component: 0.754–4.526 mm³. This is confined to the rib envelopes, not unexplained body collision.
- Floor shoe/crossbar solids do not overlap.

## Interpretation

The solid ribs intentionally extend 0.05 mm per face into the nominal socket wall. Rigid boolean overlap measures geometry, not elastic deformation or friction. The default cannot be called a validated fit until printed. Coupons span 0, 0.025, 0.05 and 0.10 mm per face.

Docking paths were sampled at 0, 2, 5, 10, 15 and 20 mm along the rail direction. Unintended interference is measured after excluding the intentional rib volumes. These are sampled checks, not a continuous tolerance analysis.

Geometric packing excludes supports, brim, printer keep-out areas and mechanical orientation constraints. New connector hardware still needs robot-clearance checks; tread preservation alone does not establish sim-to-real transfer.

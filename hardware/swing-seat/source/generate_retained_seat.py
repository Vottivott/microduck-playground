#!/usr/bin/env python3
"""Generate seat with locator pads and slots plus removable retention hardware."""

from pathlib import Path

import trimesh

from generate_seat import PP, build
from retention_system import bumper_meshes, buckle_meshes, slot_cutters, strap_mesh


def export_m(mesh, path):
    out = mesh.copy()
    out.apply_scale(0.001)
    out.export(path)


def main():
    params = {**PP, "locator_pads": True}
    seat = build(params)
    seat = trimesh.boolean.difference([seat] + slot_cutters(), engine="manifold")
    trimesh.repair.fix_normals(seat)
    out = Path(__file__).resolve().parent / "output_retained"
    out.mkdir(parents=True, exist_ok=True)
    seat.export(out / "param_seat_retained_mm.stl")
    export_m(seat, out / "param_seat_retained.stl")
    export_m(strap_mesh(), out / "retention_strap.stl")
    for i, mesh in enumerate(bumper_meshes()):
        export_m(mesh, out / f"retention_bumper_{i}.stl")
    buckle = trimesh.boolean.union(buckle_meshes(), engine="manifold")
    export_m(buckle, out / "retention_buckle.stl")
    print(
        f"seat faces={len(seat.faces)} watertight={seat.is_watertight}; "
        f"strap watertight={strap_mesh().is_watertight}"
    )
    print("exported to", out)


if __name__ == "__main__":
    main()

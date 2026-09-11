"""Geometry for a removable, compliant Microduck battery retention system."""

from __future__ import annotations

import numpy as np
import trimesh


RP = dict(
    # Closed webbing loop in the trunk x-z plane, routed through seat slots.
    strap_x0=-50.0, strap_x1=-21.0,
    # The top run is intentionally snug to the 31.67 mm battery crown.  The
    # old 4 mm visual gap sat inside the useful backward head-pump envelope.
    strap_z0=-56.0, strap_z1=32.0,
    # 0.6 mm UHMWPE/polyester webbing fits the measured 0.89 mm gap between
    # the battery crown and the closest valid sagittal head sweep.
    strap_corner=5.0, strap_width=12.0, strap_thickness=0.6,
    # Slots are wider than the 12 mm webbing and cut only the retained variant.
    rear_slot_center_x=-50.0, front_slot_center_x=-21.0,
    slot_x=4.0, slot_y=15.0,
    # Compliant battery-facing inserts carried by the strap.
    rear_pad_x=(-49.0, -47.6),
    front_pad_x=(-24.3, -22.0),
    side_pad_y=6.0, pad_z=(-29.0, 11.0),
    top_pad_x=(-43.0, -29.0), top_pad_y=6.0,
    top_pad_z=(31.80, 32.15),
)


def _rounded_path_xz(P=RP, n=160):
    # Battery-hugging contour.  The rear shoulder stays below the backward
    # head arc, then rises along the battery chamfer before crossing its crown.
    # This is a centreline; the thin compliant webbing supplies the radius.
    pts = np.asarray([
        [P["strap_x0"], P["strap_z0"]],
        [P["strap_x1"], P["strap_z0"]],
        [P["strap_x1"], P["strap_z1"]],
        [-43.2, P["strap_z1"]],
        [-46.4, 27.7],
        [P["strap_x0"], 20.0],
        [P["strap_x0"], P["strap_z0"]],
    ])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0.0, d[-1], n, endpoint=False)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def strap_mesh(P=RP, margin=0.0):
    """Watertight flat webbing loop; width is along y, thickness radial in x-z."""
    path = _rounded_path_xz(P)
    x, z = path[:, 0], path[:, 1]
    dx = np.roll(x, -1) - np.roll(x, 1)
    dz = np.roll(z, -1) - np.roll(z, 1)
    length = np.hypot(dx, dz)
    nx, nz = -dz / length, dx / length
    half_w = P["strap_width"] / 2 + margin
    half_t = P["strap_thickness"] / 2 + margin
    rings = []
    for px, pz, pnx, pnz in zip(x, z, nx, nz):
        rings.append(np.array([
            [px - pnx * half_t, -half_w, pz - pnz * half_t],
            [px - pnx * half_t,  half_w, pz - pnz * half_t],
            [px + pnx * half_t,  half_w, pz + pnz * half_t],
            [px + pnx * half_t, -half_w, pz + pnz * half_t],
        ]))
    verts = np.vstack(rings)
    faces = []
    for i in range(len(rings)):
        j = (i + 1) % len(rings)
        for k in range(4):
            k2 = (k + 1) % 4
            faces += [[4 * i + k, 4 * j + k, 4 * j + k2],
                      [4 * i + k, 4 * j + k2, 4 * i + k2]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _tapered_block(x0, x1, y_half, z0, z1, margin=0.0):
    """Soft pad with chamfered/tapered ends, represented as a convex hull."""
    x0 -= margin; x1 += margin; y_half += margin; z0 -= margin; z1 += margin
    inset = min(1.2, (z1 - z0) * 0.12)
    points = []
    for z, xi in ((z0, inset), (z1, inset)):
        points.extend([
            [x0 + xi, -y_half, z], [x1 - xi, -y_half, z],
            [x0 + xi,  y_half, z], [x1 - xi,  y_half, z],
        ])
    # Mid-height points retain the full battery-facing area.
    zm = (z0 + z1) / 2
    points.extend([[x0, -y_half, zm], [x1, -y_half, zm],
                   [x0, y_half, zm], [x1, y_half, zm]])
    return trimesh.convex.convex_hull(np.asarray(points))


def bumper_meshes(P=RP, margin=0.0):
    rear = _tapered_block(*P["rear_pad_x"], P["side_pad_y"], *P["pad_z"], margin)
    front = _tapered_block(*P["front_pad_x"], P["side_pad_y"], *P["pad_z"], margin)
    top = _tapered_block(
        *P["top_pad_x"], P["top_pad_y"], *P["top_pad_z"], margin
    )
    return [rear, front, top]


def buckle_meshes(P=RP):
    """Compact ladder-lock buckle on the accessible underside webbing run."""
    cx = -35.5
    z = P["strap_z0"] - 1.5
    parts = []
    # Brushed-metal frame in the x-y plane, centered between the legs.
    for x, y, sx, sy in (
        (cx - 7.0, 0, 2.0, 16.0), (cx + 7.0, 0, 2.0, 16.0),
        (cx, -6.9, 12.0, 2.0), (cx, 6.9, 12.0, 2.0),
    ):
        b = trimesh.creation.box(extents=[sx, sy, 1.8])
        b.apply_translation([x, y, z])
        parts.append(b)
    tongue = trimesh.creation.box(extents=[11.5, 1.2, 1.2])
    tongue.apply_translation([cx, 0, z - 1.0])
    parts.append(tongue)
    return parts


def slot_cutters(P=RP):
    cutters = []
    for x in (P["rear_slot_center_x"], P["front_slot_center_x"]):
        c = trimesh.creation.box(extents=[P["slot_x"], P["slot_y"], 28.0])
        c.apply_translation([x, 0, -46.0])
        cutters.append(c)
    return cutters

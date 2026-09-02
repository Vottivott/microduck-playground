"""Parametric swing seat for MicroDuck (pollen-robotics).

A toddler-bucket-style swing seat, fully parametric, in the design language of
a molded rubber seat: rounded-rectangle ring with rolled rim tubes, a smooth
sling (floor pan + lofted front and back scoops), and V-straps per side
converging to eyelets that the two swing strings tie through.

Coordinate frame: the robot's `trunk_base` frame, in millimetres
(x forward, y left, z up). The exported meter-scale STL therefore drops into a
MuJoCo scene with ZERO offset relative to trunk_base (weld or contact).

Key validated properties (against the MicroDuck MJCF, self-collision-filtered):
  - full forward leg swing arc (hip -90..0 deg, all knee/ankle values) is free
  - ~5 mm fore-aft play: front scoop face ~5 mm ahead of the hip-motor swing
    arc, back scoop lower face ~5 mm behind the trunk shell
  - the seat blocks ~15% of feasible sagittal leg poses (fold-back-behind
    only) and ~6% of feasible head poses (extreme beak-dive only)
  - floor pan top sits 3 mm below the battery at rest
  - string eyelets at (x=0, y=+-88, z=85): above the combined COM (z~22) and
    above the legs' upward reach (z~58)

Run:  python generate_seat.py
Outputs: output/param_seat_mm.stl (millimetres, for printing/CAD)
         output/param_seat.stl    (metres, for MuJoCo)
"""
import numpy as np
import trimesh
from trimesh.creation import cylinder, icosphere, torus, extrude_polygon, sweep_polygon
from shapely.geometry import box as shp_box, Polygon

# All lengths in mm, trunk_base frame.
PP = dict(
    # ring (rounded rectangle): wall band with rolled rim tubes
    ring_half=80.0, ring_corner=28.0, wall_t=8.0,
    ring_z0=14.0, ring_z1=44.0, rim_r=5.0,
    front_open_hw=52.0,      # |y| half-width of the front opening (head passage)
    back_hoop_z1=24.0,       # low back hoop top (above it: notch for the neck)
    # sling floor pan
    floor_x0=-44.0, floor_x1=30.0, floor_hw=24.0,
    floor_z1=-47.0, floor_t=6.0,
    # front scoop: the pan curls up IN FRONT of the hip-motor block -- a lofted
    # wall, full pan width at the base, narrowing as it rises (the motors and
    # yaw-in leg sweeps pass beside/above the narrow upper part)
    scoop_path=((12.0, -50.0), (24.0, -50.0), (31.0, -48.5), (36.0, -44.5),
                (38.5, -38.0), (39.5, -30.0), (40.0, -22.5), (40.2, -21.0)),
    scoop_widths=(24.0, 24.0, 23.5, 22.0, 18.0, 13.5, 10.0, 6.5),
    scoop_th=5.5,
    # back scoop (pan -> back wall -> hoop), mirrors the front design language;
    # lower face tight behind the trunk, upper face recessed for the head arch,
    # width tapers to 14 where back-swinging feet pass
    back_path=((-28.0, -50.0), (-44.0, -50.0), (-52.0, -48.0), (-56.5, -43.0),
               (-58.0, -35.0), (-58.0, -15.0), (-60.0, -4.0), (-65.0, 6.0),
               (-69.0, 15.0), (-70.5, 22.0)),
    back_widths=(24.0, 24.0, 22.0, 17.0, 14.5, 14.0, 14.0, 14.0, 14.0, 14.0),
    # V-straps to string eyelets
    strap_r=4.0, strap_root_x=45.0, eyelet_y=88.0, eyelet_z=75.0,
    eyelet_R=8.0, eyelet_r=3.5,
    # Optional battery-corner locator pads. Disabled for the byte-identical
    # archive default; generate_padded_seat.py enables them. The pads live at
    # the rear/lower battery corners, behind the hip axes, and taper in x/y so
    # they do not create a hard ledge beside the moving-leg corridor.
    locator_pads=False,
    locator_x0=-44.0, locator_x1=-40.0,
    locator_top_x0=-43.5, locator_top_x1=-40.5,
    locator_inner_y=13.0, locator_top_inner_y=16.0,
    locator_outer_y=17.5, locator_top_outer_y=17.1,
    locator_z0=-47.0, locator_z1=-39.8,
)


def _circle(r, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.c_[r * np.cos(a), r * np.sin(a)])


def _ring_path(P, n=160):
    """uniformly resampled rounded-rectangle centerline of the ring"""
    half = P["ring_half"]; cr = P["ring_corner"]
    base = shp_box(-(half - cr), -(half - cr), half - cr, half - cr)
    cl = base.buffer(cr, quad_segs=16)
    pts = np.array(cl.exterior.coords)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def rounded_box(x0, x1, y_hw, z0, z1, r=3.0):
    """box with vertical rounded corners via shapely buffer"""
    poly = shp_box(x0 + r, -y_hw + r, x1 - r, y_hw - r).buffer(r, quad_segs=8)
    b = extrude_polygon(poly, z1 - z0)
    b.apply_translation([0, 0, z0])
    return b


def tube_along(path3d, r, closed=False):
    return sweep_polygon(_circle(r), path3d, connect=closed)


def locator_pad_meshes(P, margin=0.0):
    """Two tapered convex battery-corner pads, optionally conservatively grown.

    ``margin`` is used only by the clearance verifier. It expands each pad
    toward the robot and beyond its x/z ends; the printable build uses zero.
    """
    pads = []
    for sy in (-1, 1):
        x0 = P["locator_x0"] - margin
        x1 = P["locator_x1"] + margin
        tx0 = (P["locator_top_x0"] if "locator_top_x0" in P
               else P["locator_x0"] + P["locator_top_inset_x"]) - margin
        tx1 = (P["locator_top_x1"] if "locator_top_x1" in P
               else P["locator_x1"] - P["locator_top_inset_x"]) + margin
        z0 = P["locator_z0"] - margin
        z1 = P["locator_z1"] + margin
        inner = P["locator_inner_y"] - margin
        top_inner = P.get("locator_top_inner_y", P["locator_inner_y"]) - margin
        outer = P["locator_outer_y"] + margin
        top_outer = P["locator_top_outer_y"] + margin
        # Work in positive y, then mirror. Convex-hulling the two rectangular
        # sections gives tapered x ends and a tapered outer face.
        points = np.array([
            [x0, inner, z0], [x1, inner, z0],
            [x0, outer, z0], [x1, outer, z0],
            [tx0, top_inner, z1], [tx1, top_inner, z1],
            [tx0, top_outer, z1], [tx1, top_outer, z1],
        ])
        points[:, 1] *= sy
        pad = trimesh.convex.convex_hull(points)
        trimesh.repair.fix_normals(pad)
        pads.append(pad)
    return pads


def loft_scoop(path_xz, widths, th, n_sub=6):
    """Loft a variable-width band along an x-z path (widths along y).
    Cross-sections are stadium-shaped (flat faces joined by constant-radius
    rounded sides of radius `th`); the path and widths are smoothed with
    monotone-cubic (PCHIP) interpolation. Returns a watertight mesh."""
    from scipy.interpolate import PchipInterpolator
    t = np.linspace(0, 1, len(path_xz))
    tt = np.linspace(0, 1, len(path_xz) * n_sub)
    px = PchipInterpolator(t, path_xz[:, 0])(tt)
    pz = PchipInterpolator(t, path_xz[:, 1])(tt)
    ws = PchipInterpolator(t, widths)(tt)
    dx = np.gradient(px); dz = np.gradient(pz)
    L = np.hypot(dx, dz); dx /= L; dz /= L
    nx, nz = -dz, dx
    n_arc = 8
    a1 = np.linspace(-np.pi / 2, np.pi / 2, n_arc)       # +y end cap
    a2 = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)    # -y end cap
    rings = []
    for i in range(len(tt)):
        w = max(ws[i] - th, 0.5)
        sec = np.vstack([
            np.c_[th * np.cos(a1), w + th * np.sin(a1)],
            np.c_[-th * np.ones(3), np.linspace(w, -w, 5)[1:-1]],
            np.c_[th * np.cos(a2), -w + th * np.sin(a2)],
            np.c_[th * np.ones(3), np.linspace(-w, w, 5)[1:-1]],
        ])
        cn, cy = sec[:, 0], sec[:, 1]
        rings.append(np.c_[px[i] + nx[i] * cn, cy, pz[i] + nz[i] * cn])
    n_ring = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    for i in range(len(rings) - 1):
        a0 = i * n_ring; b0 = (i + 1) * n_ring
        for j in range(n_ring):
            j2 = (j + 1) % n_ring
            faces += [[a0 + j, b0 + j, b0 + j2], [a0 + j, b0 + j2, a0 + j2]]
    c0 = len(verts); verts = np.vstack([verts, [px[0], 0, pz[0]]])
    c1 = len(verts); verts = np.vstack([verts, [px[-1], 0, pz[-1]]])
    last = (len(rings) - 1) * n_ring
    for j in range(n_ring):
        j2 = (j + 1) % n_ring
        faces += [[c0, j2, j], [c1, last + j, last + j2]]
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m


def build(P=PP):
    parts, cutters = [], []
    half = P["ring_half"]; t = P["wall_t"] / 2
    cl = _ring_path(P)
    cl_poly = Polygon(cl)
    wall = cl_poly.buffer(t, quad_segs=8).difference(cl_poly.buffer(-t, quad_segs=8))
    w = extrude_polygon(wall, P["ring_z1"] - P["ring_z0"])
    w.apply_translation([0, 0, P["ring_z0"]])
    parts.append(w)
    # rolled rims (top + bottom)
    for z in (P["ring_z0"], P["ring_z1"]):
        parts.append(tube_along(np.c_[cl, np.full(len(cl), z)], P["rim_r"], closed=True))
    # back hoop top rounding tube (straight segment across the back) + end spheres
    parts.append(cylinder(radius=P["rim_r"] * 0.9,
                          segment=[[-half, -P["front_open_hw"], P["back_hoop_z1"]],
                                   [-half, P["front_open_hw"], P["back_hoop_z1"]]]))
    for sy in (-1, 1):
        s = icosphere(radius=P["rim_r"] * 0.9, subdivisions=2)
        s.apply_translation([-half, sy * P["front_open_hw"], P["back_hoop_z1"]])
        parts.append(s)
    # sector cutters: front full opening, back neck notch
    from trimesh.creation import box as tbox
    front_cut = tbox(extents=[100, 2 * (P["front_open_hw"] + 4), 80])
    front_cut.apply_translation([half + 10, 0, P["ring_z0"] + 40 - 14])
    cutters.append(front_cut)
    back_cut = tbox(extents=[100, 2 * P["front_open_hw"], 60])
    back_cut.apply_translation([-half - 10, 0, P["back_hoop_z1"] + 30])
    cutters.append(back_cut)

    def path_point(x_sign, ty):
        """closest ring-centerline point to y=ty on the x_sign side"""
        sel = cl[np.sign(cl[:, 0]) == x_sign] if x_sign else cl
        return sel[np.argmin(np.abs(sel[:, 1] - ty))]
    # round the wall END FACES themselves: a flush half-round edge cylinder on
    # the wall centerline at each cut, plus spheres capping the rim-tube ends
    for sy in (-1, 1):
        pf = path_point(1, sy * (P["front_open_hw"] + 4))
        parts.append(cylinder(radius=t, segment=[[pf[0], pf[1], P["ring_z0"]],
                                                 [pf[0], pf[1], P["ring_z1"]]]))
        pb = path_point(-1, sy * P["front_open_hw"])
        parts.append(cylinder(radius=t, segment=[[pb[0], pb[1], P["back_hoop_z1"] - 2],
                                                 [pb[0], pb[1], P["ring_z1"]]]))
        for pp, zs in ((pf, (P["ring_z0"], P["ring_z1"])), (pb, (P["ring_z1"],))):
            for zz in zs:
                s = icosphere(radius=P["rim_r"], subdivisions=3)
                s.apply_translation([pp[0], pp[1], zz])
                parts.append(s)

    # ---- sling ----
    floor = rounded_box(P["floor_x0"], P["floor_x1"], P["floor_hw"],
                        P["floor_z1"] - P["floor_t"], P["floor_z1"], r=8)
    parts.append(floor)
    # rolled bottom edge on the pan (inset so it stays flush with the walls)
    fp = shp_box(P["floor_x0"] + 8, -P["floor_hw"] + 8,
                 P["floor_x1"] - 8, P["floor_hw"] - 8).buffer(8, quad_segs=8)
    rim_path = np.array(fp.buffer(-3.0, quad_segs=8).exterior.coords)
    parts.append(tube_along(
        np.c_[rim_path[:-1, 0], rim_path[:-1, 1],
              np.full(len(rim_path) - 1, P["floor_z1"] - P["floor_t"] + 2.5)],
        3.0, closed=True))
    # back scoop: one smooth lofted band from the pan floor up to the hoop
    parts.append(loft_scoop(np.array(P["back_path"]),
                            np.array(P["back_widths"]), P["scoop_th"]))
    # front scoop: lofted curling wall in front of the hip motors
    parts.append(loft_scoop(np.array(P["scoop_path"]),
                            np.array(P["scoop_widths"]), P["scoop_th"]))

    # Optional compliant locators against the fixed battery's rear/lower
    # corners. They are intentionally separate from the hip/thigh openings.
    if P.get("locator_pads", False):
        parts.extend(locator_pad_meshes(P))

    # ---- straps: V per side, converging to an eyelet ----
    for sy in (-1, 1):
        top = np.array([0, sy * P["eyelet_y"], P["eyelet_z"]])
        for rx in (P["strap_root_x"], -P["strap_root_x"]):
            root = np.array([rx, sy * (half - 2), P["ring_z1"] - 2])
            mid = (root + top) / 2 + np.array([0, sy * 4, 0])
            path = np.array([root, mid, top])
            parts.append(tube_along(path, P["strap_r"]))
            bump = icosphere(radius=P["strap_r"] + 2.5, subdivisions=2)
            bump.apply_translation(root)
            parts.append(bump)
        j = icosphere(radius=P["strap_r"] + 2, subdivisions=2)
        j.apply_translation(top)
        parts.append(j)
        ey = torus(major_radius=P["eyelet_R"], minor_radius=P["eyelet_r"])
        ey.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        ey.apply_translation([0, sy * P["eyelet_y"], P["eyelet_z"] + P["eyelet_R"] + 2])
        parts.append(ey)

    seat = trimesh.boolean.union(parts, engine="manifold")
    seat = trimesh.boolean.difference([seat] + cutters, engine="manifold")
    trimesh.repair.fix_normals(seat)
    return seat


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)
    seat = build()
    print(f"faces={len(seat.faces)} watertight={seat.is_watertight} "
          f"volume={seat.volume/1000:.0f} cm^3")
    print("bounds (mm):\n", np.round(seat.bounds, 1))
    seat.export(os.path.join(out, "param_seat_mm.stl"))
    sm = seat.copy(); sm.apply_scale(0.001)
    sm.export(os.path.join(out, "param_seat.stl"))
    print(f"exported to {out}")

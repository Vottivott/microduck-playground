# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "manifold3d>=3.2",
#   "numpy>=2.0",
#   "scipy>=1.10",  # trimesh's graph engine: counts the bodies for --check
#   "trimesh>=4.8",
# ]
# ///
"""Printable MicroDock parts (manifold3d). No screws, inserts, feet or springs:
  base   : 3 mm plate with a front ramp and a socket the board drops into (plus two gussets)
  board  : printed flat; tag pocket, window with V-seats for the hood's pegs, cable notch in the left edge
  hood   : printed upside down; top plate + lip, side walls + flares, adapter pocket, battery stop, two D-pegs
  keeper : snaps onto the pegs behind the board (stops the hood from being pulled off)
  tag    : 60 mm insert, AprilTag tag36h11 #0: black base with the white cells raised 0.6 mm (one filament change)
Meshes are in assembly frames (mm): world for base and board (board front face at x = 0), the hood frame for the hood
and keeper (= the duck's trunk frame when docked), the tag's own frame (face toward +x). Orient them in the slicer as
the README describes.

uv run hardware/microdock/source/cad.py [--out DIR] [--check]
"""
import argparse, json, os, sys
import numpy as np
import manifold3d as m3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dock import Params, hood_pieces

HERE = os.path.dirname(os.path.abspath(__file__))
BED_MM = 256.0                              # Bambu Lab P1S build volume (256 x 256 x 256 mm)


def hull(pts): return m3.Manifold.hull_points([tuple(map(float, q)) for q in np.asarray(pts)])
def cube(x0, x1, y0, y1, z0, z1): return m3.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))
def cyl_x(r, x0, x1, y, z, n=40): return m3.Manifold.cylinder(x1 - x0, r, r, n).rotate((0, 90, 0)).translate((x0, y, z))
def union(parts):
    out = parts[0]
    for q in parts[1:]: out = out + q
    return out


PEG_Y, PEG_Z, PEG_R = 14.0, 30.0, 3.0      # hood frame: two pegs through the board
KEEPER_T = 4.0
CABLE_SLOT = 4.8                          # cable exit slot in the hood's left side wall (cable ~4 mm)
TAG36H11_ID0 = 0xd5d628584                # tag36h11 code word for id 0 (AprilTag library)


def base(p: Params):
    x0, x1 = p.plate_x; w = p.plate_half_w; T = p.T; L = p.ramp_len
    s = cube(x0, x1 - L, -w, w, 0, T)
    s = s + hull(np.array([[x1 - L - 0.6, y, z] for y in (-w, w) for z in (0, T)] + [[x1, y, z] for y in (-w, w) for z in (0, p.ramp_edge)]))
    # board socket: two walls 16 mm tall, slot 0.3 mm wider than the board
    bt = p.board_t + 0.3; wb = p.board_half_w + 0.3
    s = s + cube(0.15, 4.15, -wb - 4, wb + 4, T - 0.4, T + p.socket_h)
    s = s + cube(-bt + 0.15 - 4, -bt + 0.15, -wb - 4, wb + 4, T - 0.4, T + p.socket_h)
    s = s + cube(-bt + 0.15 - 4, 4.15, wb, wb + 4, T - 0.4, T + p.socket_h) + cube(-bt + 0.15 - 4, 4.15, -wb - 4, -wb, T - 0.4, T + p.socket_h)
    for yy in (-40.0, 40.0):           # gussets behind the socket
        s = s + hull(np.array([[-bt + 0.15 - 4 + 0.5, yy + d, T + p.socket_h - 0.5] for d in (-1.5, 1.5)] +
                              [[x0 + 4, yy + d, T - 0.4] for d in (-1.5, 1.5)] + [[-bt + 0.15 - 4 + 0.5, yy + d, T - 0.4] for d in (-1.5, 1.5)]))
    return s


def board(p: Params):
    """In world frame, standing in the socket (printed lying flat on its back face)."""
    T = p.T; w = p.board_half_w
    s = cube(-p.board_t, 0.0, -w, w, T, p.board_top)
    # window for the pegs, with a V seat under each peg (centres the hood when nothing lifts it)
    zr = p.z0 - p.hood_drop + PEG_Z        # peg centre at rest
    win = cube(-p.board_t - 1, 1, -(PEG_Y + PEG_R + p.hood_float_y + 1), PEG_Y + PEG_R + p.hood_float_y + 1,
               zr - PEG_R + 4.0, p.z0 + p.hood_lift + PEG_Z + PEG_R + 1.5)
    s = s - win
    for sy in (-1, 1):                      # V seats: below the window, a shallow V opening up to the window floor
        yc = sy*PEG_Y
        v = hull(np.array([[xx, yc + dy, zr - PEG_R + dz] for xx in (-p.board_t - 1, 1) for dy, dz in
                           ((0, -0.2), (-(PEG_R + p.hood_float_y + 1), 4.2), (PEG_R + p.hood_float_y + 1, 4.2))]))
        s = s - v
    # cable notch in the left edge, level with the hood's side-wall cable exit (room for the hood to float)
    zc = p.z0 + p.port_z
    s = s - cube(-p.board_t - 1, 1, PEG_Y + PEG_R + p.hood_float_y + 4.0, w + 1, zc - p.hood_drop - 6.0, zc + p.hood_lift + 6.0)
    # tag pocket
    tw = p.tag_size/2 + 2.0 + 0.15
    s = s - cube(-1.6, 1, -tw, tw, p.tag_cz - tw, p.tag_cz + tw)
    # rounded top corners (the board prints flat on its back; the cable just hangs behind it)
    r = 8.0
    for sy in (-1, 1):
        y0, y1 = (w - r, w + 1) if sy > 0 else (-w - 1, -w + r)
        s = s - (cube(-p.board_t - 1, 1, y0, y1, p.board_top - r, p.board_top + 1) - cyl_x(r, -p.board_t - 2, 2, sy*(w - r), p.board_top - r))
    return s


def hood(p: Params):
    """Hood frame. Union of the simulator's convex pieces + pegs; adapter pocket and cable path cut out."""
    pcs = hood_pieces(p)
    s = union([hull(P) for P in pcs.values()])
    pv, ph, pl = p.plug
    ax0 = p.adapter_face_x
    # adapter pocket (0.15 mm clearance), open at the bottom: the adapter + cable go in from below as one piece
    s = s - cube(ax0 - pl - 0.2, ax0 + 1, -ph/2 - 0.15, ph/2 + 0.15, -1.0, p.port_z + pv/2 + 0.15)
    al, aw, at = p.angle_plug
    ed, el, xe = p.angle_elbow
    # Right-angle head behind the adapter: its body runs along the plug's wide side, so with the port horizontal it lies
    # flat (body diameter `at` vertical, `aw` deep behind the adapter) and points to the duck's left (+y); its cable
    # leaves through the left side wall. The channel fits it with 0.6 mm to spare behind.
    s = s - cube(ax0 - pl - aw - 0.6, ax0 - pl + 0.2, -xe - 0.65, p.side_in + 0.01, -1.0, p.port_z + at/2 + 0.3)   # open below, like the pocket
    # Head stop: a rib behind the sleeve's left wall catches the front of the head's body, so the magnet can't pull the
    # adapter forward out of place (the elbow alone would fit into the pocket's back). It clears the elbow's flared side
    # and the body's path when the pair goes in from below.
    s = s + cube(ax0 - pl - ed + 0.15, ax0 - pl + 0.4, (el - xe) + 2.1, ph/2 + 2.2, p.port_z - at/2 - 0.5, p.port_z + pv/2 + 2.0)
    # cable exit: a slot open at the wall's bottom edge, so the cable slips in from below (no need to unplug it)
    xc = ax0 - pl - (aw - at/2)                    # the cable leaves the head on its body's axis
    s = s - m3.Manifold.cylinder(p.wall_t + 2.0, CABLE_SLOT/2, CABLE_SLOT/2, 40).rotate((-90, 0, 0)).translate((xc, p.side_in - 1.0, p.port_z))
    s = s - cube(xc - CABLE_SLOT/2, xc + CABLE_SLOT/2, p.side_in - 1.0, p.side_in + p.wall_t + 1.0, p.side_bot - 2.0, p.port_z)
    for sx in (-1, 1):                     # small lead-in chamfers at the slot's mouth
        s = s - hull(np.array([[xc + sx*(CABLE_SLOT/2 - 0.01), y, z] for y in (p.side_in - 1.0, p.side_in + p.wall_t + 1.0) for z in (p.side_bot - 2.0, p.side_bot + 1.2)] +
                              [[xc + sx*(CABLE_SLOT/2 + 1.2), y, p.side_bot - 2.0] for y in (p.side_in - 1.0, p.side_in + p.wall_t + 1.0)]))
    # D-pegs (flat on top so they print without supports when the hood prints upside down)
    for sy in (-1, 1):
        peg = cyl_x(PEG_R, p.hood_rear - p.board_t - KEEPER_T - 1.2, p.hood_rear + 0.5, sy*PEG_Y, PEG_Z)
        peg = peg - cube(p.hood_rear - 20, p.hood_rear + 1, sy*PEG_Y - 4, sy*PEG_Y + 4, PEG_Z + PEG_R - 0.6, PEG_Z + 5)
        s = s + peg
    return s


def keeper(p: Params):
    """Bar behind the board with two snap holes for the pegs (hood frame, behind the board)."""
    x1 = p.hood_rear - p.board_t - 0.4; x0 = x1 - KEEPER_T
    k = cube(x0, x1, -PEG_Y - 7, PEG_Y + 7, PEG_Z - 6, PEG_Z + 6)
    for sy in (-1, 1):
        k = k - cyl_x(PEG_R + 0.12, x0 - 1, x1 + 1, sy*PEG_Y, PEG_Z)
    k = k - cube(x0 - 1, x1 + 1, -3.5, 3.5, PEG_Z - 7, PEG_Z + 2.5)    # cable notch from below, between the pegs
    return k


def tag_cells(code=TAG36H11_ID0):
    """10 x 10 cells of the tag, True = white: white quiet zone, black border ring, 6 x 6 data bits."""
    g = np.ones((10, 10), bool)
    g[1:9, 1:9] = False
    for i in range(36):
        r, c = divmod(i, 6); g[2 + r, 2 + c] = bool((code >> (35 - i)) & 1)
    return g


def tag_insert(p: Params):
    """The tag insert in its own frame (face toward +x, centred at 0): black base, white cells raised 0.6 mm."""
    im = tag_cells(); n = 10
    c = p.tag_size/n; base_w = p.tag_size/2 + 2.0
    s = cube(-1.6, 0.0, -base_w, base_w, -base_w, base_w)
    cells = []
    for i in range(n):
        for j in range(n):
            if im[i, j]:
                e = 0.02                                                   # tiny inset: cells never share an edge
                cells.append(cube(-0.01, 0.6, -p.tag_size/2 + j*c + e, -p.tag_size/2 + (j + 1)*c - e, p.tag_size/2 - (i + 1)*c + e, p.tag_size/2 - i*c - e))
    # the white border outside the black ring is the base plate's own colour change: raise it too
    border = cube(-0.01, 0.6, -base_w, base_w, -base_w, base_w) - cube(-1, 1, -p.tag_size/2 - 0.02, p.tag_size/2 + 0.02, -p.tag_size/2 - 0.02, p.tag_size/2 + 0.02)
    white = union(cells + [border])
    return s, white


def tag(p: Params):
    blk, white = tag_insert(p)
    return blk + white


# name -> builder, in assembly frames (see the module docstring). Each part is built and meshed in this order, one at a
# time: manifold evaluates booleans lazily and numbers meshes as it goes, so this keeps the triangulation (the STL bytes)
# stable from run to run.
PARTS = (("microdock_base", base), ("microdock_board", board), ("microdock_hood", hood),
         ("microdock_keeper", keeper), ("microdock_tag", tag))


def head_fit(p: Params, hood_part=None):
    """The right-angle cable head (elbow, round body, strain relief, cable to the slot) and the adapter against the hood:
    overlap (mm3) at the nominal position, the worst overlap while the joined pair rises in from below, and how far the
    head can be pulled forward (toward the duck) before the hood stops it."""
    from dock import adapter_box
    hood_part = hood_part or hood(p)
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug; ed, el, xe = p.angle_elbow
    x0 = p.adapter_face_x - pl; r = at/2; xa = x0 - (aw - r)          # adapter back face; body axis (hood x)
    def cyl_y(r0, r1, y0, y1, x, z): return m3.Manifold.cylinder(y1 - y0, r0, r1, 48).rotate((-90, 0, 0)).translate((x, y0, z))
    def pair(dx=0.0, dz=0.0, cable=True):
        z = p.port_z + dz
        h = hull(np.array([[x0 + dx + xx, y, z + zz] for xx, y in ((0, -xe), (0, el - xe), (-(ed + 2.1), el - xe + 1.8), (-(ed + 2.1), -xe)) for zz in (-r, r)]))
        h = h + cyl_y(r, r, -xe, al - 8.8 - xe, xa + dx, z) + cyl_y(r, 2.9, al - 8.8 - xe, al - xe, xa + dx, z)
        if cable: h = h + cyl_y(2.0, 2.0, al - xe, p.side_in + p.wall_t + 3.0, xa + dx, z)
        return h + hull(adapter_box(p) + np.array([dx, 0, dz]))
    nominal = (pair() ^ hood_part).volume()
    path = max((pair(dz=dz) ^ hood_part).volume() for dz in np.arange(-45.0, 0.01, 1.0))
    travel = next((dx for dx in np.arange(0.05, 5.0, 0.05) if (pair(dx=dx, cable=False) ^ hood_part).volume() > 0.01), None)
    return dict(nominal_mm3=round(nominal, 4), insertion_mm3=round(path, 4), forward_travel_mm=None if travel is None else round(float(travel), 2))


def export_all(p: Params, outdir):
    import trimesh
    os.makedirs(outdir, exist_ok=True); info = {}
    for name, build in PARTS:
        m = build(p).to_mesh(); tm = trimesh.Trimesh(m.vert_properties[:, :3], m.tri_verts)
        tm.export(os.path.join(outdir, name + ".stl"))
        e = tm.bounds[1] - tm.bounds[0]
        info[name + ".stl"] = dict(extents_mm=[round(float(v), 1) for v in e], volume_cm3=round(float(tm.volume)/1000, 1),
                                   watertight=bool(tm.is_watertight), bodies=len(tm.split(only_watertight=False)))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "stl"), help="output directory (default: ../stl)")
    ap.add_argument("--check", action="store_true", help="fail unless every part is one watertight body that fits a 256 mm bed and the cable head fits the hood")
    a = ap.parse_args()
    info = export_all(Params(), a.out)
    print(json.dumps(info, indent=1))
    if a.check:
        bad = [k for k, v in info.items() if not (v["watertight"] and v["bodies"] == 1 and max(v["extents_mm"]) <= BED_MM)]
        fit = head_fit(Params()); print("cable head fit:", json.dumps(fit))
        if fit["nominal_mm3"] > 0.01 or fit["insertion_mm3"] > 0.01 or fit["forward_travel_mm"] is None or fit["forward_travel_mm"] > 0.5:
            bad.append("cable head fit")
        if bad:
            sys.exit(f"check failed: {bad}")
        print(f"check passed: {len(info)} parts, each one watertight body within a {BED_MM:.0f} mm bed; the cable head fits, "
              f"goes in from below and stops within {fit['forward_travel_mm']} mm")

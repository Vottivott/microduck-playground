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
  board  : printed flat; tag pocket, window with V-seats for the hood's pegs, cable clip in the left edge
  hood   : printed upside down; top plate + lip, side walls + flares, adapter pocket with a snap catch, battery stop, two D-pegs
  keeper : a plate on the pegs behind the board, locked on by a piece of filament through each peg (holds the hood on)
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
# The keeper is what holds the hood on the board: the hood's weight sits in front of the board, so it always leans
# forward on its pegs, and undocking pulls it forward through the magnet. So the keeper must bear on the board's back at
# every height and side position the hood can take (the window lets the pegs travel drop + lift up and +-float
# sideways), and it must not slide off the pegs: it reaches KEEPER_UP above the pegs' bar, and a piece of 1.75 mm
# filament through a hole in each peg, just behind it, locks it on (keeper_fit checks both).
KEEPER_UP = 20.0                          # how far the keeper reaches above the 12 mm bar around the pegs (32 mm tall)
PIN_D, PIN_HOLE, PIN_GAP = 1.75, 1.85, 0.15   # filament pin; its hole (prints a snug fit); its gap to the keeper's back
CABLE_SLOT = 4.8                          # cable exit slot in the hood's left side wall (cable ~4 mm)
CABLE_D = 4.0                             # the cable's diameter (the board's cable clip is sized for it)
ADAPTER_R = 3.8                           # the magnetic adapter's section is nearly a stadium: its corner radius (product photos)
# Adapter catch: a finger hangs in each side wall of the adapter's pocket (thickness, width, free length above its tip),
# with a hook at its end that reaches CATCH_HOOK into the pocket, 0.55 mm past the adapter's side, under its rounded
# corner; CATCH_GAP: the slots around the finger.
CATCH_T, CATCH_W, CATCH_L, CATCH_HOOK, CATCH_GAP = 1.6, 5.0, 15.0, 0.75, 0.5
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


def cable_clip(p: Params):
    """Where the board's cable clip holds the cable (world y, z in mm): 4.5 mm in from the left edge, 10 mm below the
    hood's lowest reach (its rear wall slides on the board's face and would pinch a cable higher up)."""
    return p.board_half_w - 4.5, p.z0 - p.hood_drop - 10.0


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
    # cable clip: a keyhole in the left edge that the cable presses into and runs through to the back, below the hood's
    # reach (cable_clip). Holding the cable here keeps its weight, and any tug on it, off the floating hood.
    (yk, zk), mouth = cable_clip(p), CABLE_D - 0.5
    s = s - cyl_x(CABLE_D/2 + 0.05, -p.board_t - 1, 1, yk, zk)
    s = s - cube(-p.board_t - 1, 1, yk, w + 1, zk - mouth/2, zk + mouth/2)
    s = s - hull(np.array([[xx, w - 1.0, zk + dz*mouth/2] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)] +
                          [[xx, w + 0.5, zk + dz*(mouth/2 + 1.5)] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)]))   # lead-in
    # tag pocket
    tw = p.tag_size/2 + 2.0 + 0.15
    s = s - cube(-1.6, 1, -tw, tw, p.tag_cz - tw, p.tag_cz + tw)
    # rounded top corners (the board prints flat on its back; the cable just hangs behind it)
    r = 8.0
    for sy in (-1, 1):
        y0, y1 = (w - r, w + 1) if sy > 0 else (-w - 1, -w + r)
        s = s - (cube(-p.board_t - 1, 1, y0, y1, p.board_top - r, p.board_top + 1) - cyl_x(r, -p.board_t - 2, 2, sy*(w - r), p.board_top - r))
    return s


def catch_geometry(p: Params):
    """Where the catch's hooks sit (hood frame, mm). The adapter's section is nearly a stadium, so a hook can't reach
    under a flat bottom edge: its tip sits under the rounded corner, 0.1 mm below it, and the adapter rests on the tips.
    p.catch_lift (if set) raises the hooks: they then press the adapter up against the pocket's roof."""
    pv, ph, pl = p.plug; r = ADAPTER_R
    yw = ph/2 + 0.2                                  # the side walls' inner face
    yt = yw - CATCH_HOOK                             # a hook's tip
    zt = p.adapter_z - (pv/2 - r) - np.sqrt(r**2 - (yt - (ph/2 - r))**2) - 0.1 + getattr(p, "catch_lift", 0.0)
    zc = zt + CATCH_HOOK                             # the hook's 45-degree top meets the wall here
    zb = zt - 0.3 - CATCH_HOOK*np.sqrt(3)            # 0.3 mm flat, then the 30-degree lead-in down to the wall
    root = min(zt + CATCH_L, p.trunk_top - 1.0)      # the slots end here (at least 1 mm under the top plate)
    return dict(yw=yw, yt=yt, zt=zt, zc=zc, zb=zb, root=root, free_length=root - zt)


def adapter_solid(p: Params):
    """The magnetic adapter as it is: a 14 mm prism with a near-stadium section (corner radius ADAPTER_R)."""
    pv, ph, pl = p.plug; r = ADAPTER_R
    cs = m3.CrossSection.square((ph - 2*r, pv - 2*r), True).offset(r, m3.JoinType.Round, circular_segments=64)
    # extruded along z, then (section width, section height, length) -> (y, z, x)
    return m3.Manifold.extrude(cs, pl).transform(np.array([[0, 0, 1, p.adapter_face_x - pl], [1, 0, 0, 0], [0, 1, 0, p.adapter_z]], float))


def catch(p: Params, flex=0.0):
    """The adapter's catch: a finger hangs in each side wall of the pocket, with a hook at its end under the adapter's
    rounded corner. On its way up the adapter pushes the hooks aside (30-degree lead-ins) and they snap in under it; a
    firm pull down cams them out again. The fingers hang from the side walls, which rise to the top plate, so they grow
    straight up when the hood prints upside down. Returns the raised walls, the windows the fingers hang in, and the
    fingers, bent outward by `flex` mm (head_fit checks the insertion with them bent out of the way)."""
    pv, ph, pl = p.plug; g = catch_geometry(p)
    yw, yt, zt, zc, zb = g["yw"], g["yt"], g["zt"], g["zc"], g["zb"]
    xm = p.adapter_face_x - pl/2; x0, x1 = xm - CATCH_W/2, xm + CATCH_W/2
    wall = cube(p.adapter_face_x - pl + 0.4, p.adapter_face_x - 0.6, yw, ph/2 + 2.2, p.adapter_z, p.trunk_top + 0.5)
    window = cube(x0 - CATCH_GAP, x1 + CATCH_GAP, yw - 1.0, ph/2 + 2.21, -1.0, g["root"])
    finger = cube(x0, x1, yw, yw + CATCH_T, zb, g["root"] + 0.5)
    finger = finger + hull(np.array([[xx, y, z] for xx in (x0, x1) for y, z in
                                     ((yw + 0.01, zc), (yt, zt), (yt, zt - 0.3), (yw + 0.01, zb))]))
    both = lambda m: m + m.mirror((0, 1, 0))
    return both(wall), both(window), both(finger.translate((0, flex, 0)))


def hood(p: Params, flex=0.0, head=True):
    """Hood frame. Union of the simulator's convex pieces + pegs; adapter pocket and cable path cut out, adapter catch
    added (inside the hood, behind the battery stop, where the duck never reaches: the simulator leaves it out).
    head=False leaves out the right-angle cable head's channel, stop rib and cable slot (the caller fits its own)."""
    pcs = hood_pieces(p)
    s = union([hull(P) for P in pcs.values()])
    pv, ph, pl = p.plug
    ax0 = p.adapter_face_x
    # adapter pocket (0.15 mm clearance; its roof p.pocket_top, if set, over the adapter), open at the bottom: the
    # adapter + cable go in from below as one piece
    s = s - cube(ax0 - pl - 0.2, ax0 + 1, -ph/2 - 0.15, ph/2 + 0.15, -1.0, p.adapter_z + pv/2 + getattr(p, "pocket_top", 0.15))
    walls, windows, fingers = catch(p, flex)
    s = s + walls - windows + fingers
    if head:
        al, aw, at = p.angle_plug
        ed, el, xe = p.angle_elbow
        # Right-angle head behind the adapter: its body runs along the plug's wide side, so with the port horizontal it lies
        # flat (body diameter `at` vertical, `aw` deep behind the adapter) and points to the duck's left (+y); its cable
        # leaves through the left side wall. The channel fits it with 0.6 mm to spare behind.
        s = s - cube(ax0 - pl - aw - 0.6, ax0 - pl + 0.2, -xe - 0.65, p.side_in + 0.01, -1.0, p.adapter_z + at/2 + 0.3)   # open below, like the pocket
        # Head stop: a rib behind the sleeve's left wall catches the front of the head's body, so the magnet can't pull the
        # adapter forward out of place (the elbow alone would fit into the pocket's back). It clears the elbow's flared side
        # and the body's path when the pair goes in from below.
        s = s + cube(ax0 - pl - ed + 0.15, ax0 - pl + 0.4, (el - xe) + 2.1, ph/2 + 2.2, p.adapter_z - at/2 - 0.5, p.adapter_z + pv/2 + 2.0)
        # cable exit: a slot open at the wall's bottom edge, so the cable slips in from below (no need to unplug it)
        xc = ax0 - pl - (aw - at/2)                    # the cable leaves the head on its body's axis
        s = s - m3.Manifold.cylinder(p.wall_t + 2.0, CABLE_SLOT/2, CABLE_SLOT/2, 40).rotate((-90, 0, 0)).translate((xc, p.side_in - 1.0, p.adapter_z))
        s = s - cube(xc - CABLE_SLOT/2, xc + CABLE_SLOT/2, p.side_in - 1.0, p.side_in + p.wall_t + 1.0, p.side_bot - 2.0, p.adapter_z)
        for sx in (-1, 1):                     # small lead-in chamfers at the slot's mouth
            s = s - hull(np.array([[xc + sx*(CABLE_SLOT/2 - 0.01), y, z] for y in (p.side_in - 1.0, p.side_in + p.wall_t + 1.0) for z in (p.side_bot - 2.0, p.side_bot + 1.2)] +
                                  [[xc + sx*(CABLE_SLOT/2 + 1.2), y, p.side_bot - 2.0] for y in (p.side_in - 1.0, p.side_in + p.wall_t + 1.0)]))
    # D-pegs (flat on top), long enough to carry the keeper and, behind it, a vertical hole for its filament pin
    xp = pin_x(p)
    for sy in (-1, 1):
        peg = cyl_x(PEG_R, xp - PIN_HOLE/2 - 1.6, p.hood_rear + 0.5, sy*PEG_Y, PEG_Z)
        peg = peg - cube(p.hood_rear - 20, p.hood_rear + 1, sy*PEG_Y - 4, sy*PEG_Y + 4, PEG_Z + PEG_R - 0.6, PEG_Z + 5)
        peg = peg - m3.Manifold.cylinder(2*PEG_R + 2, PIN_HOLE/2, PIN_HOLE/2, 32).translate((xp, sy*PEG_Y, PEG_Z - PEG_R - 1))
        s = s + peg
    return s


def keeper_x(p: Params):
    """The keeper's back and front faces (hood frame x): 0.4 mm behind the board."""
    x1 = p.hood_rear - p.board_t - 0.4; return x1 - KEEPER_T, x1


def pin_x(p: Params):
    """The filament pins' axis (hood frame x): just behind the keeper."""
    return keeper_x(p)[0] - PIN_GAP - PIN_D/2


def keeper(p: Params):
    """Plate behind the board on the two pegs (hood frame, behind the board), locked on by the filament pins."""
    x0, x1 = keeper_x(p)
    k = cube(x0, x1, -PEG_Y - 7, PEG_Y + 7, PEG_Z - 6, PEG_Z + 6 + KEEPER_UP)
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
    overlap (mm3) at the nominal position, the worst overlap while the joined pair rises in from below (with the catch's
    fingers bent out of its way), how far the pair can drop before the catch holds it, how far the head can be pulled
    forward (toward the duck) before the hood stops it, and the strain at the fingers' roots while they are bent.
    The adapter is modelled as it is (adapter_solid: rounded corners), not as the simulator's box."""
    hood_part = hood_part or hood(p)
    bend = CATCH_HOOK - 0.2                     # a hook reaches this far past the adapter's side (the walls clear it by 0.2 mm)
    flexed = hood(p, flex=bend + 0.05)
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug; ed, el, xe = p.angle_elbow
    x0 = p.adapter_face_x - pl; r = at/2; xa = x0 - (aw - r)          # adapter back face; body axis (hood x)
    def cyl_y(r0, r1, y0, y1, x, z): return m3.Manifold.cylinder(y1 - y0, r0, r1, 48).rotate((-90, 0, 0)).translate((x, y0, z))
    def pair(dx=0.0, dz=0.0, cable=True):
        z = p.adapter_z + dz
        h = hull(np.array([[x0 + dx + xx, y, z + zz] for xx, y in ((0, -xe), (0, el - xe), (-(ed + 2.1), el - xe + 1.8), (-(ed + 2.1), -xe)) for zz in (-r, r)]))
        h = h + cyl_y(r, r, -xe, al - 8.8 - xe, xa + dx, z) + cyl_y(r, 2.9, al - 8.8 - xe, al - xe, xa + dx, z)
        if cable: h = h + cyl_y(2.0, 2.0, al - xe, p.side_in + p.wall_t + 3.0, xa + dx, z)
        return h + adapter_solid(p).translate((dx, 0, dz))
    nominal = (pair() ^ hood_part).volume()
    path = max((pair(dz=dz) ^ flexed).volume() for dz in np.arange(-45.0, 0.01, 0.5))
    drop = next((dz for dz in np.arange(0.05, 3.0, 0.05) if (pair(dz=-dz, cable=False) ^ hood_part).volume() > 0.01), None)
    travel = next((dx for dx in np.arange(0.05, 5.0, 0.05) if (pair(dx=dx, cable=False) ^ hood_part).volume() > 0.01), None)
    strain = 1.5*CATCH_T*bend/catch_geometry(p)["free_length"]**2   # a cantilever's root strain, its tip bent by `bend`
    return dict(nominal_mm3=round(nominal, 4), insertion_mm3=round(path, 4), drop_mm=None if drop is None else round(float(drop), 2),
                forward_travel_mm=None if travel is None else round(float(travel), 2), catch_strain_pct=round(float(100*strain), 2))


def keeper_fit(p: Params):
    """How the keeper holds the hood on the board, over every position the hood can take: lifted 0 .. drop + lift above
    its seats, +-float sideways and rolled up to 10 degrees (its pegs move independently in the window). The board's
    window and V seats and the keeper's holes go straight through along x, so this works on y-z sections: the keeper's
    area that bears on solid board (mm2), the worst case, and the filament pins' place behind the keeper."""
    def yz(m):
        b = m.bounding_box(); return m.rotate((0, -90, 0)).slice((b[0] + b[3])/2)     # section coords: (-z, y)
    B, K = yz(board(p)), yz(keeper(p))
    travel = p.hood_drop + p.hood_lift; worst = None
    for lift in np.arange(0.0, travel + 1e-6, 0.5):
        for roll in (-10.0, -5.0, 0.0, 5.0, 10.0):
            dz = PEG_Y*np.tan(np.radians(abs(roll)))                     # each peg sits this much above/below the middle
            if lift - dz < -1e-6 or lift + dz > travel + 1e-6: continue   # a peg would leave its seat or the window
            k = K.translate((PEG_Z, 0)).rotate(roll).translate((-PEG_Z, 0))
            for dy in np.linspace(-p.hood_float_y, p.hood_float_y, 9):
                a = (k.translate((-(p.z0 - p.hood_drop + lift), dy)) ^ B).area()
                if worst is None or a < worst[0]: worst = (a, lift, dy, roll)
    x0, _ = keeper_x(p)
    return dict(min_bearing_mm2=round(float(worst[0]), 1), worst_at=dict(lift_mm=float(worst[1]), side_mm=float(worst[2]), roll_deg=worst[3]),
                pin_gap_mm=round(float(x0 - (pin_x(p) + PIN_D/2)), 2), pin_wall_mm=round(float(PEG_R*2 - PIN_HOLE)/2, 2))


def keeper_problems(kf):
    """What keeper_fit found wrong: the keeper must always bear on the board (at least 100 mm2) and its pins must sit
    right behind it."""
    bad = []
    if kf["min_bearing_mm2"] < 100.0: bad.append(f"the keeper can come through the board's window ({kf['worst_at']})")
    if not (0.05 <= kf["pin_gap_mm"] <= 0.5): bad.append("the filament pins don't sit right behind the keeper")
    return bad


def fit_problems(fit):
    """What head_fit found wrong: the pair must sit free, go in from below past the bent fingers, be held within 0.3 mm
    when it drops and be stopped within 0.5 mm when the magnet pulls it forward; the fingers bend under 1 % strain."""
    bad = []
    if fit["nominal_mm3"] > 0.01: bad.append("the cable head or adapter overlaps the hood")
    if fit["insertion_mm3"] > 0.01: bad.append("the cable head and adapter can't go in from below")
    if fit["drop_mm"] is None or fit["drop_mm"] > 0.3: bad.append("the catch doesn't hold the adapter")
    if fit["forward_travel_mm"] is None or fit["forward_travel_mm"] > 0.5: bad.append("the rib doesn't stop the cable head")
    if fit["catch_strain_pct"] > 1.0: bad.append("the catch's fingers bend too far")
    return bad


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
    ap.add_argument("--check", action="store_true", help="fail unless every part is one watertight body that fits a 256 mm bed and the cable head fits the hood and its catch")
    a = ap.parse_args()
    info = export_all(Params(), a.out)
    print(json.dumps(info, indent=1))
    if a.check:
        bad = [k for k, v in info.items() if not (v["watertight"] and v["bodies"] == 1 and max(v["extents_mm"]) <= BED_MM)]
        fit = head_fit(Params()); print("cable head fit:", json.dumps(fit))
        bad += fit_problems(fit)
        kf = keeper_fit(Params()); print("keeper:", json.dumps(kf))
        bad += keeper_problems(kf)
        if bad:
            sys.exit(f"check failed: {bad}")
        print(f"check passed: {len(info)} parts, each one watertight body within a {BED_MM:.0f} mm bed; the cable head fits, "
              f"goes in from below, the catch holds it within {fit['drop_mm']} mm and the rib stops it within "
              f"{fit['forward_travel_mm']} mm; the keeper bears at least {kf['min_bearing_mm2']} mm2 on the board wherever the hood is")

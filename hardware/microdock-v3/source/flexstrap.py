"""Flex strap (experiment, 2026-11-05): a print-in-place strap of interlocking slats on the principle of the flexi corn
cob. Each row's bars run, with play, through tunnels in the slats of the row below, so the strap hangs like a cord (the
bars pulled to the tops of their tunnels), telescopes when pushed (the bars slide down the tunnels), and can't come
apart (every tunnel is closed around its bar). Open, the joints are loose and the strap swings; closed, the slats of
each column sit end to end and the strap stands stiff instead of folding.

Own geometry (not the cob's): 5 columns of slats in a brick bond. Even rows have slats in columns 0, 2, 4 joined by bars
through the odd row below at columns 1 and 3; odd rows have slats in columns 1, 3 joined by a bar through the even row
below at column 2. The bottom and top rows are merged into end pieces with a hanging hole.

Frame: u along the strap (up when it hangs; u = 0 at the bottom row's slats), v across it, w through its thickness
(0..T). Prints flat, w up, as it is modelled: closed (every bar at the bottom of its tunnel, cxy above the floor)."""
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class Strap:
    T: float = 5.0          # thickness (w)
    skin: float = 1.0       # the slats' skins over and under each tunnel (w)
    a: float = 2.8          # a slat's width (v)
    cxy: float = 0.3        # clearance in the print's plane
    cz: float = 0.2         # clearance through the thickness (print Z: one 0.2 mm layer)
    hb: float = 1.8         # a bar's height (u): 1.0 let rows stay twisted after an off-centre lift, 2.6 resisted the lift (0.8 N)
    e: float = 0.4          # a bar's offset above its slats' bottoms
    play: float = 3.0       # each joint's travel, closed (as printed) to open (hanging)
    t_top: float = 0.8      # a slat's material above its tunnel (u)
    rows: int = 4           # rows between the two end pieces
    end_h: float = 9.0      # the bottom end piece beyond its slats (u)
    end_h_top: float = 12.0 # the top end piece beyond its slats (u)
    hole: float = 4.2       # the bottom end piece's hole (through w): a hook or a hood tab's pin
    hole_top: float = 7.5   # the top end piece's hole: hangs on a pencil
    marks: int = 0          # notches in the top end piece's top edge (tells test prints apart)
    top_inset: float = 0.0  # > 0: the top hole this far below the top end piece's top (else in its middle)
    top_end: str = "hole"   # "hole", or "bar": a cross-bar with a rounded underside that rests in a groove (no pin)
    bar_len: float = 21.0   # the bar's length (v): it spans a slot and rests either side of it
    bar_up: float = 9.0     # the bar's axis above the top piece's lower end (u); the bar's top is 1 mm above its axis
    bottom_end: str = "hole"   # "hole", or "knuckle": a round knuckle (diameter T) on a neck, that clicks into a socket
    neck_t: float = 3.0     # the knuckle's neck (w)
    knuckle_down: float = 5.0  # the knuckle's axis below the bottom row's slats (u)
    knuckle_flat: float = 0.0  # a flat this deep on the knuckle's print-bed side: the knuckle and its neck sit this much
                               # lower (w), so the knuckle stands on the bed instead of touching it along a line
    side_gap: float = 0.0   # > 0: the gap between neighbouring slats across the strap (else cxy): wider lets the strap
                            # bend in its own plane (front to back on the hood) about as freely as through its thickness
    cols: int = 5
    soft: bool = False      # rounded slats and end pieces (outer edges only; the tunnels, bars and gaps unchanged)

    @property
    def cp(self): return self.a + (self.side_gap or self.cxy)
    @property
    def P(self): return self.e + self.hb + self.play + self.t_top + self.cxy    # row pitch, closed
    @property
    def K(self): return 2*self.P - self.cxy                                    # a slat's length (u)
    @property
    def R(self): return self.rows + 1                                          # the top row's index
    def vc(self, c): return (c - (self.cols - 1)/2)*self.cp
    def width(self): return self.cols*self.a + (self.cols - 1)*(self.side_gap or self.cxy)
    def __post_init__(self):
        if self.top_end == "bar": self.end_h_top = self.bar_up + 1.0
        if self.bottom_end == "knuckle": self.end_h = self.knuckle_down + self.T/2
    def hole_u(self):
        """The strap's two pivots (u): the holes' centres, the bar's axis or the knuckle's axis."""
        top = self.R*self.P + self.P + self.end_h_top
        bot = -self.knuckle_down if self.bottom_end == "knuckle" else -self.end_h/2
        if self.top_end == "bar": return (bot, self.R*self.P + self.P + self.bar_up)
        return (bot, top - self.top_inset if self.top_inset > 0 else top - self.end_h_top/2)
    def length_closed(self): return self.hole_u()[1] - self.hole_u()[0]       # hole to hole
    def length_open(self): return self.length_closed() + self.R*self.play


def row_boxes(s: Strap, r: int):
    """Row r's solid boxes ((u0, v0, w0), (u1, v1, w1)); row 0 and row R carry the end pieces."""
    U = r*s.P; out = []
    cols = [c for c in range(s.cols) if c % 2 == r % 2]
    up_cols = [c for c in range(s.cols) if c % 2 == (r + 1) % 2]
    tunnels = {c + 1 for c in up_cols[:-1]} if r < s.R else set()      # where the next row's bars pass through
    for c in cols:
        v0, v1 = s.vc(c) - s.a/2, s.vc(c) + s.a/2
        if c not in tunnels:
            out.append(((U, v0, 0.0), (U + s.K, v1, s.T))); continue
        t0, t1 = U + s.P + s.e - s.cxy, U + s.P + s.e + s.hb + s.play
        out += [((U, v0, 0.0), (t0, v1, s.T)), ((t1, v0, 0.0), (U + s.K, v1, s.T)),
                ((t0, v0, 0.0), (t1, v1, s.skin)), ((t0, v0, s.T - s.skin), (t1, v1, s.T))]
    if r > 0:                                            # the bars through the row below
        for c0, c1 in zip(cols[:-1], cols[1:]):
            out.append(((U + s.e, s.vc(c0), s.skin + s.cz), (U + s.e + s.hb, s.vc(c1), s.T - s.skin - s.cz)))
    v0, v1 = s.vc(0) - s.a/2, s.vc(s.cols - 1) + s.a/2
    if r == 0:                                           # the bottom end piece, up to the row above's slats
        out.append(((-s.end_h, v0, 0.0), (s.P - s.cxy, v1, s.T)))
    if r == s.R:                                         # the top end piece, above the row below's slats
        out.append(((U + s.P, v0, 0.0), (U + s.P + s.end_h_top, v1, s.T)))
    return out


def rbox(u0, u1, v0, v1, w0, w1, r_top=0.6, r_uv=(0.8, 0.8), c=0.6, n=6):
    """A box (strap frame u, v, w) with its top edges (w1) rounded r_top, its corners in the u-v plane rounded r_uv
    (at u0, at u1) and its bottom edges (w0, on the print bed) chamfered c at 45 deg: the convex hull of spheres along
    the top corners' arcs and the rounded outlines at w0 + c (full) and w0 (inset c)."""
    import manifold3d as m3
    sph = m3.Manifold.sphere(r_top, 16); ks = []; ra, rb_ = r_uv
    pts = []
    for su, sv, a0, rr in ((1, 1, math.pi, ra), (-1, 1, 1.5*math.pi, rb_), (-1, -1, 0.0, rb_), (1, -1, 0.5*math.pi, ra)):
        bu, bv = (u0 if su > 0 else u1), (v0 if sv > 0 else v1)              # the box's corner
        re = max(rr, r_top)                                                  # the spheres stay inside the box
        for a in np.linspace(a0, a0 + math.pi/2, n):
            ks.append(sph.translate((bu + su*re + (re - r_top)*math.cos(a), bv + sv*re + (re - r_top)*math.sin(a), w1 - r_top)))
            cu, cv = bu + su*rr, bv + sv*rr
            pts += [(cu + rr*math.cos(a), cv + rr*math.sin(a), w0 + c), (cu + max(rr - c, 0.05)*math.cos(a), cv + max(rr - c, 0.05)*math.sin(a), w0)]
    return m3.Manifold.batch_hull(ks + [m3.Manifold.hull_points(pts)])


def row_cad(s: Strap, r: int):
    import manifold3d as m3
    if not s.soft:
        parts = [m3.Manifold.cube(tuple(np.subtract(hi, lo))).translate(tuple(lo)) for lo, hi in row_boxes(s, r)]
        m = parts[0]
        for q in parts[1:]: m = m + q
    else:                                                # the same solids, their outer edges rounded
        U = r*s.P; out = []
        cols = [c for c in range(s.cols) if c % 2 == r % 2]
        up_cols = [c for c in range(s.cols) if c % 2 == (r + 1) % 2]
        tunnels = {c + 1 for c in up_cols[:-1]} if r < s.R else set()
        for c in cols:
            v0, v1 = s.vc(c) - s.a/2, s.vc(c) + s.a/2
            sl = rbox(U, U + s.K, v0, v1, 0.0, s.T)
            if c in tunnels:
                t0, t1 = U + s.P + s.e - s.cxy, U + s.P + s.e + s.hb + s.play
                sl = sl - m3.Manifold.cube((t1 - t0, s.a + 2, s.T - 2*s.skin)).translate((t0, v0 - 1, s.skin))
            out.append(sl)
        if r > 0:
            for c0, c1 in zip(cols[:-1], cols[1:]):
                lo = (U + s.e, s.vc(c0), s.skin + s.cz); hi = (U + s.e + s.hb, s.vc(c1), s.T - s.skin - s.cz)
                out.append(m3.Manifold.cube(tuple(np.subtract(hi, lo))).translate(lo))
        W0, W1, R = s.vc(0) - s.a/2, s.vc(s.cols - 1) + s.a/2, 3.0
        if r == 0:                                       # the bottom block: its outer (lower) corners well rounded; its
            u_lo = -1.0 if s.bottom_end == "knuckle" else -s.end_h   # top stays at the rows' gap (no fusing)
            out.append(rbox(u_lo, s.P - s.cxy, W0, W1, 0.0, s.T, r_top=1.0, r_uv=(R, 0.6)))
        if r == s.R:                                     # the top block: its outer (upper) corners well rounded
            out.append(rbox(U + s.P, U + s.P + s.end_h_top, W0, W1, 0.0, s.T, r_top=1.0, r_uv=(0.6, R)))
        m = m3.Manifold.batch_boolean(out, m3.OpType.Add)
    def along_v(r_, u_c, L, w_c=None):                   # a cylinder along v, axis at (u_c, w_c: mid-thickness)
        return m3.Manifold.cylinder(L, r_, r_, 64).rotate((90, 0, 0)).translate((u_c, L/2, s.T/2 if w_c is None else w_c))
    if r == 0 and s.bottom_end == "knuckle":            # the knuckle on its neck, instead of the block's lower part
        uk = -s.knuckle_down; W = s.width(); v0 = s.vc(0) - s.a/2
        if not s.soft: m = m - m3.Manifold.cube((s.end_h, W + 2, s.T + 2)).translate((-s.end_h - 1.0, v0 - 1, -1.0))
        wk = s.T/2 - s.knuckle_flat                      # the knuckle's axis (w), lowered onto its flat
        m = m + m3.Manifold.cube((-1.0 - uk + 0.01, W, s.neck_t)).translate((uk, v0, wk - s.neck_t/2))
        k = along_v(s.T/2, uk, W, wk)
        if s.knuckle_flat > 0: k = k ^ m3.Manifold.cube((s.T + 2, W + 2, s.T)).translate((uk - s.T/2 - 1, v0 - 1, 0.0))
        m = m + k
    elif r == 0:
        m = m - m3.Manifold.cylinder(s.T + 2, s.hole/2, s.hole/2, 48).translate((s.hole_u()[0], 0.0, -1.0))
    if r == s.R and s.top_end == "bar":                 # the cross-bar: its rounded underside rests in the groove
        uc = s.hole_u()[1]; top = uc + 1.0; L = s.bar_len
        m = m + m3.Manifold.cube((top - uc, L, s.T - 0.6)).translate((uc, -L/2, 0.3)) + along_v(s.T/2, uc, L)   # inside the
        # bar's round: its corners would catch the groove's edge as the bar rocks
    elif r == s.R:
        m = m - m3.Manifold.cylinder(s.T + 2, s.hole_top/2, s.hole_top/2, 48).translate((s.hole_u()[1], 0.0, -1.0))
    if r == s.R:
        top = s.R*s.P + s.P + s.end_h_top
        for k in range(s.marks):                         # 1.6 mm wide, 1.2 mm deep notches in the top edge
            vk = (k - (s.marks - 1)/2)*3.2
            m = m - m3.Manifold.cube((2.4, 1.6, s.T + 2)).translate((top - 1.2, vk - 0.8, -1.0))
    return m


def strap_cad(s: Strap):
    """The whole strap (one row per body), in its own frame."""
    m = row_cad(s, 0)
    for r in range(1, s.R + 1): m = m + row_cad(s, r)
    return m


def min_gap(s: Strap):
    """The smallest clearance between different rows' boxes as printed (mm): every pair of rows must stay apart."""
    def gap(A, B):
        d = np.maximum(np.maximum(np.subtract(B[0], A[1]), np.subtract(A[0], B[1])), 0.0)
        return float(np.linalg.norm(d)) if np.any(d > 0) else -float(np.min(np.minimum(np.subtract(A[1], B[0]), np.subtract(B[1], A[0]))))
    rows = [row_boxes(s, r) for r in range(s.R + 1)]
    best = 1e9
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            for A in rows[i]:
                for B in rows[j]: best = min(best, gap(A, B))
    return best


WEIGHT_G = 11.0                  # a hood corner's share of the hood (45 g / 4)


def hook_weight(s: Strap, grams=WEIGHT_G):
    """A test weight with a J hook that hangs through the strap's bottom hole. Prints flat as modelled (z up): the block
    on the bed, the hook a 2.8 mm plate on the bed beside it. Print it at 100 % infill (PLA 1.24 g/cm3)."""
    import manifold3d as m3
    t = 2.8                                              # the hook's bar: 2.8 x 2.8 (diagonal 3.96 < the 4.2 hole)
    span = 2*t + s.T + 1.4                               # the J's gap takes the strap's 5 mm end piece with 1.4 to spare
    hook = (m3.Manifold.cube((t, 16.0, t)).translate((-t/2, 0.0, 0.0)) +                 # stem out of the block
            m3.Manifold.cube((span, t, t)).translate((-span + t/2, 16.0 - t, 0.0)) +     # over the top, through the hole
            m3.Manifold.cube((t, 7.0, t)).translate((-span + t/2, 16.0 - 7.0, 0.0)))     # the J's tip, pointing back
    vol_hook = hook.volume()
    side = ((grams/1.24*1000 - vol_hook)/1.0)**(1/3)                                     # a cube for the rest
    block = m3.Manifold.cube((side, side, side)).translate((-side/2, -side, 0.0))
    return block + hook

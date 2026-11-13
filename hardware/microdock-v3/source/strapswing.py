"""The strap dock (experiment, 2026-11-05, before the egg dock; MicroDock v3 uses its egg outline, base and helpers): the rods version (swing.Rods: the hood's tooth and the hanger's
slot, the board and its tongues) with the hood hanging on four flex straps (flexstrap.py, the test print's strap B
joint) instead of ball-ended rods. No pins, no tools:

- Top, drop-in: a strap's top piece rises through a slot in the hanger plate and ends in a cross-bar whose rounded
  underside rests in a groove on the plate. The bar is no higher than the rods' balls were, so the dozing head keeps its
  clearance, and a strap pushed past its travel just lifts its bar out of the groove and settles back.
- Bottom, slide-in: a round knuckle on the strap's bottom end slides into a socket on a tab out from the hood's side wall
  (a bore along x, open at its rear end, closed at its front) and turns freely in it. The hood rests on the knuckle with
  the socket's solid top, slotted for the strap's neck; a sprung ridge in the floor at the entrance clicks in behind the
  knuckle. Hold the hood a little forward of where it hangs and push it back: all four straps click in at once.
- The bar rocks in its groove and the knuckle turns in its socket, both about x: the strap
  swings sideways at its ends and in its joints, and front to back in its joints.
- Straps at |y| 54 like the rods; pairs at world x 11 and 44 (a strap is 15.6 mm wide along x: the rear pair clear of the
  board, the front pair far enough forward that a stiff cable can't tip the hood nose-down onto it).
Frames and units as swing.py (mm; world x away from the board's face, hood frame from hood_home)."""
import math
from dataclasses import dataclass
import numpy as np
import swing, flexstrap as fs

STRAP_JOINT = dict(cxy=0.4, cz=0.4, hb=1.8, play=3.0, a=2.8, T=5.0, rows=2)   # strap B of the test print (the fluid one,
# two notches), two rows shorter: 3 joints, 9 mm of travel, 48.6 mm pivot to pivot hanging open, like the rods' 49 (with
# rows=3 the straps were 57.5 mm and a stiff cable could hold the hood ~20 mm aside: longer hangers centre it more
# weakly). The knuckle sits 29.4 mm up the hood (the rods' tabs: 30); past its travel a strap lifts its bar off the plate.


@dataclass
class Straps(swing.Params):
    strap_xw: tuple = (11.0, 42.0)  # the pairs' centres along the board (world x), rear and front (the front bars'
                                    # channels end 4.5 mm inside the hanger's arm tips, under its 6 mm round)
    strap_y: float = 54.0           # |y| of the straps' mid-thickness
    bar_half_w: float = 68.0        # the hanger plate carries the slots at |y| 54 (its arms reach |y| 68)
    slot_y: float = 0.5             # a slot's clearance around a strap's top piece through its thickness, at the top
    slot_x: float = 0.5             # and at its ends
    slot_flare: float = 30.0        # the slot widens downwards so the top piece rocks with its bar about x up to this (deg)
    gap_below: float = 1.0          # the plate's underside above the closed strap's second-highest row
    lug_t: float = 2.0              # the hood's lugs either side of a strap's bottom piece
    lug_gap: float = 0.4
    lug_hx: float = 4.5             # a lug's half length along x
    tab_t: float = 3.0              # the tab joining a strap's lugs to the side wall, below the strap's bottom piece

    bar_z: tuple = (122.0, 131.5)   # the hanger plate: its top level with the bars' tops (the bars sit flush in it)
    groove: float = 5.0             # the bar's channel in the hanger's top (depth: the bar sunk flush; its axis at 129)
    socket_wall: float = 1.5        # the click socket around the knuckle
    neck_t: float = 2.0             # the strap's neck above its knuckle (through its thickness)
    knuckle_flat: float = 0.25      # the knuckle's flat on the print bed (flexstrap.Strap.knuckle_flat): round, it touched the
                                    # bed along a line and printed lumpy (2026-11-07). 0.25 keeps it bearing on both edges of
                                    # the socket's throat (2.1) with 0.15 mm to spare on the flat side
    knuckle_down: float = 7.5       # the knuckle's axis below the strap's bottom slats: the block above the neck clears the
                                    # socket's roof with the knuckle turned knuckle_turn
    throat: float = 2.1             # the socket's neck slot: half width where it meets the bore (under the knuckle's 2.5
                                    # radius, so the knuckle can't pull through), widening upwards at knuckle_turn
    knuckle_turn: float = 30.0      # how far the knuckle turns in its socket about x before the neck meets the slot (deg)
    def strap(self):
        # the bar's axis sits so that the plate's underside is gap_below above the top piece's lower end
        bar_up = self.gap_below + (self.bar_z[1] - self.bar_z[0]) - self.groove + STRAP_JOINT["T"]/2
        return fs.Strap(**STRAP_JOINT, top_end="bar", bottom_end="knuckle", bar_up=bar_up, neck_t=self.neck_t,
                        knuckle_down=self.knuckle_down, knuckle_flat=self.knuckle_flat, soft=self.strap_soft)
    def pin_top_z(self): return self.bar_z[1] - self.groove + self.strap().T/2   # world: the bar's axis, resting in the groove
    def strap_len(self):                                       # pin to pin, hanging open
        return self.strap().length_open()
    def strap_sites(self):
        """(name, hood-frame bottom pin, world top pin) per strap."""
        h = self.hood_home(); out = []
        for i, xw in enumerate(self.strap_xw):
            for sy in (1, -1):
                pw = np.array([xw, sy*self.strap_y, self.pin_top_z()])
                ph = pw - h - np.array([0.0, 0.0, self.strap_len()])
                out.append((f"strap_{'rf'[i]}{'lr'[sy < 0]}", ph, pw))
        return out
    cord_sites = strap_sites                                   # for the simulator's tendons (cord-like: they only pull)
    def cord_len(self): return self.strap_len()
    strap_chains: bool = False      # simulator: each strap as its chain of joints (else a pull-only tendon)
    strap_soft: bool = False        # the straps' rounded slats and end pieces (flexstrap.Strap.soft). Off: printed, the
                                    # plain straps (the 2026-11-05 print set's) moved more fluidly than the rounded ones
    egg_half: bool = False          # the board: the flat egg (False) or the egg half with its dome behind (True); the
                                    # simulator is the same for both (nothing reaches behind the board)
    # the strap's joints as the simulator's chain sees them (deg): bending through its thickness is free (the bars spin
    # in their tunnels), along its face the interleaved slats stop it at ~14 deg a joint (strap_sim, strap B with 2 rows:
    # 4.6 mm of S-bend between held ends; strap A with 3 rows: 6.4 mm, ~10 deg), twist ~7 deg a joint (the bars' 0.4 mm
    # in their tunnels, 3.2 mm either side of the middle); each joint slides 3 mm shut
    chain_out: float = 45.0
    chain_in: float = 14.0
    chain_twist: float = 7.0
    chain_top_tilt: float = 4.0     # the top piece in its slot: front to back and twist
    chain_knuckle_tilt: float = 3.0 # the knuckle in its bore: front to back

    def chain_xml(self):
        """(bodies, equalities): MJCF for the four strap chains, hanging from the world, each welded to the hood at its
        knuckle. Joints in body frames aligned with the world: x across the strap (its width), y through its thickness."""
        s = self.strap(); P_open = s.P + s.play
        # pivot to the top joint, row pitch, bottom joint to the knuckle (open, hanging)
        top = (s.R*s.P + s.P + s.bar_up) - ((s.R - 0)*s.P + s.e + s.hb/2)
        bot = s.length_open() - top - (s.R - 1)*P_open
        segs = [top] + [P_open]*(s.R - 1) + [bot]
        hw, ht = s.width()/2e3, s.T/2e3
        def J(name, kind, axis, rng, arm, damp):
            r = f'range="{rng[0]} {rng[1]}" limited="true" solreflimit="0.01 1" solimplimit="0.99 0.999 0.0005"'
            return f'<joint name="{name}" type="{kind}" axis="{axis}" {r} armature="{arm}" damping="{damp}"/>'
        rad = np.radians
        bodies, eqs = [], []
        for name, ph, pw in self.strap_sites():
            n = name[6:]; xml = []
            box = lambda L: f'<geom type="box" size="{hw:.5f} {ht:.5f} {L/2e3:.5f}" pos="0 0 {-L/2e3:.5f}" rgba="0.95 0.6 0.2 1" contype="0" conaffinity="0" density="1240"/>'
            xml.append(f'<body name="sc_{n}_0" pos="{pw[0]/1e3:.5f} {pw[1]/1e3:.5f} {pw[2]/1e3:.5f}">')
            xml.append(J(f"sc_{n}_lift", "slide", "0 0 1", (0, 0.03), 0.002, 0.05))
            xml.append(J(f"sc_{n}_tx", "hinge", "1 0 0", (-rad(self.slot_flare), rad(self.slot_flare)), 2e-6, 2e-4))
            xml.append(J(f"sc_{n}_ty", "hinge", "0 1 0", (-rad(self.chain_top_tilt), rad(self.chain_top_tilt)), 2e-6, 2e-4))
            xml.append(J(f"sc_{n}_tz", "hinge", "0 0 1", (-rad(self.chain_top_tilt), rad(self.chain_top_tilt)), 2e-6, 2e-4))
            xml.append(box(segs[0]))
            xml.append(f'<geom type="capsule" fromto="{-s.bar_len/2e3:.5f} 0 0 {s.bar_len/2e3:.5f} 0 0" size="{ht:.5f}" rgba="0.95 0.6 0.2 1" contype="0" conaffinity="0" density="1240"/>')
            depth = 1
            for k, L in enumerate(segs[1:], 1):
                xml.append(f'<body name="sc_{n}_{k}" pos="0 0 {-segs[k - 1]/1e3:.5f}">')
                xml.append(J(f"sc_{n}_s{k}", "slide", "0 0 1", (0, s.play/1e3), 0.002, 0.05))
                xml.append(J(f"sc_{n}_o{k}", "hinge", "1 0 0", (-rad(self.chain_out), rad(self.chain_out)), 2e-6, 2e-4))
                xml.append(J(f"sc_{n}_i{k}", "hinge", "0 1 0", (-rad(self.chain_in), rad(self.chain_in)), 2e-6, 2e-4))
                xml.append(J(f"sc_{n}_w{k}", "hinge", "0 0 1", (-rad(self.chain_twist), rad(self.chain_twist)), 2e-6, 2e-4))
                xml.append(box(L)); depth += 1
            # the knuckle in the hood's socket: turns about x, barely about y
            xml.append(f'<body name="sc_{n}_kn" pos="0 0 {-segs[-1]/1e3:.5f}">')
            xml.append(J(f"sc_{n}_kx", "hinge", "1 0 0", (-rad(self.knuckle_turn), rad(self.knuckle_turn)), 2e-6, 2e-4))
            xml.append(J(f"sc_{n}_ky", "hinge", "0 1 0", (-rad(self.chain_knuckle_tilt), rad(self.chain_knuckle_tilt)), 2e-6, 2e-4))
            xml.append(f'<geom type="capsule" fromto="{-hw:.5f} 0 0 {hw:.5f} 0 0" size="{ht:.5f}" rgba="0.95 0.6 0.2 1" contype="0" conaffinity="0" density="1240"/>')
            xml.append("</body>"*(depth + 1))
            bodies.append("".join(xml))
            eqs.append(f'<weld body1="sc_{n}_kn" body2="hood" solref="0.01 1" solimp="0.99 0.999 0.0005"/>')
        return bodies, eqs

    def static_pieces(self): return static_pieces(self)       # the simulator's base (rounded) and egg board
    def hanger_extra_boxes(self):                              # the simulator's hanger: the plate's front sides
        import dock as dk
        return {f"ear_{k}": dk.box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]) for k, (lo, hi) in zip("lr", ears(self))}
    def hood_extra_pieces(self):
        """The simulator's hood additions (hood frame convex pieces): each strap's tab, and its socket as a solid box."""
        import dock as dk, cad
        s = self.strap(); yo = self.side_in + self.wall_t; rb = s.T/2 + 0.15; out = {}
        xc = swing.cable_exit(self)[0]; slot = (xc - (cad.CABLE_SLOT/2 + 1.7), xc + (cad.CABLE_SLOT/2 + 1.7))
        for name, ph, pw in self.strap_sites():
            sy = 1.0 if ph[1] > 0 else -1.0; x, ys, z = ph[0], sy*self.strap_y, ph[2]; zb = z - rb - 1.2
            x0, x1 = x - s.width()/2 - 2.4, x + (s.width() + 0.8)/2 + 1.6
            out[f"socket_{name[6:]}"] = dk.box(x0, x1, ys - rb - self.socket_wall, ys + rb + self.socket_wall, zb, z + rb + 1.4)
            a0, a1 = tab_x(x, sy, xc, slot)
            yin = ys - sy*(rb + self.socket_wall - 0.5)
            out[f"stab_{name[6:]}"] = np.array([[xx, sy*yy, zz] for xx in (a0, a1) for yy in (yo - 0.5, yo + 3.0) for zz in (zb, zb + 3.0)] +
                                               [[xx, yy, zz] for xx in (a0, a1) for yy in (yin - sy*3.0, yin) for zz in (zb, zb + 3.0)], float)
        if not FUNNEL.get("flat"): out.update(funnel_pieces(self))   # the rounded funnel (replaces the flat one's pieces)
        return out


FUNNEL = dict(fillet=16.0,                 # the flares bend out of the side walls on this radius (plan view)
              bot_mouth=17.0, top_mouth=43.6, # their bottom and top edges at the mouth (at the throat: the walls').
                                             # Full height to the front, as the flat flares: lowered to 19-37 mm, the
                                             # mouth let a duck 32 mm off a stiff-cable-held hood ride over the fin
                                             # (seed 43 trial 8 failed; 28/30 instead of 29/30)
              nose=4.0,                      # their front ends rounded in side view
              lip_bend=5.0, lip_corner=4.0)  # the lips: the skids' undersides bend up on this radius into the ramp;
                                             # the tips' corners rounded in plan
# The working surfaces stay where they were (dock.hood_pieces): the mouth 106 mm wide at x 16 and the flares' 42 deg
# slope, the skids' flat undersides, the lips' 6 mm rise to x -14; only the bend into each flare is rounded (it opens
# the throat by at most 1.2 mm) and the lips' ramps start 1.9 mm earlier, on a 5 mm bend.


def _funnel_geom(p):
    """The funnel's mid-surfaces (hood frame mm). Flares (right side; the left mirrors): the inner face's plan curve
    (wall -> fillet -> the old straight flare) as (points, material-side normals) and the side outline z_b(x), z_t(x).
    Lips: the underside's profile along x."""
    F = FUNNEL; t = p.wall_t; R = F["fillet"]; xt = p.wall_front
    phi = math.atan2(p.flare_out, p.flare_len); T = R*math.tan(phi/2)
    A = np.array([xt - T, p.side_in]); C = np.array([xt - T, p.side_in + R]); B = np.array([xt + T*math.cos(phi), p.side_in + T*math.sin(phi)])
    xm = xt + p.flare_len; E = np.array([xm, p.side_in + p.flare_out])
    pts, nrm = [], []
    for x in np.linspace(A[0] - 4.0, A[0], 4, endpoint=False): pts.append([x, p.side_in]); nrm.append([0.0, 1.0])
    for a in np.linspace(0.0, phi, 25):
        q = C + R*np.array([math.sin(a), -math.cos(a)]); pts.append(q); nrm.append((C - q)/R)
    for f in np.linspace(0.0, 1.0, 30)[1:]:
        pts.append(B + f*(E - B)); nrm.append([-math.sin(phi), math.cos(phi)])
    pts, nrm = np.array(pts), np.array(nrm)
    top1 = p.trunk_top + 3.0
    def sm(u): u = min(1.0, max(0.0, u)); return u*u*(3 - 2*u)
    def zb(x): return p.side_bot + (F["bot_mouth"] - p.side_bot)*sm((x - xt)/(xm - xt))
    def zt(x): return top1 - (top1 - F["top_mouth"])*sm((x - (xt + 4.0))/(xm - xt - 4.0))
    return dict(A=A, B=B, E=E, phi=phi, pts=pts, nrm=nrm, zb=zb, zt=zt, xm=xm, r=t/2)


def _lip_profile(p):
    """The lips' underside along x: flat (the skids') to x_a, a lip_bend arc up to the old ramp's slope, the ramp to its
    old top end (x neck_x, 6 mm up). Returns (x_a, list of (x, z, slope angle))."""
    F = FUNNEL; R = F["lip_bend"]; z0 = p.trunk_top; th = math.atan2(p.lip_rise, p.lip_len)
    dx_arc, dz_arc = R*math.sin(th), R*(1 - math.cos(th))
    dx_lin = (p.lip_rise - dz_arc)/math.tan(th); xa = p.neck_x - dx_lin - dx_arc
    prof = [(x, z0, 0.0) for x in np.linspace(xa - 6.0, xa, 4, endpoint=False)]
    for a in np.linspace(0.0, th, 10):
        prof.append((xa + R*math.sin(a), z0 + R*(1 - math.cos(a)), a))
    for f in np.linspace(0.0, 1.0, 8)[1:]:
        prof.append((xa + dx_arc + f*dx_lin, z0 + dz_arc + f*(p.lip_rise - dz_arc), th))
    return xa, prof


def _sphere_swept(grid, r, seg=12):
    """A plate swept by a sphere (radius r) over a mid-surface given as a grid of points [i][j] (3D): every quad's hull of
    four spheres, unioned: a plate 2r thick with fully rounded rims."""
    import manifold3d as m3
    sph = m3.Manifold.sphere(r, seg); out = []
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            out.append(m3.Manifold.batch_hull([sph.translate(tuple(map(float, q))) for q in (grid[i][j], grid[i + 1][j], grid[i][j + 1], grid[i + 1][j + 1])]))
    return m3.Manifold.batch_boolean(out, m3.OpType.Add)


def funnel_cad(p):
    """The funnel (hood frame): the two flares as curved fins bending out of the walls and the two lips as up-curved
    tips of the skids, each a plate with fully rounded rims (swept sphere over its mid-surface)."""
    import manifold3d as m3
    G = _funnel_geom(p); r = G["r"]; nose = FUNNEL["nose"]; parts = []
    mid = G["pts"] + r*G["nrm"]; sl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mid, axis=0), axis=1))]; S = sl[-1]
    for sy in (1, -1):
        grid = []
        for (x, y), sk in zip(mid, sl):
            u = max(0.0, sk - (S - nose)); cut = nose - math.sqrt(max(nose**2 - u**2, 0.0))   # the nose, rounded in side view
            z0, z1 = G["zb"](x) + r + cut, G["zt"](x) - r - cut
            if z1 < z0: z0 = z1 = (z0 + z1)/2
            grid.append([[x, sy*y, z] for z in np.linspace(z0, z1, 5)])
        parts.append(_sphere_swept(grid, r))
    xa, prof = _lip_profile(p); rl = 1.5; yi, yo = p.skid_y, p.side_in + p.wall_t; c = FUNNEL["lip_corner"]
    xs = [q[0] for q in prof]; x_end = xs[-1]
    for sy in (1, -1):
        grid = []
        for x, z, a in prof:
            u = max(0.0, x - (x_end - c)); cut = c - math.sqrt(max(c**2 - u**2, 0.0))   # the tip's corners rounded in plan
            zm, xm_ = z + rl*math.cos(a), x - rl*math.sin(a)                            # the mid-surface: rl above the underside
            ya, yb = yi + rl + cut, yo - rl - cut
            if yb < ya: ya = yb = (ya + yb)/2
            grid.append([[xm_, sy*yy, zm] for yy in np.linspace(ya, yb, 4)])
        parts.append(_sphere_swept(grid, rl))
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def funnel_pieces(p, n_flare=10, n_lip=6):
    """The simulator's funnel (hood frame convex pieces), replacing dock.hood_pieces' flares, lips, the walls' and skids'
    front ends: the flares in n_flare segments along their curve, the lips in n_lip segments along x."""
    import dock as dk
    G = _funnel_geom(p); r = G["r"]; pc = {}; yo = p.side_in + p.wall_t; top0, top1 = p.trunk_top, p.trunk_top + 3.0
    mid = G["pts"] + r*G["nrm"]; inner, outer = G["pts"], G["pts"] + 2*r*G["nrm"]
    idx = np.linspace(4, len(mid) - 1, n_flare + 1).astype(int)
    for sy, side in ((1, "L"), (-1, "R")):
        for k in range(n_flare):
            P = []
            for i in (idx[k], idx[k + 1]):
                for q in (inner[i], outer[i]):
                    P += [[q[0], sy*q[1], G["zb"](mid[i][0])], [q[0], sy*q[1], G["zt"](mid[i][0])]]
            pc[f"flare{side}" if k == 0 else f"flare{side}_{k}"] = np.array(P, float)
        y0, y1 = sorted((sy*p.side_in, sy*yo))
        pc[f"wall{side}"] = dk.box(p.hood_rear, G["A"][0] + 0.5, y0, y1, p.side_bot, top1)          # ends where the bend starts
    xa, prof = _lip_profile(p)
    ii = np.linspace(0, len(prof) - 1, n_lip + 1).astype(int)
    for sy, side in ((1, "L"), (-1, "R")):
        y0, y1 = sorted((sy*p.skid_y, sy*yo))
        pc[f"skid{side}"] = dk.box(p.brim_x - 0.5, xa, y0, y1, top0, top1)
        for k in range(n_lip):
            P = []
            for i in (ii[k], ii[k + 1]):
                x, z, a = prof[i]
                for yy in (y0, y1): P += [[x, yy, z], [x - 3.0*math.sin(a), yy, z + 3.0*math.cos(a)]]
            pc[f"lip{side}" if k == 0 else f"lip{side}_{k}"] = np.array(P, float)
    return pc


def _sphere_hull(points, r, seg=20):
    """The hull of spheres (radius r) at the points pulled r toward their centroid on each axis: about the points' own
    hull, every edge rounded r."""
    import manifold3d as m3
    P = np.asarray(points, float); c = P.mean(0)
    Q = P - np.sign(P - c)*np.minimum(r, np.abs(P - c))
    sph = m3.Manifold.sphere(r, seg)
    return m3.Manifold.batch_hull([sph.translate(tuple(q)) for q in Q])


def tab_x(x, sy, xc, slot):
    """A socket's tab along x (hood frame): 12 mm centred on the strap, or, where that would cross the left wall's cable
    slot, all of it behind (or in front of) the slot, so the cable can leave the wall and turn down between the tab and
    the socket's front half."""
    a0, a1 = x - 6.0, x + 6.0
    if sy > 0 and a0 < slot[1] and a1 > slot[0]:
        a0, a1 = (a0 - 2.0, slot[0]) if x < xc else (slot[1], a1 + 2.0)
    return a0, a1


def socket_blockers(p):
    """Hood frame boxes (lo, hi) for the slicer's support blockers: each socket's bore and neck slot. The hood prints
    upside down with supports from the plate, and the neck slot opens towards the plate, so without these supports would
    grow up through it and fill the bore. The roof around the slot's mouth stays supported."""
    s = p.strap(); rb = s.T/2 + 0.15; out = []
    for name, ph, pw in p.strap_sites():
        sy = 1.0 if ph[1] > 0 else -1.0; x, ys, z = ph[0], sy*p.strap_y, ph[2]
        x0, x1 = x - s.width()/2 - 2.4, x + (s.width() + 0.8)/2
        out.append(((x0 - 1.5, ys - rb - 0.3, z - rb - 0.6), (x1 + 0.3, ys + rb + 0.3, z + rb + 1.4 - 0.3)))
    return out


HANGER_ARMS = (37.0, 68.0, 18.0, 57.0)   # the hanger's horseshoe: its arms' inner and outer |y|, the arch's back (x)
                                         # and the arms' reach (x, past the front bars' channels). The dozing head drops
                                         # past the plate's middle (v2 video's doze): 1.8-3.9 mm clear for a duck docked
                                         # up to 20 mm off centre (the earlier wavy front edge: 1.5-3.5)
HANGER_R = (3.0, 3.0, 4.0)               # the hanger's corner radii: where the arch meets the arms, at the board, and
                                         # the tooth slot's front corners
EDGE_R = 2.0                             # the egg board's front edge round
HANGER_EDGE = (6.0, 1.2)                 # the hanger's top edge round and bottom edge chamfer (the edge along the board
                                         # stays square)
HOOD_EDGE_R = 1.5                        # the hood's top edges' round
TONGUE_FIT = (0.2, 0.3, 1.0)             # the hanger's tongues inside the board's slots (the slots unchanged): off each
                                         # side, off the top (the bottom stays on the bed) and the tip's lead-in chamfer
                                         # (mm). Printed with only the slots' 0.15 / 0.1 mm clearances, the first tongues
                                         # (2026-11-06) wouldn't go in. Once pinned, the pins hold the tongue against the
                                         # board's back face, so the extra play doesn't let the hanger droop
BASE_R = 25.0                            # the base's corner radius (the duck steps within ~20 mm of its front corners while
                                         # it turns in, so the corners stay nearly square: the floor it walks on is the same)


def ears(p):
    """The simulator's hanger beyond the rods version's plate (x past bar_x1): the new outline's front sides, where the
    front bars' channels are (world boxes)."""
    z0, z1 = p.bar_z; y_in, y_out, x_back, x_front = HANGER_ARMS; r = (y_out - y_in)/2
    return [((p.bar_x1 - 0.01, y_in + 3.0, z0), (x_front - r*0.3, y_out - 3.0, z1)), ((p.bar_x1 - 0.01, -y_out + 3.0, z0), (x_front - r*0.3, -y_in - 3.0, z1))]


def _xsec(pg):
    """A shapely polygon (or multipolygon) as a manifold CrossSection."""
    import manifold3d as m3
    loops = []
    for g in getattr(pg, "geoms", [pg]):
        ext = np.asarray(g.exterior.coords)[:-1]; loops.append(ext if g.exterior.is_ccw else ext[::-1])
        for h in g.interiors:
            hh = np.asarray(h.coords)[:-1]; loops.append(hh[::-1] if h.is_ccw else hh)
    return m3.CrossSection(loops)


def _rounded_slab(pg, z0, z1, r, steps=16):
    """Extrude a shapely outline from z0 to z1 with its top edges rounded (r), in thin slabs (z up)."""
    import manifold3d as m3
    m = m3.Manifold.extrude(_xsec(pg), z1 - r - z0).translate((0, 0, z0))
    for k in range(steps):
        zk = z1 - r + k*r/steps; h = (k + 0.5)*r/steps; d = r - math.sqrt(max(r**2 - h**2, 0.0))
        m = m + m3.Manifold.extrude(_xsec(pg.buffer(-d, join_style=1)), r/steps).translate((0, 0, zk))
    return m


def _soft_plate(pg, z0, z1, r_top, c_bot, keep_x=None, steps=24):
    """A plate from a shapely outline (world x-y), z0..z1, its top edges rounded (r_top) and its bottom edges chamfered
    (c_bot, 45 deg: it prints flat without support), in thin slabs. keep_x: the edge along x = keep_x (against the
    board) stays square: there the outline is carried on past it before each inset, then cut back."""
    import manifold3d as m3
    from shapely.geometry import box
    from shapely.ops import unary_union
    if keep_x is not None:
        b = pg.bounds; eps = 0.05; R = r_top + c_bot + 1.0
        touch = pg.intersection(box(keep_x, b[1] - 1, keep_x + eps, b[3] + 1))
        segs = [g.bounds for g in getattr(touch, "geoms", [touch]) if not g.is_empty and g.area > 0]
        carried = pg.union(unary_union([box(keep_x - R, y0, keep_x + eps, y1) for _, y0, _, y1 in segs]))
        clip = box(keep_x, b[1] - 10, b[2] + 10, b[3] + 10)
        def inset(d): return carried.buffer(-d, join_style=1).intersection(clip) if d > 0 else pg
    else:
        def inset(d): return pg.buffer(-d, join_style=1) if d > 0 else pg
    m = m3.Manifold.extrude(_xsec(pg), (z1 - z0) - r_top - c_bot).translate((0, 0, z0 + c_bot))
    n = max(4, int(round(steps*c_bot/r_top))) if c_bot > 0 else 0
    for k in range(n):                                            # the bottom chamfer
        d = c_bot - (k + 0.5)*c_bot/n
        m = m + m3.Manifold.extrude(_xsec(inset(d)), c_bot/n + 0.001).translate((0, 0, z0 + k*c_bot/n))
    for k in range(steps):                                        # the top round
        h = (k + 0.5)*r_top/steps; d = r_top - math.sqrt(max(r_top**2 - h**2, 0.0))
        m = m + m3.Manifold.extrude(_xsec(inset(d)), r_top/steps + 0.001).translate((0, 0, z1 - r_top + k*r_top/steps))
    return drop_slivers(m)


def hanger_outline(p):
    """The hanger plate's outline (shapely, world x-y), a horseshoe: two pill-shaped arms carrying the straps (|y|
    HANGER_ARMS[0]..[1], their ends rounded to half their width, reaching x HANGER_ARMS[3]), joined along the board
    behind a semicircular arch (radius = the arms' inner |y|, its back HANGER_ARMS[2] from the board) where the dozing
    duck's head drops past the hanger; the tooth's slot (open at the board) in the band behind the arch."""
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    y_in, y_out, x_back, x_front = HANGER_ARMS; rf, rb, rt = HANGER_R
    r_arm = (y_out - y_in)/2; yc = (y_out + y_in)/2
    arms = [box(0.0, sy*yc - r_arm, x_front - r_arm, sy*yc + r_arm).union(Point(x_front - r_arm, sy*yc).buffer(r_arm, 96)) for sy in (-1, 1)]
    whole = unary_union(arms + [box(0.0, -y_out, x_back + y_in, y_out)])
    arch = Point(x_back + y_in, 0.0).buffer(y_in, 192).union(box(x_back + y_in, -y_in, 300.0, y_in))
    poly = whole.difference(arch).buffer(-rf, join_style=1).buffer(rf, join_style=1)      # soften the arch's ends
    poly = poly.union(box(0.0, -y_out, 3*rb + 6, y_out).buffer(-rb, join_style=1).buffer(rb, join_style=1).intersection(whole))
    tooth = box(-5.0, -p.slot_hw, p.slot_x1, p.slot_hw).buffer(-rt, join_style=1).buffer(rt, join_style=1)
    return poly.difference(tooth)


def base_outline(p):
    from shapely.geometry import box
    x0, x1 = p.plate_x; w = p.plate_half_w
    return box(x0, -w, x1, w).buffer(-BASE_R, join_style=1).buffer(BASE_R, join_style=1)


def base_cad(p):
    """The MicroDock's base with rounded corners (BASE_R): the plate with its front ramp, the board's socket and its
    gussets as cad.base."""
    import cad, manifold3d as m3
    x0, x1 = p.plate_x; w = p.plate_half_w; T = p.T; L = p.ramp_len
    plate = m3.Manifold.extrude(_xsec(base_outline(p)), T)
    zr = T - (L + 1.0)/L*(T - p.ramp_edge)                                  # the ramp's plane 1 mm past the front edge
    prof = cad.hull(np.array([[x, y, z] for y in (-w - 1, w + 1) for x, z in ((x0 - 1, 0), (x0 - 1, T), (x1 - L, T), (x1 + 1, zr), (x1 + 1, 0))]))
    s = plate ^ prof
    bt = p.board_t + 0.3; wb = p.board_half_w + 0.3
    s = s + cad.cube(0.15, 4.15, -wb - 4, wb + 4, T - 0.4, T + p.socket_h)
    s = s + cad.cube(-bt + 0.15 - 4, -bt + 0.15, -wb - 4, wb + 4, T - 0.4, T + p.socket_h)
    s = s + cad.cube(-bt + 0.15 - 4, 4.15, wb, wb + 4, T - 0.4, T + p.socket_h) + cad.cube(-bt + 0.15 - 4, 4.15, -wb - 4, -wb, T - 0.4, T + p.socket_h)
    for yy in (-40.0, 40.0):
        s = s + cad.hull(np.array([[-bt + 0.15 - 4 + 0.5, yy + d, T + p.socket_h - 0.5] for d in (-1.5, 1.5)] +
                                  [[x0 + 4, yy + d, T - 0.4] for d in (-1.5, 1.5)] + [[-bt + 0.15 - 4 + 0.5, yy + d, T - 0.4] for d in (-1.5, 1.5)]))
    return s


def static_pieces(p):
    """The simulator's base and board for this dock (world mm, convex pieces): the base with BASE_R corners (its ramp
    kept), and the egg board (boxes inside the egg)."""
    import dock as dk
    pc = dict(dk.static_pieces(p)); pc.pop("plate"); pc.pop("ramp"); pc.pop("board")
    x0, x1 = p.plate_x; w = p.plate_half_w; T = p.T; L = p.ramp_len; R = BASE_R
    def top(x): return T if x <= x1 - L else T - (x - (x1 - L))/L*(T - p.ramp_edge)
    def lift(pts2): return np.array([(x, y, 0.0) for x, y in pts2] + [(x, y, top(x)) for x, y in pts2], float)
    def rect(xa, xb, ya, yb): return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]
    def quarter(cx, cy, sx, sy, n=10):
        return [(cx, cy)] + [(cx + sx*R*math.cos(a), cy + sy*R*math.sin(a)) for a in np.linspace(0, math.pi/2, n)]
    pc["plate_mid"] = lift(rect(x0 + R, x1 - L, -w, w)); pc["plate_back"] = lift(rect(x0, x0 + R, -w + R, w - R))
    pc["ramp_mid"] = lift(rect(x1 - L, x1 - R, -w, w)); pc["ramp_front"] = lift(rect(x1 - R, x1, -w + R, w - R))
    for sy in (-1, 1):
        pc[f"plate_corner{sy}"] = lift(quarter(x0 + R, sy*(w - R), -1, sy))
        pc[f"ramp_corner{sy}"] = lift(quarter(x1 - R, sy*(w - R), 1, sy))
    bx0 = -p.board_t
    for k, (za, zb) in enumerate(((T - 0.5, p.T + p.socket_h), (p.T + p.socket_h, 40.0), (40.0, 70.0), (70.0, 150.0), (150.0, 200.0), (200.0, 245.0))):
        hw = p.board_half_w if k == 0 else min(egg_hw(z) for z in np.linspace(za, zb, 21))
        pc[f"board{k}"] = dk.box(bx0, 0.0, -hw, hw, za, zb)
    return pc


def round_top(m, ztop, r, steps=10):
    """Round the outer edges of a part's top face (at ztop) by r: the band under the top face rebuilt in thin slabs from
    its outline, inset along a quarter circle. Whatever rises above the top face (a tooth, a boss) keeps its full
    footprint, so it stays attached and the face isn't grooved around it."""
    import manifold3d as m3
    lo, hi = m.slice(ztop - 0.02), m.slice(ztop + 0.02)
    above = hi.offset(0.02, m3.JoinType.Miter)
    out = m - m3.Manifold.extrude(lo.offset(0.05, m3.JoinType.Round), r).translate((0, 0, ztop - r))
    for k in range(steps):
        h = (k + 0.5)*r/steps; d = r - math.sqrt(max(r**2 - h**2, 0.0))
        out = out + (m ^ m3.Manifold.extrude(lo.offset(-d, m3.JoinType.Round) + above, r/steps + 0.001).translate((0, 0, ztop - r + k*r/steps)))
    return drop_slivers(out)


def drop_slivers(m, vmin=0.01):
    """The part without the zero-volume slivers that stacking thin slabs can leave (mm3)."""
    import manifold3d as m3
    parts = [q for q in m.decompose() if q.volume() > vmin]
    m = m3.Manifold.compose(parts) if len(parts) > 1 else parts[0]
    return m.simplify(1e-4)                        # and the zero-volume fins where slabs meet


def slot_half_t(p, z):
    """A strap's slot in the hanger plate: its half width through the strap's thickness (world y) at height z. The top
    piece hangs from its bar's axis, above the plate, and rocks with the bar about x by up to slot_flare: the slot widens
    linearly from the plate's top down to its underside."""
    s = p.strap(); return s.T/2 + p.slot_y + max(0.0, p.pin_top_z() - z)*math.tan(math.radians(p.slot_flare))


def hanger_cad(p):
    """World frame: the hanger plate from hanger_outline (its top edges rounded), the tooth's slot with the chamfer under
    its front edge, the tongues through the board with their pin holes, and for each strap a slot (flared downwards, so
    the strap swings sideways at its bar) and a channel on top that the bar drops into, flush with the top. Prints flat,
    as it sits."""
    import cad, manifold3d as m3
    z0, z1 = p.bar_z; x1, c = p.slot_x1, p.chamfer; s = p.strap()
    m = _soft_plate(hanger_outline(p), z0, z1, *HANGER_EDGE, keep_x=0.0)
    m = m - cad.hull(np.array([[x, y, z] for y in (-p.slot_hw, p.slot_hw) for x, z in ((x1 - 0.01, z0 - 0.01), (x1 - 0.01, z0 + c), (x1 + c, z0 - 0.01))]))
    ds, dt, ch = TONGUE_FIT
    for sy in (-1, 1):                                       # the tongues: inside the board's slots by TONGUE_FIT, the tip
        y0, y1 = sorted((sy*swing.TONGUE_Y[0], sy*swing.TONGUE_Y[1])); y0, y1 = y0 + ds, y1 - ds     # chamfered (a lead-in)
        xt, zt = -p.board_t - swing.TONGUE_L, z1 - dt
        m = m + cad.hull(np.array([[x, y, z] for x in (xt + ch, 0.01) for y in (y0, y1) for z in (z0, zt)] +
                                  [[xt, y, z] for y in (y0 + ch, y1 - ch) for z in (z0, zt - ch)]))
        m = m - m3.Manifold.cylinder(z1 - z0 + 2, cad.PIN_HOLE/2, cad.PIN_HOLE/2, 24).translate((swing.pin_x(p), sy*sum(swing.TONGUE_Y)/2, z0 - 1))
    hw = s.width()/2 + p.slot_x
    for name, ph, pw in p.strap_sites():                                             # the slots, flared downwards
        m = m - cad.hull(np.array([[pw[0] + dx, pw[1] + sy*slot_half_t(p, z), z] for dx in (-hw, hw) for sy in (-1, 1) for z in (z0 - 1, z1 + 1)]))
        L = s.bar_len + 1.0; rg = s.T/2 + 0.15                                        # the bar's channel: round below, open above
        m = m - m3.Manifold.cylinder(L, rg, rg, 64).rotate((0, 90, 0)).translate((pw[0] - L/2, pw[1], pw[2]))
        m = m - cad.cube(pw[0] - L/2, pw[0] + L/2, pw[1] - rg, pw[1] + rg, pw[2], z1 + 1)
    return drop_slivers(m)


def hood_cad(p):
    """The rods version's hood without its rod tabs: for each strap a tab out from the side wall carrying the knuckle's
    socket: a bore along x, open at the rear end and closed at the front, under a solid top slotted for the neck (a fan
    from a throat narrower than the knuckle, so the knuckle turns +-30 deg with the strap swinging sideways but can't
    pull through), and at the entrance a sprung tongue in the floor with a ridge that the knuckle rides over and clicks
    behind. The tabs stay clear of the left wall's cable slot."""
    import cad, manifold3d as m3
    top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t; s = p.strap()
    m = cad.hood(p) - cad.cube(p.hood_rear - 40, p.hood_rear, -80, 80, -20, 90)
    # the funnel is rebuilt (funnel_cad): off come the flat flares with the walls' front ends, and the bent lips with the
    # skids' front ends (the walls keep their full height under the lips)
    G = _funnel_geom(p); xa, _ = _lip_profile(p); yo_ = p.side_in + p.wall_t
    for sy in (-1, 1):
        y0, y1 = sorted((sy*(p.side_in - 0.01), sy*80.0))
        m = m - cad.cube(G["A"][0] + 0.5, 60.0, y0, y1, p.side_bot - 2.0, top1 + 12.0)
        y0, y1 = sorted((sy*(p.skid_y - 0.01), sy*p.side_in))
        m = m - cad.cube(xa, p.neck_x + 2.0, y0, y1, p.trunk_top - 0.01, top1 + 8.0)
        y0, y1 = sorted((sy*p.side_in, sy*(yo_ + 0.01)))
        m = m - cad.cube(xa, p.neck_x + 2.0, y0, y1, top1 - 0.01, top1 + 8.0)
    m = m - cad.cube(p.hood_rear - 1, p.hood_rear + 3.01, -yo - 1, yo + 1, top1, 90)
    m = m + cad.cube(p.hood_rear, p.hood_rear + 3.0, -p.tooth_hw, p.tooth_hw, top1 - 0.5, p.tooth_top)
    xc = swing.cable_exit(p)[0]; slot = (xc - (cad.CABLE_SLOT/2 + 1.7), xc + (cad.CABLE_SLOT/2 + 1.7))   # open at the wall's bottom
    rb = s.T/2 + 0.15; L = s.width() + 0.8; wl = p.socket_wall; end = 1.6; roof = 1.4
    w0, fan = p.throat, math.tan(math.radians(p.knuckle_turn)); c0 = math.sqrt(rb**2 - w0**2)
    assert w0 < s.T/2 - 0.3                                  # the knuckle bears on the throat's edges
    assert c0*math.tan(math.radians(p.knuckle_turn)) + s.neck_t/2/math.cos(math.radians(p.knuckle_turn)) <= w0   # the neck clears it
    for name, ph, pw in p.strap_sites():
        sy = 1.0 if ph[1] > 0 else -1.0; x, ys, z = ph[0], sy*p.strap_y, ph[2]
        x0, x1 = x - s.width()/2 - 2.4, x + L/2              # the bore: open at x0 (rear), closed past x1 (front); the knuckle
                                                             # sits 2.4 mm in from the entrance, the ridge between
        zb = z - rb - 1.2                                    # the socket's underside
        body = _sphere_hull([[xx, yy, zz] for xx in (x0, x1 + end) for yy in (ys - rb - wl, ys + rb + wl) for zz in (zb, z + rb + roof)], 1.8)
        a0, a1 = tab_x(x, sy, xc, slot)                     # the tab's span along x, from the wall out to the socket
        yin = ys - sy*(rb + wl - 0.5)
        near = [[xx, sy*yy, zz] for xx in (a0, a1) for yy in (yo - 0.5, yo + 3.0) for zz in (zb, zb + 3.0)]
        far = [[xx, yy, zz] for xx in (a0, a1) for yy in (yin - sy*3.0, yin) for zz in (zb, zb + 3.0)]
        m = m + _sphere_hull(near + far, 1.2) + body                 # the arm and the socket, rounded
        m = m + _sphere_hull([[xx, sy*yy, zz] for xx in (a0, a1) for yy, zz in ((yo - 1.5, zb), (yo - 1.5, zb + 9.0), (yo + 4.0, zb))], 1.0)
        m = m - m3.Manifold.cylinder(x1 - x0 + 1.0, rb, rb, 64).rotate((0, 90, 0)).translate((x0 - 1.0, ys, z))  # the bore
        t = z + rb + roof + 0.01; wt = w0 + (t - z - c0)*fan                                     # the neck's slot: a fan from
        m = m - cad.hull(np.array([[xx, ys + dy, zz] for xx in (x0 - 1.0, x1) for dy, zz in             # the throat at the bore, so
                                   ((-w0, z), (w0, z), (-w0, z + c0), (w0, z + c0), (-wt, t), (wt, t))]))  # the knuckle turns +-30 deg
        tl = 6.0                                             # the sprung tongue: the floor's rear 6 mm, cut free on 3 sides
        for dy in (-2.0, 2.0):
            m = m - cad.cube(x0 - 0.1, x0 + tl, ys + dy - 0.3, ys + dy + 0.3, zb - 1, z - rb + 0.5)
        m = m - cad.cube(x0 - 0.1, x0 + tl, ys - 2.3, ys + 2.3, zb - 1, zb + 0.4)          # thinned underneath, so it bends
        ridge = m3.Manifold.cylinder(4.0, 0.5, 0.5, 24).rotate((90, 0, 0)).translate((x - s.width()/2 - 0.9, ys + 2.0, z - rb))
        m = m + ridge
    return drop_slivers(round_top(m, top1, HOOD_EDGE_R) + funnel_cad(p))


def board_cad(p):
    """The v3 board (swing.board_cad) with the MicroDock v2's cable clip back in its left edge: a keyhole the cable presses
    into and runs through to the back, 12 mm below the hanging hood's lowest point (cad.cable_clip). It keeps the cable's
    weight, and any tug on it, off the hood; the loop between the hood and the clip lets the hood float."""
    import cad
    s = swing.board_cad(p); w = p.board_half_w
    (yk, zk), mouth = cad.cable_clip(p), cad.CABLE_D - 0.5
    assert zk < p.hood_home()[2] - 5.0                          # below the hood (its lowest point is the hood frame's z 0)
    s = s - cad.cyl_x(cad.CABLE_D/2 + 0.05, -p.board_t - 1, 1, yk, zk)
    s = s - cad.cube(-p.board_t - 1, 1, yk, w + 1, zk - mouth/2, zk + mouth/2)
    s = s - cad.hull(np.array([[xx, w - 1.0, zk + dz*mouth/2] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)] +
                              [[xx, w + 0.5, zk + dz*(mouth/2 + 1.5)] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)]))
    return s


EGG = dict(L=254.0, z0=-1.0, B=0.7088, w=0.067, D=0.5354)   # the egg board: Narushin et al. (2021)'s universal egg
# formula fitted to a reference egg outline (2026-11-06; 0.4 mm rms at this size): breadth, the widest point's offset
# from the middle and the diameter a quarter from the tip, over the length. 254 mm tall from 1 mm below the floor (the
# P1S bed's limit), 181 mm wide 100 mm up (around the hood), 110 mm where it enters the base's socket.
EGG_TAG_CZ = 205.0                  # the tag 6 mm lower than on the rectangular board: its top corners stay 7 mm inside


def egg_hw(z, egg=EGG):
    """The egg's half width at world height z (0 outside it)."""
    L, z0, B, w, D = egg["L"], egg["z0"], egg["B"], egg["w"], egg["D"]; t = (z - z0)/L
    if t <= 0.0 or t >= 1.0: return 0.0
    x = t - 0.5                                                 # along the axis, from the middle toward the tip (L = 1)
    base = B/2*math.sqrt(max((1 - 4*x**2)/(1 + 8*w*x + 4*w**2), 0.0))
    A = math.sqrt(5.5 + 11*w + 4*w**2); C = math.sqrt(1 + 2*w + 4*w**2)
    k = (A*(math.sqrt(3)*B - 2*D*C))/(math.sqrt(3)*B*(A - 2*C))
    inner = math.sqrt(max((1 + 8*w*x + 4*w**2)/(2*(1 - 2*w)*x**2 + (1 + 8*w - 4*w**2)*x + 2*w**2 + w + 1), 0.0))
    return L*base*(1 - k*(1 - inner))


def egg_outline(p, n=480):
    """The egg board's outline (shapely polygon, world y-z): the egg above the floor plus the board's 110 mm foot in the
    base's socket (the egg is 110 mm wide just where it leaves the socket, so it seems to stand in it)."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    L, z0 = EGG["L"], EGG["z0"]; zs = z0 + L*(1 - np.cos(np.linspace(0.0, np.pi, n)))/2   # denser at the round ends
    egg = Polygon([(egg_hw(z), z) for z in zs] + [(-egg_hw(z), z) for z in zs[::-1]]).buffer(0)
    foot = box(-p.board_half_w, p.T, p.board_half_w, p.T + p.socket_h)
    return unary_union([egg.intersection(box(-1e3, p.T, 1e3, 1e3)), foot])


def egg_clip(p):
    """The cable clip on the egg board: at v2's height, 4.5 mm in from the egg's left edge (world y, z)."""
    import cad
    zk = cad.cable_clip(p)[1]; return egg_hw(zk) - 4.5, zk


def egg_board_cad(p, r=2.0, steps=16):
    """The board as an egg (the duck docks at its egg): EGG's outline, 6 mm thick, its front edge rounded (r), with the
    tag pocket (at EGG_TAG_CZ), the hanger's two tongue slots and the cable clip at the egg's left edge. Prints flat on
    its back, on a plate of its own (250 x 181 mm)."""
    import cad, manifold3d as m3
    poly = egg_outline(p)
    M = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]              # local (y, z, x) -> world (x, y, z)
    def slab(pg, x0, x1):
        cs = m3.CrossSection([np.asarray(pg.exterior.coords)[:-1][::(1 if pg.exterior.is_ccw else -1)]])
        return m3.Manifold.extrude(cs, x1 - x0).translate((0, 0, x0)).transform(M)
    s = slab(poly, -p.board_t, -r)
    for k in range(steps):                                       # the rounded front edge, in thin slabs
        x0 = -r + k*r/steps; xm = (k + 0.5)*r/steps; d = r - math.sqrt(max(r**2 - xm**2, 0.0))
        s = s + slab(poly.buffer(-d, join_style=1), x0, x0 + r/steps)
    s = drop_slivers(s)
    tw = p.tag_size/2 + 2.0 + 0.15                               # the tag pocket, as on the rectangular board
    s = s - cad.cube(-1.6, 1, -tw, tw, EGG_TAG_CZ - tw, EGG_TAG_CZ + tw)
    z0, z1 = p.bar_z
    for sy in (-1, 1):                                           # the hanger's tongues
        y0, y1 = sorted((sy*(swing.TONGUE_Y[0] - 0.15), sy*(swing.TONGUE_Y[1] + 0.15)))
        s = s - cad.cube(-p.board_t - 1, 1, y0, y1, z0 - 0.1, z1 + 0.1)
    (yk, zk), mouth = egg_clip(p), cad.CABLE_D - 0.5; w = egg_hw(zk)   # the cable clip: a keyhole in the egg's left edge
    s = s - cad.cyl_x(cad.CABLE_D/2 + 0.05, -p.board_t - 1, 1, yk, zk)
    s = s - cad.cube(-p.board_t - 1, 1, yk, w + 3, zk - mouth/2, zk + mouth/2)
    s = s - cad.hull(np.array([[xx, w - 1.0, zk + dz*mouth/2] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)] +
                              [[xx, w + 1.5, zk + dz*(mouth/2 + 1.5)] for xx in (-p.board_t - 1, 1) for dz in (-1, 1)]))
    return s


EGG_DOME = dict(depth=22.0, z_cut=19.5)      # the egg half: its dome behind the board (deepest behind the egg's widest
                                             # point; it reaches 28 mm behind the face, the base already 30) and where it
                                             # stops at the bottom (just above the base's socket)
EGG_RIM = dict(w=3.0, h=2.5, z0=20.5)        # the cut edge's rim in front: width, height, from this height up


def egg_board(p):
    """The dock's board: the flat egg, or the egg half when p.egg_half."""
    return egg_half_cad(p) if getattr(p, "egg_half", False) else egg_board_cad(p)


def egg_half_cad(p, r_rim=1.4, steps=10):
    """The egg board as an egg half: the flat cut face in front (with the tag pocket and the hanger's slots; its edge
    rounded), and behind it a dome: the egg as a solid of revolution about its axis, cut through the axis, its depth
    scaled to EGG_DOME's. Behind the hanger's tongues the dome has pockets, open at the back, for the tongues and their
    pins; the cable clip goes through its edge. (A rim along the cut face's edge was tried and dropped, 2026-11-06.)"""
    import cad, manifold3d as m3
    from shapely.geometry import Polygon, box as sbox
    L, z0e = EGG["L"], EGG["z0"]; zs = z0e + L*(1 - np.cos(np.linspace(0.0, np.pi, 480)))/2
    egg = Polygon([(egg_hw(z), z) for z in zs] + [(-egg_hw(z), z) for z in zs[::-1]]).buffer(0)
    M = [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]                       # local (y, z, x) -> world (x, y, z)
    def slab(pg, x0, x1): return m3.Manifold.extrude(_xsec(pg), x1 - x0).translate((0, 0, x0)).transform(M)
    outline = egg_outline(p); r = EDGE_R; H = 0.0
    s = slab(outline, -p.board_t, -r)                                   # the cut face's plate (and the foot in the socket)
    for k in range(16):                                                 # its front edge rounded (as the flat egg board)
        x0 = -r + k*r/16; xm = (k + 0.5)*r/16; d = r - math.sqrt(max(r**2 - xm**2, 0.0))
        s = s + slab(outline.buffer(-d, join_style=1), x0, x0 + r/16 + 0.001)
    k_d = EGG_DOME["depth"]/max(egg_hw(z) for z in np.linspace(z0e, z0e + L, 400))
    dome_pg = egg.intersection(sbox(-200, EGG_DOME["z_cut"], 200, 400))
    dome = slab(dome_pg, -p.board_t - 1.0, -p.board_t + 0.01).refine_to_length(2.0)
    def warp(V):
        V = np.array(V, dtype=np.float64); back = V[:, 0] < -p.board_t - 0.5
        hw = np.array([egg_hw(z) for z in V[back, 2]]); d = k_d*np.sqrt(np.clip(hw**2 - V[back, 1]**2, 0.0, None))
        V[back, 0] = -p.board_t - np.maximum(0.6, d); return V
    dome = dome.warp_batch(warp)
    s = s + dome
    tw = p.tag_size/2 + 2.0 + 0.15                                       # the tag's pocket
    s = s - cad.cube(-1.6, 1, -tw, tw, EGG_TAG_CZ - tw, EGG_TAG_CZ + tw)
    z0, z1 = p.bar_z; back = -p.board_t - EGG_DOME["depth"] - 5.0
    for sy in (-1, 1):                                                  # the hanger's tongues: slots, and pockets behind
        y0, y1 = sorted((sy*(swing.TONGUE_Y[0] - 0.15), sy*(swing.TONGUE_Y[1] + 0.15)))
        s = s - cad.cube(-p.board_t - 1, 1, y0, y1, z0 - 0.1, z1 + 0.1)
        y0, y1 = sorted((sy*(swing.TONGUE_Y[0] - 1.5), sy*(swing.TONGUE_Y[1] + 1.5)))
        s = s - cad.cube(back, -p.board_t + 0.01, y0, y1, z0 - 1.5, z1 + 1.5)        # open at the back: drop the pins in
    (yk, zk), mouth = egg_clip(p), cad.CABLE_D - 0.5; w = egg_hw(zk)                # the cable clip, through the dome's edge
    s = s - cad.cyl_x(cad.CABLE_D/2 + 0.05, back, H + 1, yk, zk)
    s = s - cad.cube(back, H + 1, yk, w + 3, zk - mouth/2, zk + mouth/2)
    s = s - cad.hull(np.array([[xx, w - 1.0, zk + dz*mouth/2] for xx in (back, H + 1) for dz in (-1, 1)] +
                              [[xx, w + 1.5, zk + dz*(mouth/2 + 1.5)] for xx in (back, H + 1) for dz in (-1, 1)]))
    return drop_slivers(s)


def strap_world(p, site, opened):
    """One strap's rows placed in the world (meshes), hanging from its top pin; opened: how far each joint is open (mm)."""
    import trimesh
    s = p.strap(); name, ph, pw = site; u_top = s.hole_u()[1]; out = []
    for r in range(s.R + 1):
        m = fs.row_cad(s, r).to_mesh(); V = np.asarray(m.vert_properties)[:, :3].copy()
        du = -(s.R - r)*opened; dw = s.knuckle_flat if r == 0 else 0.0   # the knuckle on the socket's axis (the joints
        W = np.c_[pw[0] + V[:, 1], pw[1] + V[:, 2] - s.T/2 + dw, pw[2] + (V[:, 0] - u_top) + du]   # take it: cz 0.4)
        out.append(trimesh.Trimesh(W, np.asarray(m.tri_verts), process=False))
    return out

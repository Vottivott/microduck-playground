"""MicroDock v3, the egg dock (in development "v4, egg C"): the egg as a true egg half -- a flat front plate (the hood's face, the tag, the window) and a
domed back (the egg of revolution cut through its axis, its depth scaled) -- with the spring stage re-laid out to fit
inside the dome.

v4's stage hung a wide frame from an anchor bar at the top and carried the lift on that frame, to one side of the hood:
swinging, the frame came within 6 mm of the egg's edge, so the egg had to be 25 mm deep right out to its edge. Here the
order is the other way round, with the same blades (so the same springs, the ones the simulator docked with):

  front plate -> lift post (fixed, pegged to the front plate) -> two lift blades (42 mm, in the middle, between the
  hanging blades) -> tab -> hanger bar (up and down only) -> four hanging blades (64 mm, pulled, never pushed) ->
  the carriage: a crossbar and, below it, the socket for the hood's arm (sideways and up).

Everything that swings stays within +-62 mm of the middle at the hood's height and +-46 mm at the top, where the
egg half is deep enough (fit()). Fully printed, as the MicroDock: the dome snaps onto the front plate (five fingers
inside it), the hood's arm snaps into the carriage, the post sits on pegs under the dome's clamp, the tag presses in.
"""
import dataclasses, math
import numpy as np

import strapswing as ss
import flexdock as fd
from flexdock import STAGE_X, ARM, ARM_FIT, CARRIAGE, FRAME, Y_BLADES, Z_BLADES, _box, _blade, _slab

DOME = dict(n=2.0, D=33.0, rim=6.0, shell=2.0)   # profile (2: egg of revolution), depth at the middle, at the rim, wall
HW_MAX = max(ss.egg_hw(z) for z in np.linspace(0.0, 253.0, 4000))
FW = fd.EGG_T[0]                                 # the front plate (its face at x = 0)
HANG_Y = (37.0, 43.0)                            # the hanging blades: +- these (inner pair clear of the lift post)
LIFT_Z = (127.0, 146.0)                          # the lift blades' roots on the post (straight as printed): 19 mm apart
# (v4 and the first C: 12 mm) -- the hood's elastic pitch, 2.5x stiffer (rot_springs)
LIFT_SPAN = (-21.0, 21.0)                        # the lift blades: from the tab to the post (42 mm)
CLEAR = 1.0                                      # stage to the dome's inside
POST_STEP_Z, POST_LOW_W = 131.0, 4.0             # below this the post is 4 mm wide (the inner hanging blades swing close)


def depth(y, z, n=DOME["n"], D=DOME["D"], e=DOME["rim"]):
    """The dome's outer surface behind the face (mm, >= 0), at world (y, z)."""
    y, z = np.atleast_1d(np.asarray(y, float)), np.atleast_1d(np.asarray(z, float))
    hw = np.array([ss.egg_hw(zz) for zz in z])
    rho = np.clip(np.abs(y)/np.maximum(hw, 1e-9), 0.0, 1.0)
    g = np.where(hw > 1e-6, hw/HW_MAX*(1 - rho**n)**(1/n), 0.0)
    return e + (D - e)*g


def layout_c(p):
    """The stage's parts at rest (the hood's weight on the post): world y-z boxes."""
    zc = p.hood_home()[2] + ARM[2]; cw, ch = CARRIAGE; zl = zc - ch/2
    xb = HANG_Y[1] + 2.5                                         # the crossbar and the hanger bar: +- xb
    cross = (zl + ch, zl + ch + FRAME)
    hanger = (cross[1] + Y_BLADES["L"], cross[1] + Y_BLADES["L"] + FRAME)
    zn = p.z_neutral
    tab = ((LIFT_SPAN[0] - FRAME, LIFT_SPAN[0]), (LIFT_Z[0] - zn - 2.0, hanger[0]))
    post = ((LIFT_SPAN[1], LIFT_SPAN[1] + FRAME), (POST_STEP_Z, hanger[0]))      # its top: the down-stop
    post_low = ((LIFT_SPAN[1], LIFT_SPAN[1] + POST_LOW_W), (LIFT_Z[0] - 2.0, POST_STEP_Z))   # narrow: the blades swing by
    up_stop = ((-10.0, 10.0), (hanger[1] + p.z_travel + 1.0, hanger[1] + p.z_travel + 5.0))   # a backup only
    cross_stop = ((-12.0, 12.0), (cross[1] + p.z_travel, cross[1] + p.z_travel + 4.0))           # the hood's up-stop
    side = xb + p.y_travel
    stops = dict(left=(-side - 4.0, -side), right=(side, side + 4.0), z=(cross[0] - 2.0, cross[1] + p.z_travel + 2.0))
    return dict(zc=zc, carriage=((-cw/2, cw/2), (zl, zl + ch)), cross=((-xb, xb), cross), hanger=((-xb, xb), hanger),
                hang_y=(-HANG_Y[1], -HANG_Y[0], HANG_Y[0], HANG_Y[1]), tab=tab, post=post, post_low=post_low, up_stop=up_stop, stops=stops,
                cross_stop=cross_stop,
                lift_z=LIFT_Z, window=fd.layout(p)["window"])


def rot_springs(lift_dz=None, E=3500.0):
    """The stage's rotational stiffness (N m/rad: roll, pitch, yaw), the hanging and the lift blades in series; lift_dz:
    the lift blades' spacing (mm, default LIFT_Z's). The simulator turns these into soft equalities (rot_solref)."""
    t, b = Y_BLADES["t"], STAGE_X[1] - STAGE_X[0]; I, A = t*b**3/12, t*b; Ly, Lz = Y_BLADES["L"], Z_BLADES["L"]
    d = (LIFT_Z[1] - LIFT_Z[0]) if lift_dz is None else lift_dz; S2 = 2*sum(y*y for y in HANG_Y)
    ser = lambda *k: 1/sum(1/x for x in k)
    roll = ser(E*A/Ly*S2, 2*E*A/Lz*(d/2)**2); pitch = ser(4*4*E*I/Ly, 2*12*E*I/Lz**3*(d/2)**2); yaw = ser(12*E*I/Ly**3*S2, 2*4*E*I/Lz)
    return tuple(round(k/1e3, 1) for k in (roll, pitch, yaw))


def springs_c(p):
    """Same blades as v4 (4 hanging 0.4 x 16 x 64, 2 lift 0.4 x 16 x 42): the same springs (flexdock.springs)."""
    return fd.springs(p)


def _sweep_points(p, n_dy=9, n_dz=6):
    """Points (y, z) of every part's outline over the hood's travel (sideways +-y_travel, up 0..z_travel): the
    carriage, crossbar and hanging blades move with the hood; the hanger and the tab with its lift; the lift blades
    bend between the post and the tab."""
    Lo = layout_c(p); P = []
    def box_pts(y0, y1, z0, z1, k=8): return [(y, z) for y in np.linspace(y0, y1, k) for z in np.linspace(z0, z1, k)]
    L = Y_BLADES["L"]; (c0, c1) = Lo["cross"][1]
    for dz in np.linspace(0, p.z_travel, n_dz):
        for dy in np.linspace(-p.y_travel, p.y_travel, n_dy):
            V = 0.6*dy**2/L                                           # the carriage rises as its blades swing
            for nm in ("carriage", "cross"):
                (y0, y1), (z0, z1) = Lo[nm]; P += [(y + dy, z + dz + V) for y, z in box_pts(y0, y1, z0, z1)]
            for yb in Lo["hang_y"]:                                   # fixed at the hanger, guided at the crossbar
                for s in np.linspace(0, 1, 17):
                    z = Lo["hanger"][1][0] + dz - s*L; P.append((yb + dy*(3*s**2 - 2*s**3), z))
        for nm in ("hanger", "tab"):
            (y0, y1), (z0, z1) = Lo[nm]; P += [(y, z + dz) for y, z in box_pts(y0, y1, z0, z1)]
        for zr in LIFT_Z:                                             # the lift blades: bent (dz - zn) at the tab
            w = dz - p.z_neutral
            for s in np.linspace(0, 1, 17):
                P.append((LIFT_SPAN[1] - s*(LIFT_SPAN[1] - LIFT_SPAN[0]), zr + w*(3*s**2 - 2*s**3)))
    for nm in ("post", "post_low", "up_stop"):
        (y0, y1), (z0, z1) = Lo[nm]; P += box_pts(y0, y1, z0, z1)
    return np.array(P)


def fit(p):
    """Margins (mm): the stage's back (STAGE_X[0]) to the dome's inside (with CLEAR), everywhere it goes; and the
    hanging blades to the lift post and tab (the closest any blade gets, over the travel)."""
    P = _sweep_points(p)
    inside = depth(P[:, 0], P[:, 1]) - DOME["shell"]
    m = inside - (-STAGE_X[0]) - CLEAR; i = int(np.argmin(m))
    Lo = layout_c(p); L = Y_BLADES["L"]; gap = 1e9
    for dz in np.linspace(0, p.z_travel, 16):
        for dy in np.linspace(-p.y_travel, p.y_travel, 33):
            for yb in Lo["hang_y"]:
                for s in np.linspace(0, 1, 65):
                    z = Lo["hanger"][1][0] + dz - s*L; y = yb + dy*(3*s**2 - 2*s**3)
                    for nm, mv in (("post", 0.0), ("post_low", 0.0), ("tab", dz)):
                        (y0, y1), (z0, z1) = Lo[nm]
                        if z0 + mv <= z <= z1 + mv:
                            gap = min(gap, max(y0 - y, y - y1) - Y_BLADES["t"]/2)
    return dict(dome_margin=round(float(m.min()), 2), at=tuple(np.round(P[i], 1)), blade_to_post_tab=round(gap, 2))


# ------------------------------------------------------------------------------------------------ CAD (world mm)
FOOT_Z = fd.FOOT_Z                       # below: the front plate's 6 mm foot, in the base's socket
FLOOR = 3.0                              # the dome's floor (z FOOT_Z .. +3), on the base
PEG = dict(d=3.0, fit=0.15, h=3.5, z=(136.0, 144.0))   # the post's two locating pegs (on the front plate)
SNAP = dict(t=1.4, h=8.0, barb=0.8, lead=1.4, flat=0.4, gap=0.15, block=4.0, reach=6.5)
# the dome's snap fingers: 1.4 mm thick (they flex sideways, along their printed layers), 8 mm tall at the tip, a barb
# 0.8 mm (30 deg lead-in, 45 deg back: a firm pull takes the dome off); the plate's catch blocks reach 6.5 mm back
SNAPS = ((-40.0, 36.0, -1), (40.0, 36.0, 1), (-79.0, 120.0, -1), (79.0, 120.0, 1), (0.0, 215.0, 1))   # (y, z, barb side)
ARM_JOINT = dict(floor=STAGE_X[0] + 1.5, fit=0.15, rib_h=0.35, rib_w=0.8, lead=0.8, pin_x=-12.0, pin_hole=1.85, arm_hole=1.95,
                 draw=0.1, pin_floor=1.5)
# The hood's arm in the carriage: 14.5 mm deep (to 1.5 mm off the back), 0.15 mm clearance taken up by crush ribs (0.35 mm,
# two on each face, the whole depth: no play, however the parts print), a 0.8 mm lead-in; a 1.75 mm filament pin down
# through the crossbar, the carriage and the arm (its hole 0.1 mm toward the hood: the pin draws the arm home) carries
# the undocking pull. (The first C had a snap: its 0.6 mm barb wore out after a few dozen undockings.)
PLUG_C = dict(tip_out=3.2, tip_w=11.85, tip_h=7.6, tip_front=(11.2, 6.8, 1.65), face_offset=0.025, plug=(8.11, 12.91, 14.27),
              catch_lift=0.45, pocket_top=0.05)
# The magnetic pair as it is (the tip measured on photos against a tape, its scale checked by its 0.5 mm pin pitch; the
# first hoods used product-photo sizes: a 14 mm adapter, an 11.8 x 6.0 mm tip 3.4 mm out). The duck's tip: a black
# base 11.85 x 7.6 mm on the shell (calipers: 7.59), a silver front part 11.2 x 6.8 mm (near-stadium, corners ~3 mm)
# 1.65 mm long, its face 3.2 mm out (the head is 3.44 mm thick: its port sits a little below the shell). The hood-side
# adapter, calipers (11-11): 14.27 long, 12.91 x 8.11 (the photos had 13.4 x 12.0 x 8.0, so the earlier pockets were
# 0.3 mm a side too narrow for it). face_offset: the adapter face's nominal rest, 0.025 mm toward the duck from the
# docked tip's face; the lead-in (GUIDE) and the battery stop are placed from it. With the right-angle cable head
# (CABLE_HEAD "ugreen") a leaf spring behind the head holds the face there (SPRING, FIT); with the goobay adapter the
# stack floats instead (FLOAT). catch_lift, pocket_top: the clamp (from the goobay hood's second print: the adapter
# had 0.25 mm of play up and down in its pocket, 0.1 under the catch's hooks and 0.15 to the roof, and with its round
# corners that let it turn +-4 deg about its axis, through the play of its USB-C joint with the goobay, which the
# goobay's own click holds within 1 deg): the hooks 0.3 higher and the roof 0.05 over the adapter, so the hooks press
# its flat top against the roof (0.15 mm interference at each hook, ~0.7 N: it can still slide along x).
FIT = dict(len_mid=16.4, reach=0.05, base=0.1)
# The face's window (face_window): no further back than the tip's face at the battery stop (+ reach), no further
# forward than the lead-in's narrow stadium (- reach), which itself stays `base` short of where the tip's bigger base
# is at the battery stop. 1.35 mm, centred on an elbow + adapter of len_mid: 15.7 to 17.1 mm seat and dock (the first
# egg-C hoods' 14 + 2.7 = 16.7 and the photos' 13.4 + 2.7 = 16.1 both inside). Longer still goes in (the stadium's back
# edge is chamfered, the spring gives) and rests against it.
GUIDE = dict(w=11.2 + 0.6, h=6.8 + 0.6, gap=1.025, narrow=0.1, flare=1.35, wall=1.5)
# A lead-in in front of the adapter: the tip's front part slides into a stadium 0.3 mm bigger each side, through a
# 45 deg flare 1.4 mm deep, so the soft-sprung hood is pushed into line before the faces meet (the adapter's rim needs
# ~0.4 mm). `gap`: the stadium's back from the face at rest for 13.4 + 2.7 mm (face_offset); it ends 1.55 mm from the
# tip's face at the battery stop, so the tip's bigger base (1.65 mm back) never reaches it (face_window). Sides and top
# only: below the middle the side walls run straight down to the stadium's bottom and stop (see _guide).
ARM_TRIM = 0.1                           # the hood's arm 0.1 mm thinner each side in the carriage's socket: the crush
                                         # ribs (0.2 mm interference as printed in the stage) were a little too tight
SPRING = dict(t=0.8, w=8.0, tip_z=-2.0, preload=0.4, travel=1.4, lead=0.6)
# The leaf behind the cable head (hood frame): 0.8 x 8 mm, hanging from the top plate to 2 mm below the head's axis; it
# pushes the head's round body forward at its axis (no lift on the adapter) onto the stop rib, 0.4 mm preloaded (~0.5
# N); the duck can push the head back `travel` before it meets the hard stop behind (about 1.8 N there, 1.1 % strain).
# `lead`: the chamfer at its tip that the head's body cams it back with as the pair goes in from below.
CABLE_HEAD = "goobay"
# What plugs into the magnetic adapter behind it: "goobay", a straight USB-C cable through goobay's 90 deg adapter (the
# hood from 11-11), or "ugreen", UGREEN's right-angle cable (the hoods up to print_dome: SPRING, FIT, _rib_down, ...).
GOOBAY = dict(L=24.8, u=12.34, t=6.34, plug_at=5.6, collar=2.05, collar_l=10.6, plug_out=6.77)
# goobay's 90 deg "horizontal" USB-C adapter (USB4, 240 W; Kjell's), calipers 11-11: a 24.8 mm body (L: its socket's
# axis, which lies along y here, the plug's wide side horizontal) 12.34 deep (u: along its plug, x) and 6.34 thick (t,
# z); its plug, on the body's side 5.6 mm from the far end, sticks out 6.77 mm from a black collar 2.05 mm proud of the
# body (10.6 long); the female socket at the other end, centred. In the hood the body's far end points right (-y) and
# the socket left (+y), at the left wall.
CORD = dict(w=12.34, t=6.40, l=19.95, relief=8.09, relief_d=5.90, cable_d=3.2)
# The straight cable's plug (Clas Ohlson's 20 W kit, 39-2051, 60 W cable), calipers: moulding 19.95 long, 12.34 wide
# (it lies flat, across x) and 6.40 thick, then a strain relief 8.09 long and 5.90 across, a 3.2 mm cable. Plugged into the goobay it reaches out through the left wall (the wall's inside is
# 33 mm from the adapter's axis, the socket's end 19.2): the moulding pokes 3.75 mm out of the wall, the relief 8 more.
FLOAT = dict(fwd=0.80, back=-0.65, wall_lo=4.1, top=4.0, rib_down=12.0)
# The adapter, goobay and cable plug float as one stack along x between two stops, no spring: the tip pushes it back
# as the duck docks (nothing but the stack's light friction against it), the magnet pulls it forward again as the duck
# leaves, onto the front stop, where it then waits. (There's no room for a spring: adapter + collar + goobay = 28.66 mm
# against the 28.2 from the nominal face to the rear wall.) fwd/back: the front and rear stops for the adapter's face,
# from the docked tip's face (+: toward the duck). The front stop (a rib before the goobay's front face, left of its
# collar) holds the face 0.8 mm toward the duck, 0.25 behind the lead-in's stadium (the tip's front part is in the
# stadium at first touch); there the goobay's back is 0.31 mm off the rear wall, which is its path up from below. The
# rear stop: 0.65 behind the docked tip's face, 0.25 past where the tip goes with the battery on its stop (so the
# battery stop, not the tip, takes the docking push), in a pocket 1.14 mm into the rear wall's 3 mm, wall_lo below and
# `top` above the adapter's axis (above the arm's root). top: the stack's channel and the left wall's opening, above the
# axis. rib_down: how far below the axis the front stop reaches (its foot chamfered: a stack rising too far forward is
# cammed back, as _rib_down did).
GCATCH = dict(y=(13.0, 18.5), t=1.8, hook=2.2, gap=0.1, tip=0.4, lead=30.0, roof=0.3)
CSEAT = dict(y=(26.0, 31.5))
# The cable plug's seat: GCATCH's finger and hook again, under the plug's moulding by the left wall, so the
# plug side sits on a click of its own; the roof runs on over the moulding to the wall.
# The goobay's own click (the first goobay hood had only the adapter's, and the plug and cable, hanging 45 mm out to
# the side, rolled the stack round it: the adapter's hooks are 45 deg on top, made to let go): a finger from the top
# plate before the goobay's front face, on its cable side (y), its hook `hook` mm under the goobay's flat bottom, `gap`
# below it, flat on top so it doesn't let go (push the finger forward to take the stack out); long enough to hold over
# the whole float (2.05 mm under at the front stop, 0.6 at the rear), its foot a `lead` deg ramp the goobay pushes it
# aside with on its way up. roof: over the goobay's top from its far end to its socket (it was open up to the top
# plate): its far end can't rise as the plug pulls down.
CLAMP_GAP = 2.5                          # the dome's clamp stops this short of the post: a backup only (the first C
                                         # domes: 0.3, so they braced the post, and the docking push levered them off
                                         # their snaps); POST_PIN holds the post
POST_PIN = dict(x=-11.0, z=139.5, hole=1.85, ear_t=2.5, ear_gap=0.2, ear_z=(134.5, 144.6), ear_x=-15.0)
# The lift post pinned to the front plate: an ear each side of it (2.5 mm, 0.2 off the post, between the lift blades'
# roots at z 127 and 146) and a 1.75 mm filament pin through ear, post and ear along y (round 1.85 mm holes, see
# _pin_hole). The docking push goes into the front plate, not through the dome.
LIP = dict(gap=0.3, w=1.6, h=1.3)
# A lip round the front plate's back just inside the dome's side wall (`gap` off it, `h` tall: clear of the stage's
# front plane at -4.5): the dome's wall bears on it all round, so plate and dome act as one shell (the plate alone, 3 mm,
# bends ~0.5 mm/N at the post's height) and the dome can't slide off its snaps. Broken at the floor and the cable clip.
CABLE = dict(d=5.0, mouth=4.3)
# The cable at the egg's edge: its hole for up to 5 mm (the braided cable looks ~4.5-5 on the photos; the first C: 4.1,
# mouth 3.5), the front plate's mouth a light snap; the dome's channel no narrower than the hole (no snap there).
STOP_DEPTH = 14.0                        # the side stops reach this far back (they only need to meet the crossbar)


def _poly_above(z0):
    from shapely.geometry import box as sbox
    import strapswing as ss_
    return lambda p: ss_.egg_outline(p).intersection(sbox(-300, z0, 300, 400))


def _warped(pg, x_front, xdepth, L=0.8, ext=1.0):
    """pg (shapely, y-z) extruded back from x_front, its back face pushed to x = -xdepth(y, z) (vectorised)."""
    m = _slab(pg, x_front - ext, x_front).refine_to_length(L)
    def warp(V):
        V = np.array(V, dtype=np.float64); b = V[:, 0] < x_front - 1e-6
        t = (x_front - V[b, 0])/ext
        V[b, 0] = x_front - t*(xdepth(V[b, 1], V[b, 2]) + x_front)
        return V
    return m.warp_batch(warp)


def _inner_depth(t=DOME["shell"], h=0.5):
    """The dome's inside, a wall t thick (measured square to the surface, not along x): the outer height field eroded
    by a ball of radius t. Returns f(y, z) -> depth behind the face (an interpolator)."""
    from scipy.interpolate import RegularGridInterpolator
    ys = np.arange(-95.0, 95.0 + h/2, h); zs = np.arange(15.0, 258.0 + h/2, h)
    hw = np.array([ss.egg_hw(z) for z in zs])
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    rho = np.abs(Y)/np.maximum(hw[None, :], 1e-9); inside = rho < 1.0
    n, D, e = DOME["n"], DOME["D"], DOME["rim"]
    d = np.where(inside, e + (D - e)*(hw[None, :]/HW_MAX)*(1 - np.clip(rho, 0, 1)**n)**(1/n), 0.0)
    out = np.full_like(d, np.inf); k = int(round(t/h))
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            r2 = (i*h)**2 + (j*h)**2
            if r2 > t*t + 1e-9: continue
            sh = np.full_like(d, 0.0)
            ys0, ys1 = max(i, 0), d.shape[0] + min(i, 0); zs0, zs1 = max(j, 0), d.shape[1] + min(j, 0)
            sh[ys0 - i:ys1 - i, zs0 - j:zs1 - j] = d[ys0:ys1, zs0:zs1]
            out = np.minimum(out, sh - math.sqrt(t*t - r2))
    f = RegularGridInterpolator((ys, zs), out, bounds_error=False, fill_value=0.0)
    return lambda y, z: f(np.c_[np.asarray(y, float), np.asarray(z, float)])


def _bracket(x_back, x_front, y0, y1, zc, h, into=8.0):
    """A boss from the dome's inside (x_back) forward to x_front, y0..y1 wide, h tall round zc, with a 45 deg gusset
    under it; both reach `into` mm past the inside, to be clipped by the dome's outside, so the gusset meets the wall
    wherever the dome deepens below it (the dome prints standing up: nothing starts in mid-air)."""
    import cad
    xb = x_back - into; L = x_front - xb
    pts = [[xb, y, zc + h/2] for y in (y0, y1)] + [[x_front, y, zc + h/2] for y in (y0, y1)] + \
          [[x_front, y, zc - h/2] for y in (y0, y1)] + [[xb, y, zc - h/2 - L] for y in (y0, y1)]
    return cad.hull(np.array(pts))


def dome_c(p):
    """The domed back: shell (2 mm square to its surface), its rim on the front plate's back (x = -FW), a 3 mm floor on
    the base, the five snap fingers, the clamp that holds the lift post on its pegs, the cable clip's channel. It
    prints standing on its floor: the outside all walls, the fingers and the clamp on 45 deg undersides, the clip's
    mouth at 45 deg."""
    import cad
    from shapely.geometry import box as sbox, Polygon
    outline = _poly_above(FOOT_Z)(p)
    outer = _warped(outline, -FW, lambda y, z: depth(y, z))
    inner = _inner_depth()
    # the inside's top is a small dome under the tip (overhangs past 45 deg only its last ~15 mm, 28 mm across:
    # it prints with a little sag, unseen; a 45 deg gable instead would make the top third solid, +40 cm3)
    cav_pg = outline.buffer(-fd.RIM, join_style=1).intersection(sbox(-300, FOOT_Z + FLOOR, 300, 400))
    cav = _warped(cav_pg, -FW + 0.5, lambda y, z: np.maximum(inner(y, z), FW + 0.6), ext=1.0)
    s = outer - cav
    Lo = layout_c(p)
    (py0, py1), (pz0, pz1) = Lo["post"]                                # the post's clamp: touches its back face
    xin = -float(inner([0.5*(py0 + py1)], [0.5*(pz0 + pz1)])[0])
    s = s + (_bracket(xin, STAGE_X[0] - CLAMP_GAP, py0 + 0.5, py1 - 0.5, 0.5*(pz0 + pz1), pz1 - pz0 - 2.0) ^ outer)
    for ys, zs, side in SNAPS:                                         # the snap fingers
        s = s + (_finger(-float(inner([ys], [zs])[0]), ys, zs, side) ^ outer)
    s = s - _cable_notch(p, -DOME["D"] - 5, -FW + 0.01)
    return ss.drop_slivers(s)


def _barb_pts(ys, zs, side, x_tip, z_bot, gap=0.0):
    """The finger's barb (its profile in x-y: lead-in, flat, 45 deg back), from the finger's face out by SNAP['barb'];
    z from the finger's sloping underside (z_bot(x)) to its top. gap inflates it (the catch's notch)."""
    S = SNAP; y0 = ys + side*S["t"]/2
    prof = [(x_tip + gap, 0.0), (x_tip - S["lead"], S["barb"] + gap), (x_tip - S["lead"] - S["flat"], S["barb"] + gap),
            (x_tip - S["lead"] - S["flat"] - S["barb"] - gap, 0.0), (x_tip + gap, -0.3), (x_tip - S["lead"] - S["flat"] - S["barb"] - gap, -0.3)]
    return [[x, y0 + side*hh, z] for x, hh in prof for z in (z_bot(x) - gap, zs + S["h"]/2 + gap)]


def _finger(x_in, ys, zs, side):
    """A snap finger from the dome's inside (x_in) forward to just behind the plate, at (ys, zs), its barb toward
    `side` (+-y). A vertical plate when the dome prints: it flexes sideways, along its layers; its underside rises at
    45 deg to the tip, and its root reaches 10 mm past the inside (clipped by the dome's outside) so that underside
    meets the wall even where the dome deepens below the finger (nothing starts in mid-air)."""
    import cad
    S = SNAP; x_tip = -FW - 0.4; x_root = x_in - 10.0; t = S["t"]
    z_bot = lambda x: zs - S["h"]/2 - (x_tip - x)
    plate = cad.hull(np.array([[x, ys + dy, z] for dy in (-t/2, t/2) for x, z in
                               ((x_tip, zs + S["h"]/2), (x_tip, z_bot(x_tip)), (x_root, zs + S["h"]/2), (x_root, z_bot(x_root)))]))
    return plate + cad.hull(np.array(_barb_pts(ys, zs, side, x_tip, z_bot)))


def _catch(ys, zs, side):
    """The finger's catch on the front plate's back: a block beside the finger's path, its notch the barb's shape (with
    clearance) where the finger ends; its back end chamfered. Prints face down: the notch's roof is the barb's 45 deg."""
    import cad
    S = SNAP; x_tip = -FW - 0.4; g = S["gap"]
    y_face = ys + side*(S["t"]/2 + g); y_out = y_face + side*S["block"]
    ya, yb = sorted((y_face, y_out)); za, zb = zs - S["h"]/2 - 1.0, zs + S["h"]/2 + 1.0
    blk = cad.cube(-FW - S["reach"], -FW + 0.01, ya, yb, za, zb)
    ch = cad.hull(np.array([[x, y, z] for z in (za - 1, zb + 1) for x, y in
                            ((-FW - S["reach"] - 0.01, y_face - side*0.01), (-FW - S["reach"] + 1.0, y_face - side*0.01), (-FW - S["reach"] - 0.01, y_face + side*1.0))]))
    z_bot = lambda x: zs - S["h"]/2 - (x_tip - x)
    notch = cad.hull(np.array([[x, y, z] for x, y, _ in _barb_pts(ys, zs, side, x_tip, z_bot, g) for z in (za - 1, zb + 1)]))
    return blk - ch - notch


def _cable_notch(p, x0, x1, teardrop=True):
    """The cable clip at the egg's edge (where v4's was), through x0..x1: the cable's hole and its mouth to the edge,
    the mouth rising at 45 deg (the cable hangs down: it can't slip out; and the dome, standing up to print, has no
    flat roof over it that would leave the edge above floating). In the dome (teardrop) hole and mouth are one convex
    channel, the hole's top a 45 deg point: no sliver between them starts in mid-air; the front plate's narrow mouth
    holds the cable."""
    import cad
    (yk, zk), r = ss.egg_clip(p), CABLE["d"]/2 + 0.1; half = CABLE["mouth"]/2 if not teardrop else r + 0.1
    pts = [[yk + r*math.cos(a), zk + r*math.sin(a)] for a in np.linspace(0.0, 2*math.pi, 25)[:-1]]
    if teardrop: pts.append([yk, zk + r*math.sqrt(2)])
    hole = cad.hull(np.array([[x, y, z] for x in (x0, x1) for y, z in pts]))
    u, n = np.array([1.0, 1.0])/math.sqrt(2), np.array([-1.0, 1.0])/math.sqrt(2); c = np.array([yk, zk])
    band = [c + n*half, c - n*half, c + n*half + 25*u, c - n*half + 25*u]
    band = [[x, y, z] for x in (x0, x1) for y, z in band]
    if teardrop: return cad.hull(np.array([[x, y, z] for x in (x0, x1) for y, z in pts] + band))
    return hole + cad.hull(np.array(band))


def front_plate_c(p, r=2.0, steps=16):
    """The front plate: the egg's outline 3 mm thick (its face at x = 0, the front edge rounded r), the 6 mm foot in
    the base's socket, the window behind the hood, the tag's pocket, the cable clip; on its back: the lift post's two
    pegs, the side stops, the up-stops (the crossbar's, and the hanger's as a backup), the snaps' catches. Prints
    face down."""
    import cad
    from shapely.geometry import box as sbox
    poly = ss.egg_outline(p)
    s = _slab(poly, -FW, -r)
    for k in range(steps):
        x0 = -r + k*r/steps; xm = (k + 0.5)*r/steps; d = r - math.sqrt(max(r**2 - xm**2, 0.0))
        s = s + _slab(poly.buffer(-d, join_style=1), x0, x0 + r/steps + 0.001)
    s = s + _slab(poly.intersection(sbox(-300, -10, 300, p.T + p.socket_h)), -p.board_t, -FW + 0.01)   # the foot: 6 mm, in the socket
    Lo = layout_c(p)
    (wy0, wy1), (wz0, wz1) = Lo["window"]
    s = s - _slab(sbox(wy0, wz0, wy1, wz1).buffer(-3, join_style=2).buffer(3, join_style=1), -FW - 1, 1)
    tw = p.tag_size/2 + 2.0 + 0.15
    s = s - cad.cube(-1.6, 1, -tw, tw, ss.EGG_TAG_CZ - tw, ss.EGG_TAG_CZ + tw)
    s = s - _cable_notch(p, -p.board_t - 1, 1, teardrop=False)
    (py0, py1), _ = Lo["post"]
    for zp in PEG["z"]:                                                 # the post's pegs
        s = s + cad.cyl_x(PEG["d"]/2, -FW - PEG["h"], -FW + 0.01, 0.5*(py0 + py1), zp)
    sz = Lo["stops"]["z"]
    for k in ("left", "right"):                                         # the side stops (meet the crossbar's ends)
        s = s + cad.cube(-STOP_DEPTH, -FW + 0.01, *Lo["stops"][k], *sz)
    (uy0, uy1), (uz0, uz1) = Lo["up_stop"]
    s = s + cad.cube(STAGE_X[0], -FW + 0.01, uy0, uy1, uz0, uz1)
    (cy0, cy1), (cz0, cz1) = Lo["cross_stop"]                           # the hood's up-stop: over the crossbar
    s = s + cad.cube(-STOP_DEPTH, -FW + 0.01, cy0, cy1, cz0, cz1)
    for ys, zs, side in SNAPS:
        s = s + _catch(ys, zs, side)
    s = s + _lip(p)
    s = s + _post_ears(Lo) - _pin_hole(-30.0, 60.0)
    return ss.drop_slivers(s)


def _pin_hole(y0, y1):
    """POST_PIN's hole along y, y0..y1: round (1.85 mm bridges fine printed sideways, in the face-down front plate and
    the stage on its back alike; a teardrop's point would sit on the loaded side in both and let the post move)."""
    import manifold3d as m3
    P = POST_PIN
    return m3.Manifold.cylinder(y1 - y0, P["hole"]/2, P["hole"]/2, 32).rotate((-90, 0, 0)).translate((P["x"], y0, P["z"]))


def _post_ears(Lo):
    """The two ears beside the lift post (front plate's back), POST_PIN's hole through them (cut in front_plate_c)."""
    import cad
    P = POST_PIN; (py0, py1), _ = Lo["post"]; g, t = P["ear_gap"], P["ear_t"]
    return (cad.cube(P["ear_x"], -FW + 0.01, py0 - g - t, py0 - g, *P["ear_z"]) +
            cad.cube(P["ear_x"], -FW + 0.01, py1 + g, py1 + g + t, *P["ear_z"]))


def _lip(p):
    """LIP: a band on the front plate's back just inside the dome's side wall, broken at the dome's floor and round the
    cable clip."""
    from shapely.geometry import Point, box as sbox
    L = LIP; outline = ss.egg_outline(p)
    band = outline.buffer(-(fd.RIM + L["gap"]), join_style=1).difference(outline.buffer(-(fd.RIM + L["gap"] + L["w"]), join_style=1))
    band = band.intersection(sbox(-300, FOOT_Z + FLOOR + 1.0, 300, 400)).difference(Point(*ss.egg_clip(p)).buffer(7.0))
    import cad
    pieces = list(getattr(band, "geoms", [band]))                       # the clip's gap splits it in two
    return cad.union([_slab(pg, -FW - L["h"], -FW + 0.01) for pg in pieces if pg.area > 1.0])


def stage_c_print(p, reach=0.6):
    """The stage as printed (one piece, half-lifted: the lift blades straight with everything they carry z_neutral
    above rest), the post reaching forward to the plate with its two peg holes; laid flat (x -> plate z). Returns the
    manifold in plate coordinates and the transform."""
    import cad
    Lo = layout_c(p); zn = p.z_neutral; ty = Y_BLADES["t"]; tz = Z_BLADES["t"]
    (py0, py1), (pz0, pz1) = Lo["post"]; (ty0, ty1), (tz0, tz1) = Lo["tab"]
    (hy0, hy1), (hz0, hz1) = Lo["hanger"]; (xy0, xy1), (xz0, xz1) = Lo["cross"]; (cy0, cy1), (cz0, cz1) = Lo["carriage"]
    post = _post(Lo)
    moving = [_box(ty0, ty1, tz0 + zn, tz1 + zn), _box(hy0, hy1, hz0 + zn, hz1 + zn), _box(xy0, xy1, xz0 + zn, xz1 + zn)]
    zc = Lo["zc"] + zn
    car = _carriage(Lo, zn)
    blades = []
    for zr in LIFT_Z:                                                   # straight, post to tab
        blades.append(_box(ty1 - reach, py0 + reach, zr - tz/2, zr + tz/2))
    for yb in Lo["hang_y"]:                                             # straight, hanger to crossbar
        blades.append(_box(yb - ty/2, yb + ty/2, xz1 + zn - reach, hz0 + zn + reach))
    m = cad.union([post, car] + moving + blades)
    T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -STAGE_X[0]], [0, 0, 0, 1.0]])
    return m.transform(T[:3].tolist()), T


def _post(Lo):
    """The lift post: its full width above POST_STEP_Z, 4 mm wide below it (the lower lift blade's root), reaching
    forward to the front plate, with the two pegs' holes and POST_PIN's hole across it."""
    import cad
    (py0, py1), (pz0, pz1) = Lo["post"]; (ly0, ly1), (lz0, lz1) = Lo["post_low"]
    post = _box(py0, py1, pz0, pz1, x1=-FW) + _box(ly0, ly1, lz0, lz1 + 0.01, x1=-FW)
    for zp in PEG["z"]:
        post = post - cad.cyl_x(PEG["d"]/2 + PEG["fit"], -FW - PEG["h"] - 0.5, -FW + 1, 0.5*(py0 + py1), zp)
    return post - _pin_hole(py0 - 1.0, py1 + 1.0)


def stage_c_parts(p, dy=0.0, dz=0.0):
    """The stage posed for the hood at (dy, dz) from rest (renders): {name: trimesh}."""
    import trimesh
    def tm(man): q = man.to_mesh(); return trimesh.Trimesh(np.asarray(q.vert_properties)[:, :3], np.asarray(q.tri_verts))
    def T(m, ty, tz): m = m.copy(); m.apply_translation([0.0, ty, tz]); return m
    import cad
    Lo = layout_c(p); L = Y_BLADES["L"]; Lz = Z_BLADES["L"]
    V = 0.6*dy**2/L; wz = dz - p.z_neutral; uz = 0.6*wz**2/Lz          # the carriage's rise, the tab's pull toward the post
    (py0, py1), (pz0, pz1) = Lo["post"]; (ty0, ty1), (tz0, tz1) = Lo["tab"]
    out = {"post": tm(_post(Lo))}
    out["hanger"] = T(tm(_box(*Lo["hanger"][0], *Lo["hanger"][1]) + _box(ty0, ty1, tz0, tz1)), uz, dz)
    out["carriage"] = T(tm(_carriage(Lo, 0.0)), dy + uz, dz + V)
    for k, yb in enumerate(Lo["hang_y"]):
        out[f"yblade{k}"] = T(_blade((yb, Lo["hanger"][1][0]), (0.0, -1.0), (1.0, 0.0), L, Y_BLADES["t"], dy), uz, dz)
    for k, zr in enumerate(LIFT_Z):
        out[f"zblade{k}"] = _blade((py0, zr), (-1.0, 0.0), (0.0, 1.0), Lz, Z_BLADES["t"], wz)
    return out


def _carriage(Lo, dz):
    """The carriage (its arm socket with crush ribs and lead-in) and its crossbar, with the pin's hole down through both
    (blind 1.5 mm above the carriage's bottom), dz above rest. The hole is horizontal when the stage prints flat: its
    top a 45 deg point (toward +x, the print's up)."""
    import cad
    J = ARM_JOINT; (cy0, cy1), (cz0, cz1) = Lo["carriage"]; zc = Lo["zc"] + dz
    hy, hz = ARM[0]/2 + J["fit"], ARM[1]/2 + J["fit"]; x0, x1 = J["floor"], STAGE_X[1] + 1.0
    car = _box(cy0, cy1, cz0 + dz, cz1 + dz) + _box(*Lo["cross"][0], Lo["cross"][1][0] + dz, Lo["cross"][1][1] + dz)
    car = car - cad.cube(x0, x1, -hy, hy, zc - hz, zc + hz)
    L = J["lead"]; xm = STAGE_X[1]                                         # the socket's mouth: 45 deg lead-in
    car = car - cad.hull(np.array([[xm - L, y, z] for y in (-hy, hy) for z in (zc - hz, zc + hz)] +
                                  [[xm + 0.01, y, z] for y in (-hy - L, hy + L) for z in (zc - hz - L, zc + hz + L)]))
    rh, rw = J["rib_h"], J["rib_w"]; xr0, xr1 = x0 + 0.5, xm - L - 0.3
    ribs = []
    for sz in (-1, 1):                                                    # top and bottom faces, two ribs each
        for yr in (-5.0, 5.0):
            zf = zc + sz*hz
            ribs.append(cad.hull(np.array([[x, yr + dy_, zf] for x in (xr0, xr1) for dy_ in (-rw/2, rw/2)] +
                                          [[x, yr, zf - sz*rh] for x in (xr0, xr1)] + [[x, yr + dy_, zf + sz*0.2] for x in (xr0, xr1) for dy_ in (-rw/2, rw/2)])))
    for sy in (-1, 1):                                                    # the side faces, two ribs each
        for zr in (zc - 2.5, zc + 2.5):
            yf = sy*hy
            ribs.append(cad.hull(np.array([[x, yf, zr + dz_] for x in (xr0, xr1) for dz_ in (-rw/2, rw/2)] +
                                          [[x, yf - sy*rh, zr] for x in (xr0, xr1)] + [[x, yf + sy*0.2, zr + dz_] for x in (xr0, xr1) for dz_ in (-rw/2, rw/2)])))
    car = car + cad.union(ribs)
    r = J["pin_hole"]/2; zt = Lo["cross"][1][1] + dz + 1.0; zb = cz0 + dz + J["pin_floor"]
    pin = cad.hull(np.array([[J["pin_x"] + r*math.cos(a), r*math.sin(a), z] for a in np.linspace(0, 2*math.pi, 25)[:-1] for z in (zb, zt)] +
                            [[J["pin_x"] + r*math.sqrt(2), 0.0, z] for z in (zb, zt)]))
    return car - pin


def _zcyl(r, z0, z1, x, y, n=40):
    import manifold3d as m3
    return m3.Manifold.cylinder(z1 - z0, r, r, n).translate((x, y, z0))


def hood_c_cad(p):
    """v4's hood (flexdock.hood_v4_cad) for the magnetic pair as it is (PLUG_C), with the arm reaching ARM_JOINT's floor
    (a 0.8 mm lead-in round its end) and the filament pin's hole down through it (0.1 mm toward the hood, so the pin
    draws it home), the leaf spring behind the cable head (SPRING) and the lead-in (GUIDE) in front of the adapter."""
    import cad
    p = dataclasses.replace(p, **PLUG_C); goob = CABLE_HEAD == "goobay"
    # the strap dock's pocket and catch; for the right-angle head also its channel and stop rib, cut for a face 0.15 mm
    # further back: its rib stands 0.15 mm off the head's body, and the spring takes that up, so the face rests at
    # adapter_face_x (the goobay's stack gets its own room and stop below)
    J = ARM_JOINT
    m = fd.hood_v4_cad(p, head=False) if goob else fd.hood_v4_cad(dataclasses.replace(p, face_offset=p.face_offset - 0.15))
    (x0, _), _, (z0, z1) = fd.arm_offset(p); h = p.hood_home()
    xe = J["floor"] - h[0]; za, zb = z0 - h[2], z1 - h[2]; yo = ARM[0]/2; L = J["lead"]
    end = cad.hull(np.array([[xe, y, z] for y in (-yo + L, yo - L) for z in (za + L, zb - L)] +
                            [[xe + L, y, z] for y in (-yo, yo) for z in (za, zb)] + [[x0 - h[0] + 0.5, y, z] for y in (-yo, yo) for z in (za, zb)]))
    m = m - cad.cube(xe - 5.0, x0 - h[0] + 0.5, -yo - 1, yo + 1, za - 1, zb + 1) + end
    m = m - _zcyl(J["arm_hole"]/2, za - 1.0, zb + 1.0, J["pin_x"] + J["draw"] - h[0], 0.0)
    t, xa, xb = ARM_TRIM, J["floor"] - 1.0 - h[0], STAGE_X[1] + 1.0 - h[0]        # thinner where the crush ribs bear
    m = m - (cad.cube(xa, xb, -yo - 1, yo + 1, za - 1, zb + 1) - cad.cube(xa - 1, xb + 1, -yo + t, yo - t, za + t, zb - t))
    pv, ph, pl = p.plug; xf, zc = p.adapter_face_x, p.adapter_z; xn0 = xf + GUIDE["gap"]
    m = m - cad.cube(xf - 3.0, xn0 - 0.02, -ph/2 - 0.15, ph/2 + 0.15, -1.0, zc + pv/2 + p.pocket_top)   # the adapter's way up as far
    # forward as the lead-in's stadium (through the battery stop's back edge): any length in FACE's window goes in
    if goob: return ss.drop_slivers(m + _guide(p) - _goobay_room(p) + _goobay_stop(p) + _goobay_roof(p) + _goobay_catch(p) + _cord_seat(p))
    return ss.drop_slivers(m + _guide(p) - _spring_room(p) + _spring(p) + _rib_down(p) - _cable_slot(p))


def float_x(p):
    """The floating stack (FLOAT, p with PLUG_C), hood frame x: the adapter's face at the front and rear stops, the
    stack's depth (adapter + collar + goobay), the goobay's front face at the front stop, its back at both stops, the
    rear wall's inside, the gap between them on the stack's way up, the socket's axis at the front stop."""
    pv, ph, pl = p.plug; G = GOOBAY
    f1, f0 = p.tip_face_x + FLOAT["fwd"], p.tip_face_x + FLOAT["back"]
    depth = pl + G["collar"] + G["u"]; wall = p.hood_rear + 3.0            # dock.hood_pieces' rear wall: 3 mm
    return dict(face_fwd=f1, face_back=f0, depth=depth, front_fwd=f1 - pl - G["collar"], back_fwd=f1 - depth,
                back_back=f0 - depth, wall=wall, rise_gap=f1 - depth - wall, socket_x=f1 - pl - G["collar"] - G["u"]/2)


def _goobay_room(p):
    """The stack's room (hood frame), all of it open below (the stack goes in from below as one piece, held forward):
    the adapter's pocket back to the rear stop; the collar's; the goobay's and the cable plug's channel from the rear
    wall's inside to the goobay's front face at the front stop, across to the left wall; the pocket in the rear wall
    (the rear stop: the goobay's back with the face at the rear stop), from just above the arm to `top`; the left
    wall's opening for the plug's moulding (its lower corners chamfered) and a 45 deg chamfer along the rear wall's
    bottom inside edge (a stack going up too far back is cammed forward)."""
    import cad
    pv, ph, pl = p.plug; G, F = GOOBAY, FLOAT; X = float_x(p); zc = p.adapter_z; a = G["plug_at"]
    ztop = zc + F["top"]; y_out = p.side_in + p.wall_t
    s = cad.cube(X["face_back"] - pl - 0.2, X["face_fwd"] - pl + 1.0, -ph/2 - 0.15, ph/2 + 0.15, -1.0, zc + pv/2 + p.pocket_top)
    s = s + cad.cube(X["face_back"] - pl - G["collar"] - 0.3, X["face_fwd"] - pl + 0.5, -(G["collar_l"]/2 + 0.4), G["collar_l"]/2 + 0.4, -1.0, ztop)
    s = s + cad.cube(X["wall"] - 0.01, X["front_fwd"], -(a + 0.35), p.side_in + 0.01, -1.0, ztop)
    s = s + cad.cube(X["back_back"], X["wall"] + 0.01, -(a + 0.35), y_out + 1.0, zc - F["wall_lo"], ztop)
    x0, x1 = X["back_back"], X["front_fwd"] + 0.3; ys = (p.side_in - 0.5, y_out + 1.0); zb = p.side_bot - 2.0; ch = 1.2
    s = s + cad.cube(x0, x1, ys[0], ys[1], zb, ztop)
    s = s + cad.hull(np.array([[x, y, z] for y in ys for x, z in ((x0 - ch, zb), (x0 + 0.01, zb), (x0 + 0.01, p.side_bot + ch))]))
    s = s + cad.hull(np.array([[x, y, z] for y in ys for x, z in ((x1 + ch, zb), (x1 - 0.01, zb), (x1 - 0.01, p.side_bot + ch))]))
    c = 1.0; w = X["wall"]
    return s + cad.hull(np.array([[x, y, z] for y in (-(a + 0.35), y_out + 1.0) for x, z in ((w - c, -0.01), (w + 0.01, -0.01), (w + 0.01, c))]))


def _goobay_stop(p):
    """The front stop: a rib before the goobay's front face (at the front stop) left of its collar and the adapter's
    side, from the top plate down to FLOAT rib_down below the axis, its foot chamfered 45 deg on the goobay's side."""
    import cad
    pv, ph, pl = p.plug; X = float_x(p); zc = p.adapter_z; c = 1.5
    x0, x1 = X["front_fwd"], X["face_fwd"] - pl + 0.6; y0, y1 = ph/2 + 0.35, 12.5; z0, z1 = zc - FLOAT["rib_down"], p.trunk_top + 0.01
    return cad.hull(np.array([[x, y, z] for y in (y0, y1) for x, z in ((x0 + c, z0), (x1, z0), (x0, z0 + c), (x0, z1), (x1, z1))]))


def _goobay_roof(p):
    """GCATCH roof: over the goobay and the cable plug's moulding (the goobay's far end to the left wall, front to back
    over the float), GCATCH['roof'] above the higher of the two, up into the top plate (solid: it grows from the plate
    in the upside-down print)."""
    import cad
    G = GOOBAY; X = float_x(p); z0 = p.adapter_z + max(G["t"], CORD["t"])/2 + GCATCH["roof"]
    return cad.cube(X["wall"] - 0.01, X["front_fwd"], -(G["plug_at"] + 1.1), p.side_in + 0.01, z0, p.trunk_top + 0.01)


def _goobay_catch(p, flex=0.0, y=None, thick=None):
    """GCATCH's finger and hook (hood frame), `flex` mm forward (bent aside, as the goobay passes it going in); y and
    thick for another part of the stack (CSEAT: the plug's moulding)."""
    import cad
    C, G = GCATCH, GOOBAY; X = float_x(p)
    xi = X["front_fwd"] + 0.15; xo = xi + C["t"]; xt = xi - C["hook"]; y0, y1 = y or C["y"]
    zt = p.adapter_z - (thick or G["t"])/2 - C["gap"]; zb = zt - C["tip"] - C["hook"]/math.tan(math.radians(C["lead"]))
    finger = cad.cube(xi, xo, y0, y1, zb, p.trunk_top + 0.01)
    hook = cad.hull(np.array([[x, y, z] for y in (y0, y1) for x, z in ((xi + 0.01, zt), (xt, zt), (xt, zt - C["tip"]), (xi + 0.01, zb))]))
    return (finger + hook).translate((flex, 0, 0))


def _cord_seat(p, flex=0.0):
    """CSEAT: the plug moulding's click (GCATCH's finger under the moulding, by the left wall)."""
    return _goobay_catch(p, flex, CSEAT["y"], CORD["t"])


def goobay_stack(p, face_x, dz=0.0, cable=True):
    """The magnetic adapter, the goobay and the cable plug as one solid (hood frame), the adapter's face at face_x, dz
    below or above its seat; cable: the strain relief and 6 mm of cable beyond it."""
    import cad, manifold3d as m3
    pv, ph, pl = p.plug; G, C = GOOBAY, CORD; zc = p.adapter_z + dz; a = G["plug_at"]
    x_b = face_x - pl; xg = x_b - G["collar"]; xs = xg - G["u"]/2; y_s = G["L"] - a
    s = cad.adapter_solid(dataclasses.replace(p, face_offset=face_x - p.tip_face_x)).translate((0, 0, dz))
    s = s + cad.cube(xg, x_b, -G["collar_l"]/2, G["collar_l"]/2, zc - G["t"]/2 + 0.5, zc + G["t"]/2 - 0.5)
    s = s + cad.cube(xg - G["u"], xg, -a, y_s, zc - G["t"]/2, zc + G["t"]/2)
    s = s + cad.cube(xs - C["w"]/2, xs + C["w"]/2, y_s, y_s + C["l"], zc - C["t"]/2, zc + C["t"]/2)
    if cable:
        y1 = y_s + C["l"]
        s = s + m3.Manifold.cylinder(C["relief"], C["relief_d"]/2, C["cable_d"]/2 + 0.3, 48).rotate((-90, 0, 0)).translate((xs, y1, zc))
        s = s + m3.Manifold.cylinder(6.0, C["cable_d"]/2, C["cable_d"]/2, 32).rotate((-90, 0, 0)).translate((xs, y1 + C["relief"], zc))
    return s


def goobay_cable_out(p):
    """Where the cable leaves the stack (hood frame mm): the strain relief's end, the stack at its front stop."""
    G, C = GOOBAY, CORD; X = float_x(p)
    return (round(X["socket_x"], 2), round(G["L"] - G["plug_at"] + C["l"] + C["relief"], 2), p.adapter_z)


def face_window(p):
    """Where the adapter's face may rest (hood frame x, p with PLUG_C): from the tip's face at the battery stop (+ FIT
    reach) to the lead-in's narrow stadium (- reach); the stadium's back and front; the stop rib's back face, which
    puts an elbow + adapter of FIT len_mid in the middle; and the elbow + adapter lengths that rest in the window."""
    pv, ph, pl = p.plug; ed = p.angle_elbow[0]; fl = p.tip_front[2]
    tip = p.tip_face_x - (p.back_face_x - p.stop_face_x)                 # the battery stop lets the duck 0.4 mm further
    xn0 = p.adapter_face_x + GUIDE["gap"]; xn1 = xn0 + GUIDE["narrow"]
    lo, hi = tip + FIT["reach"], xn0 - FIT["reach"]; rib = p.adapter_face_x - pl - ed
    return dict(tip_at_stop=round(tip, 3), stadium=(round(xn0, 3), round(xn1, 3)), base_at_stop=round(tip + fl, 3),
                base_margin=round(tip + fl - xn1, 3), window=(round(lo, 3), round(hi, 3)), rib=round(rib, 3),
                lengths=(round(lo - rib, 2), round(hi - rib, 2)), mouth=round(xn1 + GUIDE["flare"], 3),
                shell_at_stop=round(p.shell_back_x - (p.back_face_x - p.stop_face_x), 3))


def _rib_down(p, low=12.0, c=1.0):
    """The cable head's stop rib (cad.hood's, at the left of the elbow) carried down to `low` below the head's axis,
    with a 45 deg chamfer at its foot on the head's side: on its way up from below the pair is pushed back as soon as
    it rises, so an adapter's face can't reach the lead-in's stadium (cad.hood's rib starts at the head's bottom)."""
    import cad
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug; ed, el, xe = p.angle_elbow
    x0 = p.adapter_face_x - pl - ed; x1 = p.adapter_face_x - 0.15 - pl + 0.4    # as cad.hood's rib (with the 0.15)
    y0, y1 = (el - xe) + 2.1, ph/2 + 2.2; z0, z1 = p.adapter_z - low, p.adapter_z - at/2 - 0.5 + 0.3
    return cad.hull(np.array([[x, y, z] for y in (y0, y1) for x, z in ((x0 + c, z0), (x1, z0), (x0, z0 + c), (x0, z1), (x1, z1))]))


def _cable_slot(p, w=5.6, ch=1.6):
    """The cable's exit through the hood's left side wall (cad.hood's, 4.8 mm), widened to w for a braided cable up to
    ~5 mm: round at the head's axis, open at the wall's bottom edge (the cable slips in from below), its mouth there
    chamfered ch."""
    import cad, manifold3d as m3
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug
    xc = p.adapter_face_x - 0.15 - pl - (aw - at/2); y0, y1 = p.side_in - 1.0, p.side_in + p.wall_t + 1.0; zb = p.side_bot - 2.0
    s = m3.Manifold.cylinder(y1 - y0, w/2, w/2, 48).rotate((-90, 0, 0)).translate((xc, y0, p.adapter_z))
    s = s + cad.cube(xc - w/2, xc + w/2, y0, y1, zb, p.adapter_z)
    for sx in (-1, 1):
        s = s + cad.hull(np.array([[xc + sx*(w/2 - 0.01), y, z] for y in (y0, y1) for z in (zb, p.side_bot + ch)] +
                                  [[xc + sx*(w/2 + ch), y, zb] for y in (y0, y1)]))
    return s


def head_back_x(p):
    """Where the cable head's body ends behind (hood frame x), the head on its stop rib (p with PLUG_C)."""
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug
    return p.adapter_face_x - pl - aw


def _spring_room(p):
    """The head's channel, `travel` deeper behind the head: the side wall's end there is the hard stop behind it."""
    import cad
    S = SPRING; al, aw, at = p.angle_plug; ed, el, xe = p.angle_elbow; xb = head_back_x(p)
    return cad.cube(xb - S["travel"], xb - 0.3, -xe - 0.65, p.side_in + 0.01, -1.0, p.adapter_z + at/2 + 0.3)


def _spring(p):
    """The leaf behind the head (SPRING): from the top plate down past the head's axis, its front face `preload` into
    the head's body (the head bends it back as it goes in), a 45 deg chamfer at the tip's front edge to cam it."""
    import cad
    S = SPRING; xb = head_back_x(p); xf = xb + S["preload"]; xr = xf - S["t"]; zt = p.adapter_z + S["tip_z"]; ztop = p.trunk_top + 1.0
    w = S["w"]/2
    return cad.hull(np.array([[x, y, z] for y in (-w, w) for x, z in
                              ((xr, zt), (xr, ztop), (xf - S["lead"], zt), (xf, zt + S["lead"]), (xf, ztop))]))


def _guide(p):
    """The lead-in in front of the adapter (hood frame): a collar from the hood's sleeve round the adapter (behind its
    face) forward to the mouth. Inside: room for the adapter (its face up to the stadium, and its way up from below),
    the tip-sized stadium (GUIDE['narrow'] deep), then a 45 deg flare (GUIDE['flare']). Open below: the side walls run
    straight down from the middle and end at the stadium's bottom, nothing below (the simulator's guide boxes)."""
    import cad, manifold3d as m3
    G = GUIDE; pv, ph, pl = p.plug; xf, zc = p.adapter_face_x, p.adapter_z
    xn0 = xf + G["gap"]; xn1 = xn0 + G["narrow"]; xm = xn1 + G["flare"]
    hw, hh = G["w"]/2, G["h"]/2; ow, oh = hw + G["flare"] + G["wall"] + 0.07, hh + G["flare"] + G["wall"] + 0.13   # off the hood's faces
    def stad(x0, x1, w, h):                                               # a w x h stadium (ends round, radius h/2)
        cs = m3.CrossSection.square((max(w - h, 0.01), 0.01), True).offset((h - 0.01)/2, m3.JoinType.Round, circular_segments=48)
        return m3.Manifold.extrude(cs, x1 - x0).transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, zc]], float))
    pts = lambda man: np.asarray(man.to_mesh().vert_properties)[:, :3]
    zb = zc - oh - 1.0
    collar = cad.cube(xf - 1.93, xm, -ow, ow, zc - oh, zc + oh)
    collar = collar - cad.cube(xf - 3.0, xn0 + 0.01, -ph/2 - 0.3, ph/2 + 0.3, -50.0, zc + pv/2 + 0.3)  # the adapter, its way up
    narrow = stad(xn0 - 0.01, xn1 + 0.03, 2*hw, 2*hh) + cad.cube(xn0 - 0.01, xn1 + 0.03, -hw, hw, zb, zc)   # (overlapping
    flare = cad.hull(np.vstack([pts(stad(xn1 - 0.02, xn1 - 0.01, 2*hw, 2*hh)),                             # cuts: no
                                pts(stad(xm - 0.01, xm + 0.05, 2*(hw + G["flare"]), 2*(hh + G["flare"])))]))   # coincident
    flare_below = cad.hull(np.array([[xn1 - 0.02, y, z] for y in (-hw, hw) for z in (zb, zc)] +            # faces)
                                    [[xm + 0.05, y, z] for y in (-hw - G["flare"] - 0.05, hw + G["flare"] + 0.05) for z in (zb, zc)]))
    # no bottom lip: as the duck scoots out its tip first slides down ~3 mm (the simulator's undocking), and a lip there
    # (or the stadium's lower arcs, or side walls reaching further down) would hold it; it is guided sideways (the big
    # error) and from above
    open_below = cad.cube(xf - 0.1, xm + 1.0, -(hw + G["flare"]) - 0.01, hw + G["flare"] + 0.01, zb - 1.0, zc - hh + 0.01)
    # the side walls' feet chamfered 45 deg on the adapter's side: an adapter longer than FIT allows, rising from below,
    # slides back off them instead of catching
    c = 0.6; z_ = zc - hh
    bevel = cad.hull(np.array([[x, y, z] for y in (-(ph/2 + 0.31), ph/2 + 0.31) for x, z in
                               ((xn0 - 0.02, z_ - 0.02), (xn0 + c, z_ - 0.02), (xn0 - 0.02, z_ + c))]))
    return collar - narrow - flare - flare_below - open_below - bevel


def fit_test_c(p):
    """The hood's slice round the magnetic adapter (its catch, the cable head's channel, stop rib and spring, the
    lead-in), printed as the hood is: to try the adapter, its cable and the duck's tip in before printing the hood."""
    import cad
    p = dataclasses.replace(p, **PLUG_C); G = GUIDE
    xm = p.adapter_face_x + G["gap"] + G["narrow"] + G["flare"]
    if CABLE_HEAD == "goobay":                  # from the rear wall's back face, across to the left wall (its opening)
        return hood_c_cad(p) ^ cad.cube(p.hood_rear, xm + 0.3, -9.6, p.side_in + p.wall_t + 0.01, p.adapter_z - 12.5, p.trunk_top + 3.5)
    return hood_c_cad(p) ^ cad.cube(head_back_x(p) - 2.5, xm + 0.3, -9.6, 9.6, p.adapter_z - 12.5, p.trunk_top + 3.5)

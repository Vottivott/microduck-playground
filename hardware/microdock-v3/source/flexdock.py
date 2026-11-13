"""The first spring egg (in development "MicroDock v4"; flexdock_c.py builds MicroDock v3 on it): the strap dock's hood (its funnel, plug and cable exit unchanged) held from behind by a printed PLA
spring stage hidden inside a thicker egg, instead of hanging from a hanger on four straps. Nothing above the hood.

The stage is a flat compliant mechanism (blades that bend, nothing that slides): two long vertical blades hang a
carriage (sideways, like a pendulum: the blades are always pulled, never pushed), and on that carriage a pair of
horizontal blade springs carries a smaller carriage up and down (the lift). An arm from the small carriage reaches out
through a window in the egg's face, hidden behind the hood, and the hood clips onto it. At rest the hood's weight sits on
a hard stop and the blades are straight, so nothing holds them bent while the dock waits: PLA's creep can't move the
hood's rest position (the stop sets it); it could only soften the springs a little, and the simulator docks across a
wide range of stiffness.

In the simulator the stage is what it does: the hood slides sideways, up and back (toward the board) on springs, with
the stops, without turning (the blades hold its orientation), and without the hanger, the straps, their sockets and
the tooth.
"""
import math, os
from dataclasses import dataclass
import numpy as np

import strapswing as ss
import swing


@dataclass
class FlexDock(ss.Straps):
    # the stage as the simulator sees it (N/m, mm); the CAD's blades are sized to these (flexure_springs)
    k_x: float = 3000.0             # back toward the board / out: the blades through their depth (a wall of 10 mm)
    k_y: float = 40.0               # sideways: the hanging blades bending, plus gravity (a pendulum)
    k_z: float = 40.0               # up: the lift springs
    y_travel: float = 16.0          # the side stops, either way
    z_travel: float = 14.0          # the up-stop above rest (the rest: the hood's weight on the down-stop)
    z_neutral: float = 0.0          # where the lift springs are straight, above rest (printed half-lifted: they push the
                                    # hood's weight a little at rest, and hold it a little when docked, never the whole lift)
    x_travel: tuple = (2.5, 3.0)    # back (the board stops the hood at rest_gap first), out
    hood_mass: float = 0.065        # the hood with its plug and cable head, the arm and the carriages
    damping_ratio: float = 0.03     # PLA blades ring a little (loss factor ~0.05)
    k_rot: float = 0.0              # N m/rad: >0 lets the hood tilt and turn on springs (a compliant joint in the arm)
    x_solref: float = 0.0           # >0 (s): toward the board / out as a stiff spring the solver can take (an equality with
                                    # this time constant: 0.01 s ~ 25 N/mm on the hood's mass), instead of k_x
    k_rx: float = 0.0               # N m/rad, per axis (roll, pitch, yaw), about rot_pivot: the stage's own rotational
    k_ry: float = 0.0               # stiffness (flexdock_c.rot_springs); >0 overrides k_rot for that axis
    k_rz: float = 0.0
    rot_pivot: tuple = ()           # world mm the hood turns about (the stage, behind the egg's face); () = the hood's centre
    rot_solref: tuple = ()          # per axis (roll, pitch, yaw) time constants (s): the tilt springs as soft equalities the
                                    # solver takes implicitly (explicit springs this stiff blow up at the sim's step);
                                    # flexdock_c.rot_solref calibrates them to k_rx, k_ry, k_rz
    guide: tuple = ()               # the lead-in in front of the adapter (flexdock_c.GUIDE): narrow w, h, gap to the face,
                                    # narrow depth, 45 deg flare (mm)[, how far its side walls reach below the stadium];
                                    # () = none
    tip_front: tuple = ()           # the magnetic tip as it is: a front part (w, h, length mm) on a bigger base (tip_w x
                                    # tip_h, the rest of tip_out); () = one block of tip_w x tip_h
    face_offset: float = -0.1       # the adapter's face at rest from the tip's face with the duck docked (+: toward the
                                    # duck; the egg-C hood's spring holds it there, the duck can push it back)
    cable_out: tuple = ()           # where the cable leaves the hood's parts (hood frame mm); () = swing.cable_exit's
                                    # (the right-angle head's axis at the left wall)
    catch_lift: float = 0.0         # the adapter catch's hooks raised this far (cad.catch_geometry): a preload
    pocket_top: float = 0.15        # the adapter pocket's roof over the adapter (cad.hood)

    @property
    def adapter_face_x(self): return self.tip_face_x + self.face_offset

    def cord_sites(self): return []                            # no straps, no tendons
    def hood_extra_pieces(self):                               # the funnel only (no strap sockets, no tabs)
        return {} if ss.FUNNEL.get("flat") else ss.funnel_pieces(self)
    def hanger_extra_boxes(self): return {}


def build_scene(p, robot_dir, scene_path=None):
    """The strap dock's scene without the hanger, the straps and the hood's tooth, the hood on the stage's springs."""
    import mujoco
    spec = swing.build_scene(p, robot_dir, scene_path)
    for g in list(spec.geoms):
        if g.name in ("dock_bar", "dock_bar_l", "dock_bar_r", "dock_ear_l", "dock_ear_r", "hood_tooth"):
            spec.delete(g)
    m = p.hood_mass
    def setj(name, k, rng):
        j = next(j for j in spec.joints if j.name == name)
        j.stiffness = np.array([k, 0.0, 0.0]); j.springref = p.z_neutral/1e3 if name == "hood_z" else 0.0
        j.damping = np.array([2*p.damping_ratio*np.sqrt(k*m), 0.0, 0.0])
        j.range = np.array(rng)/1e3; j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.solref_limit = np.array([0.004, 1.0])
    if p.x_solref > 0:
        j = next(j for j in spec.joints if j.name == "hood_x"); j.stiffness = np.zeros(3); j.damping = np.zeros(3)
        e = spec.add_equality(); e.type = mujoco.mjtEq.mjEQ_JOINT; e.name1 = "hood_x"
        e.solref = np.array([p.x_solref, 1.0])
    elif p.k_x > 0: setj("hood_x", p.k_x, (-p.x_travel[0], p.x_travel[1]))
    else: spec.delete(next(j for j in spec.joints if j.name == "hood_x"))      # rigid toward the board and out
    setj("hood_y", p.k_y, (-p.y_travel, p.y_travel))
    setj("hood_z", p.k_z, (0.0, p.z_travel))
    if len(p.guide) >= 5: add_guide(spec, p)
    per = dict(hood_rx=p.k_rx, hood_ry=p.k_ry, hood_rz=p.k_rz)
    piv = (np.asarray(p.rot_pivot, float) - p.hood_home())/1e3 if len(p.rot_pivot) == 3 else None
    for j in list(spec.joints):
        if j.name in ("hood_rx", "hood_ry", "hood_rz"):
            k = per[j.name] if per[j.name] > 0 else p.k_rot
            if k <= 0: spec.delete(j); continue
            if piv is not None: j.pos = piv                       # the stage's pivot, in the hood's frame
            ax = ("hood_rx", "hood_ry", "hood_rz").index(j.name)
            if len(p.rot_solref) == 3 and p.rot_solref[ax] > 0:       # an implicit spring: a soft equality to 0
                j.stiffness = np.zeros(3); j.damping = np.zeros(3)
                e = spec.add_equality(); e.type = mujoco.mjtEq.mjEQ_JOINT; e.name1 = j.name
                e.solref = np.array([p.rot_solref[ax], 1.0]); continue
            I = m*float(np.sum(piv**2)) if piv is not None else 6e-5
            j.stiffness = np.array([k, 0.0, 0.0]); j.springref = 0.0
            j.damping = np.array([2*p.damping_ratio*np.sqrt(k*I), 0.0, 0.0])
    return spec


def shape_tip(spec, p):
    """The magnetic tip as it is (dock's mag_tip is one block): its base (tip_w x tip_h) on the shell, the front part
    (tip_front) out to the face. After dock.modify_robot, which adds the tip."""
    import mujoco
    fw, fh, fl = p.tip_front; bl = p.tip_out - fl
    g = next(g for g in spec.geoms if g.name == "mag_tip")
    g.size = np.array([bl/2, p.tip_w/2, p.tip_h/2])/1e3; g.pos = np.array([p.shell_back_x - bl/2, 0.0, p.port_z])/1e3
    f = spec.body("trunk_base").add_geom(name="mag_tip_front", type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array([fl/2, fw/2, fh/2])/1e3,
                                         pos=np.array([p.tip_face_x + fl/2, 0.0, p.port_z])/1e3, rgba=[0.75, 0.76, 0.78, 1], mass=0.0)
    f.contype = 4; f.conaffinity = 4; f.friction = [0.3, 0.005, 0.0001]


def add_guide(spec, p):
    """The lead-in as the tip meets it (hood frame): the narrow section's walls and the flare's 45 deg walls (sides and
    top: no bottom lip), boxes only the magnetic tip (contype 4) touches. Flat walls at the stadium's extremes."""
    import mujoco
    w, h, gap, nd, fl = p.guide[:5]; xf, zc = p.adapter_face_x, p.adapter_z; hw, hh, t = w/2, h/2, 1.5
    x0 = xf + gap; x1 = x0 + nd; xm = x1 + fl; L = fl*math.sqrt(2)
    hb = next(b for b in spec.bodies if b.name == "hood")
    def box(name, c, half, ax=None, ang=0.0):
        g = hb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(half)/1e3, pos=np.array(c)/1e3, rgba=[0.9, 0.9, 0.9, 1])
        if ax is not None:
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.array(ax, float), ang); g.quat = q
        g.contype = 4; g.conaffinity = 4; g.friction = [0.3, 0.005, 0.0001]; g.mass = 0.0
        g.priority = 1; g.solref = [0.01, 1.0]; g.solimp = [0.95, 0.99, 0.0005, 0.5, 2.0]   # hard as PLA (the tip sank ~0.5 mm into soft walls)
    xc = (x0 + x1)/2; dn = p.guide[5] if len(p.guide) > 5 else 0.0   # the side walls reach this far below the stadium
    for s_ in (-1, 1):
        if s_ > 0: box(f"guide_n_z{s_}", (xc, 0, zc + s_*(hh + t/2)), (nd/2, hw + t, t/2))   # no bottom lip
        box(f"guide_n_y{s_}", (xc, s_*(hw + t/2), zc - dn/2), (nd/2, t/2, hh + dn/2))
        # flare walls: the inner face runs from (x1, edge) to (xm, edge + fl); a box of thickness t behind it
        cz = np.array([(x1 + xm)/2, hh + fl/2]) + np.array([-t/2/math.sqrt(2), t/2/math.sqrt(2)])   # mid-face, out by t/2
        if s_ > 0: box(f"guide_f_z{s_}", (cz[0], 0, zc + s_*cz[1]), (L/2, hw + fl + t, t/2), (0, 1, 0), -s_*math.pi/4)
        cy = np.array([(x1 + xm)/2, hw + fl/2]) + np.array([-t/2/math.sqrt(2), t/2/math.sqrt(2)])
        box(f"guide_f_y{s_}", (cy[0], s_*cy[1], zc), (L/2, t/2, hh + fl + t), (0, 0, 1), s_*math.pi/4)


# ------------------------------------------------------------------------------------------------ CAD (world mm)
# The egg is 25 mm thick: a 3 mm front wall (its face at x = 0, where the board's face was), a 19 mm cavity, a 3 mm back
# lid. The stage fills the cavity's depth (16 mm, 1.5 mm off both walls) and prints flat as one PLA piece: the blades are
# walls printed along their length (the strongest way for a flexure), 16 mm tall on the plate.
EGG_T = (3.0, 19.0, 3.0)                 # front wall, cavity, back lid
RIM = 2.4                                # the egg's side wall
STAGE_X = (-20.5, -4.5)                  # the stage through the cavity's depth
ARM = (16.0, 10.0, 18.0)                 # the hood's arm: width (y), height (z), centre's height on the hood (hood z)
ARM_FIT = 0.3                            # the arm in the carriage's socket: clearance per side (the hanger's lesson)
CARRIAGE = (24.0, 16.0)                  # the lift carriage behind the window: width, height
Z_BLADES = dict(t=0.4, L=42.0, at=(2.0, 14.0))    # lift blades: thickness, length, heights above the carriage's bottom
Y_BLADES = dict(t=0.4, L=64.0, y=(-18.0, -8.0, 46.0, 56.0))   # the hanging blades and where they are across the egg
FRAME = 6.0                              # the sideways carriage's bars and posts
STOP_GAP = 1.0                           # the lift carriage's down-stop under it at rest: 0 (it rests there); printed
                                         # half-lifted, the gap is z_neutral


def layout(p):
    """The stage's parts at rest (the hood hanging at rest_drop, centred): world y-z boxes and the blades' lines."""
    zc = p.hood_home()[2] + ARM[2]                       # the arm's centre height at rest
    cw, ch = CARRIAGE; zl = zc - ch/2                    # the carriage's bottom at rest
    L = Z_BLADES["L"]; yp = cw/2 + L                     # the post's inner face
    post = (yp, yp + FRAME)
    frame_y = (-cw/2 - 4.0 - FRAME, post[1])             # the sideways carriage's span (its left post 4 mm off the carriage)
    bar_bot = (zl - 1.0 - 5.0, zl - 1.0)                 # its bottom bar, and the down-stop block on it
    top_z0 = zl + ch + p.z_travel + 4.0                  # its top bar: above the up-stop block
    bar_top = (top_z0, top_z0 + FRAME)
    anchor = (bar_top[1] + Y_BLADES["L"], bar_top[1] + Y_BLADES["L"] + 8.0)
    return dict(zc=zc, carriage=((-cw/2, cw/2), (zl, zl + ch)), zblades=[zl + a for a in Z_BLADES["at"]],
                post=post, frame_y=frame_y, bar_bot=bar_bot, bar_top=bar_top, anchor=anchor,
                left_post=(frame_y[0], frame_y[0] + FRAME),
                down_stop=((-8.0, 8.0), (bar_bot[1], zl)), up_stop=((-8.0, 8.0), (zl + ch + p.z_travel, top_z0)),
                window=((-ARM[0]/2 - p.y_travel - 1.5, ARM[0]/2 + p.y_travel + 1.5),
                        (zc - ARM[1]/2 - 1.5, zc + ARM[1]/2 + p.z_travel + 1.5)),
                stops=dict(left=(frame_y[0] - p.y_travel - 6.0, frame_y[0] - p.y_travel), right=(post[1] + p.y_travel, post[1] + p.y_travel + 6.0)))


def springs(p, E=3500.0):
    """The stage's stiffness and the blades' strains from its dimensions (PLA, E in N/mm2): N/m, N/mm, %."""
    b = STAGE_X[1] - STAGE_X[0]
    ty, Ly, ny = Y_BLADES["t"], Y_BLADES["L"], len(Y_BLADES["y"]); tz, Lz = Z_BLADES["t"], Z_BLADES["L"]
    ky = ny*12*E*(b*ty**3/12)/Ly**3*1e3; kz = 2*12*E*(b*tz**3/12)/Lz**3*1e3
    kxy = ny*E*ty*b**3/Ly**3; kxz = 2*E*tz*b**3/Lz**3
    m = p.hood_mass; g = m*9.81/(0.83*Ly/1e3)                          # the hanging carriage's pendulum (N/m)
    eps = lambda t, d, L: 300*t*d/L**2
    return dict(k_y=round(ky + g, 1), k_y_blades=round(ky, 1), k_z=round(kz, 1), k_x=round(1/(1/kxy + 1/kxz), 1),
                strain_y_at_stop=round(eps(ty, p.y_travel, Ly), 2), strain_z_at_rest=round(eps(tz, p.z_neutral, Lz), 2),
                strain_z_docked_9mm=round(eps(tz, 9.0 - p.z_neutral, Lz), 2), strain_z_at_stop=round(eps(tz, p.z_travel - p.z_neutral, Lz), 2),
                rest_on_stop_N=round(m*9.81 - kz/1e3*p.z_neutral, 2))


def _box(y0, y1, z0, z1, x0=STAGE_X[0], x1=STAGE_X[1]):
    import cad
    return cad.cube(x0, x1, y0, y1, z0, z1)


def _blade(fixed, along, across, L, t, d=0.0, n=48, x=STAGE_X):
    """A blade as a mesh (trimesh): from `fixed` (y, z) along `along` (unit y-z) for L, t thick across `across`; its far
    end displaced d across (fixed-guided: the S of a beam clamped at both ends) and pulled back by the arc."""
    import trimesh
    s = np.linspace(0.0, 1.0, n + 1); w = d*(3*s**2 - 2*s**3); back = 0.6*d**2/L*s   # the arc's shortening, along
    f, a, c = np.asarray(fixed, float), np.asarray(along, float), np.asarray(across, float)
    V = []
    for k in range(n + 1):
        cy, cz = f + a*(s[k]*L - back[k]) + c*w[k]
        for sx in (x[0], x[1]):
            for st in (-t/2, t/2):
                V.append((sx, cy + c[0]*st, cz + c[1]*st))
    V = np.array(V); F = []
    for k in range(n):
        i0, i1 = 4*k, 4*(k + 1)
        for a0, a1 in ((0, 1), (1, 3), (3, 2), (2, 0)):
            F += [(i0 + a0, i1 + a0, i1 + a1), (i0 + a0, i1 + a1, i0 + a1)]
    F += [(0, 2, 3), (0, 3, 1), (4*n + 0, 4*n + 1, 4*n + 3), (4*n + 0, 4*n + 3, 4*n + 2)]
    m = trimesh.Trimesh(V, np.array(F), process=True)
    if m.volume < 0: m.invert()
    return m


def stage_parts(p, dy=0.0, dz=0.0):
    """The stage posed for the hood at (dy, dz) from rest (sideways, up; mm): {name: trimesh}. The lift carriage's own
    motion and the sideways carriage's rise on its hanging blades as the blades bend (both arcs)."""
    import trimesh
    Lo = layout(p); Ly, Lz = Y_BLADES["L"], Z_BLADES["L"]
    wz = dz - p.z_neutral                                # the lift blades' bend from straight (as printed)
    uz = 0.6*wz**2/Lz                                    # the lift carriage pulled toward the post by their arc
    Y = dy - uz                                          # the sideways carriage
    V = 0.6*Y**2/Ly                                      # which rises as its blades swing
    lift = dz - V                                        # the lift carriage above rest, relative to the sideways carriage
    wz = lift - p.z_neutral; uz = 0.6*wz**2/Lz; Y = dy - uz; V = 0.6*Y**2/Ly; lift = dz - V; wz = lift - p.z_neutral
    def T(m, ty, tz): m = m.copy(); m.apply_translation([0.0, ty, tz]); return m
    def tm(man): q = man.to_mesh(); return trimesh.Trimesh(np.asarray(q.vert_properties)[:, :3], np.asarray(q.tri_verts))
    (cy0, cy1), (cz0, cz1) = Lo["carriage"]; fy0, fy1 = Lo["frame_y"]
    out = {}
    out["anchor"] = tm(_box(fy0, fy1, *Lo["anchor"]))
    frame = (_box(fy0, fy1, *Lo["bar_top"]) + _box(fy0, fy1, *Lo["bar_bot"]) + _box(*Lo["left_post"], Lo["bar_bot"][0], Lo["bar_top"][1])
             + _box(*Lo["post"], Lo["bar_bot"][0], Lo["bar_top"][1]) + _box(*Lo["down_stop"][0], *Lo["down_stop"][1]) + _box(*Lo["up_stop"][0], *Lo["up_stop"][1]))
    out["frame"] = T(tm(frame), Y, V)
    car = _box(cy0, cy1, cz0, cz1)
    import cad
    zc = Lo["zc"]; f = ARM_FIT
    car = car - cad.cube(STAGE_X[1] - 10.5, STAGE_X[1] + 1, -ARM[0]/2 - f, ARM[0]/2 + f, zc - ARM[1]/2 - f, zc + ARM[1]/2 + f)
    out["carriage"] = T(tm(car), Y + uz, V + lift)
    for k, yb in enumerate(Y_BLADES["y"]):              # hanging: fixed at the anchor bar, guided at the top bar
        out[f"yblade{k}"] = _blade((yb, Lo["anchor"][0]), (0.0, -1.0), (1.0, 0.0), Lo["anchor"][0] - Lo["bar_top"][1], Y_BLADES["t"], Y)
    for k, zb in enumerate(Lo["zblades"]):               # lift: fixed at the post, guided at the carriage
        bl = _blade((Lo["post"][0], zb + p.z_neutral), (-1.0, 0.0), (0.0, 1.0), Lz, Z_BLADES["t"], wz)
        out[f"zblade{k}"] = T(bl, Y, V)
    return out


def arm_offset(p):
    """The arm's world box at rest: x from the carriage's socket floor to the hood's rear wall."""
    zc = layout(p)["zc"]
    return (STAGE_X[1] - 10.0, p.dock_x + p.rest_gap + 0.5), (-ARM[0]/2, ARM[0]/2), (zc - ARM[1]/2, zc + ARM[1]/2)


def _slab(pg, x0, x1):
    """A shapely (y, z) polygon extruded along world x from x0 to x1."""
    import manifold3d as m3
    cs = m3.CrossSection([np.asarray(pg.exterior.coords)[:-1][::(1 if pg.exterior.is_ccw else -1)]])
    return m3.Manifold.extrude(cs, x1 - x0).translate((0, 0, x0)).transform([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])


FOOT_Z = 19.5                            # below this the egg is the board's 6 mm foot, in the base's socket


def egg_v4_cad(p, r=2.0, steps=16):
    """The egg: a shell 22 mm deep (front wall and side wall; the lid closes it), its front edge rounded as the
    board's, the window behind the hood, the tag pocket and the cable clip; below FOOT_Z the board's 6 mm foot."""
    import cad
    from shapely.geometry import box
    fw, cav, _ = EGG_T; xb = -(fw + cav); poly = ss.egg_outline(p)
    s = _slab(poly, xb, -r)
    for k in range(steps):                                       # the rounded front edge, in thin slabs
        x0 = -r + k*r/steps; xm = (k + 0.5)*r/steps; d = r - math.sqrt(max(r**2 - xm**2, 0.0))
        s = s + _slab(poly.buffer(-d, join_style=1), x0, x0 + r/steps)
    s = s - _slab(poly.buffer(-RIM, join_style=1).intersection(box(-200, FOOT_Z + 3.0, 200, 400)), xb - 1, -fw)
    s = s - cad.cube(xb - 5, -p.board_t, -200, 200, -10, FOOT_Z)   # behind the foot
    (wy0, wy1), (wz0, wz1) = layout(p)["window"]
    s = s - _slab(box(wy0, wz0, wy1, wz1).buffer(-3, join_style=2).buffer(3, join_style=1), -fw - 1, 1)
    tw = p.tag_size/2 + 2.0 + 0.15
    s = s - cad.cube(-1.6, 1, -tw, tw, ss.EGG_TAG_CZ - tw, ss.EGG_TAG_CZ + tw)
    (yk, zk), mouth = ss.egg_clip(p), cad.CABLE_D - 0.5; w = ss.egg_hw(zk)   # the cable clip, through the side wall
    s = s - cad.cyl_x(cad.CABLE_D/2 + 0.05, xb - 1, 1, yk, zk)
    s = s - cad.cube(xb - 1, 1, yk, w + 3, zk - mouth/2, zk + mouth/2)
    return ss.drop_slivers(s)


def lid_cad(p):
    """The back lid: the egg's outline above the foot, 3 mm, with the stage's mounts (standoffs under the anchor bar)
    and its side stops (blocks reaching through the cavity beside the sideways carriage's posts)."""
    from shapely.geometry import box
    fw, cav, lt = EGG_T; xb = -(fw + cav); Lo = layout(p)
    poly = ss.egg_outline(p).intersection(box(-200, FOOT_Z, 200, 400))
    s = _slab(poly, xb - lt, xb)
    fy0, fy1 = Lo["frame_y"]; a0, a1 = Lo["anchor"]
    for yy in (fy0 + 4.0, fy1 - 10.0):
        s = s + _box(yy, yy + 6.0, a0 + 1.0, a1 - 1.0, xb - 0.01, STAGE_X[0] + 0.01)          # standoffs
    zs = (Lo["bar_bot"][0] + 6.0, Lo["bar_top"][1] - 3.0)
    s = s + _box(*Lo["stops"]["left"], *zs, xb - 0.01, STAGE_X[1])
    s = s + _box(*Lo["stops"]["right"], zs[0] + 18.0, zs[1], xb - 0.01, STAGE_X[1])
    return s


def hood_v4_cad(p, head=True):
    """The strap dock's hood (its funnel, plug pocket, catch and cable slot) without the tooth and the straps' tabs and
    sockets, with the arm out of its rear wall's middle (through the egg's window into the lift carriage). head=False:
    without the right-angle cable head's channel, stop rib and cable slot (cad.hood)."""
    import cad
    top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    m = cad.hood(p, head=head) - cad.cube(p.hood_rear - 40, p.hood_rear, -80, 80, -20, 90)
    G = ss._funnel_geom(p); xa, _ = ss._lip_profile(p)
    for sy in (-1, 1):
        y0, y1 = sorted((sy*(p.side_in - 0.01), sy*80.0))
        m = m - cad.cube(G["A"][0] + 0.5, 60.0, y0, y1, p.side_bot - 2.0, top1 + 12.0)
        y0, y1 = sorted((sy*(p.skid_y - 0.01), sy*p.side_in))
        m = m - cad.cube(xa, p.neck_x + 2.0, y0, y1, p.trunk_top - 0.01, top1 + 8.0)
        y0, y1 = sorted((sy*p.side_in, sy*(yo + 0.01)))
        m = m - cad.cube(xa, p.neck_x + 2.0, y0, y1, top1 - 0.01, top1 + 8.0)
    m = m - cad.cube(p.hood_rear - 1, p.hood_rear + 3.01, -yo - 1, yo + 1, top1, 90)
    m = ss.drop_slivers(ss.round_top(m, top1, ss.HOOD_EDGE_R) + ss.funnel_cad(p))
    (x0, x1), (y0, y1), (z0, z1) = arm_offset(p); h = p.hood_home()
    return m + cad.cube(x0 - h[0], p.hood_rear + 1.0, y0, y1, z0 - h[2], z1 - h[2])     # the arm, in the hood's frame


def stage_print(p, reach=0.6):
    """The stage as it prints: one piece, flat on its back (16 mm tall), the hanging blades straight, the lift blades
    straight with the lift carriage half-lifted (z_neutral above its down-stop); every blade reaches `reach` into what it
    joins. Returns (manifold in print coordinates: x, y on the plate = world y, z; z up = world x from the stage's back),
    the world -> print 4x4."""
    import cad
    Lo = layout(p); zn = p.z_neutral
    (cy0, cy1), (cz0, cz1) = Lo["carriage"]; fy0, fy1 = Lo["frame_y"]; ty, tz = Y_BLADES["t"], Z_BLADES["t"]
    parts = [_box(fy0, fy1, *Lo["anchor"]), _box(fy0, fy1, *Lo["bar_top"]), _box(fy0, fy1, *Lo["bar_bot"]),
             _box(*Lo["left_post"], Lo["bar_bot"][0], Lo["bar_top"][1]), _box(*Lo["post"], Lo["bar_bot"][0], Lo["bar_top"][1]),
             _box(*Lo["down_stop"][0], *Lo["down_stop"][1]), _box(*Lo["up_stop"][0], *Lo["up_stop"][1])]
    for yb in Y_BLADES["y"]:
        parts.append(_box(yb - ty/2, yb + ty/2, Lo["bar_top"][1] - reach, Lo["anchor"][0] + reach))
    for zb in Lo["zblades"]:
        parts.append(_box(cy1 - reach, Lo["post"][0] + reach, zb + zn - tz/2, zb + zn + tz/2))
    zc = Lo["zc"] + zn; f = ARM_FIT
    car = _box(cy0, cy1, cz0 + zn, cz1 + zn) - cad.cube(STAGE_X[1] - 10.5, STAGE_X[1] + 1, -ARM[0]/2 - f, ARM[0]/2 + f,
                                                          zc - ARM[1]/2 - f, zc + ARM[1]/2 + f)
    m = cad.union(parts + [car])
    T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -STAGE_X[0]], [0, 0, 0, 1.0]])
    return m.transform(T[:3].tolist()), T

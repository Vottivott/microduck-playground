"""MicroDock v3: docking and undocking trials. --design flex is the egg dock: the hood on its printed spring stage
(source/flexdock.py: it floats sideways and up on springs, stiff toward the egg, and tilts as the stage lets it), the
magnetic pair and the stack's front stop as they are, and the cable's spring at its exit. The other designs are the
experiments on the way: the hood hanging on cords, rods, bars or flex straps (source/swing.py, strapswing.py), and the
MicroDock v2's sliding hood with the friction and cable spring the print has (slider).

Each trial docks as undocking_seated.py does (the scoot policies with the training's latencies), rests 2 s, scoots out
against the magnet, stands up and walks off. Logged per trial: docking and undocking as undocking_seated.py, how far
the hood floated, which stops it touched, and the closest the duck came to a hanger (all of its meshes, every second
control step).

  export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
  C="--k_y 67 --k_z 97 --z_neutral 5 --z_travel 15 --hood_mass 0.083 --x_solref 0.03"
  TILT="--k_rxyz 92.3,12.5,55.3 --rot_pivot=-12.5,0,136.5 --rot_solref 0.025,0.053,0.041"
  GB="--tip 3.2,11.2,6.8,1.65,11.85,7.6 --plug 8.11,12.91,14.27 --face_offset 0.80 --guide 11.8,7.4,0.25,0.1,1.35 --cable_out=-72.02,47.15,26.5"
  uv run python hardware/microdock-v3/sim/swing_trials.py --design flex $C $TILT $GB --seeds 41,42,43,44,45,46,47,48,49 --n 10 --out egg.json
  uv run python hardware/microdock-v3/sim/swing_trials.py --design flex $C $TILT $GB --cable_k 10 --cable_pre 0.4 --seeds 41,42,43,44,45,46,47,48,49 --n 10 --out stiff.json
"""
import argparse, json, math, os, sys
import numpy as np, mujoco, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "source"))
# MicroDock v3 builds on the v2 (seated docking) and v1 (dock, CAD, plates) modules beside it
_HW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("microdock-v2/source", "microdock-v2/sim", "microdock/source", "microdock/sim"):
    if os.path.join(_HW, _p) not in sys.path: sys.path.append(os.path.join(_HW, _p))
import docking_seated as ds
import undocking_seated as us
import latency
import swing, strapswing, flexdock
dg, dk = ds.dg, ds.dk

_build = dk.build_scene
dk.build_scene = lambda p, robot_dir, scene_path=None: (flexdock.build_scene(p, robot_dir, scene_path) if isinstance(p, flexdock.FlexDock)
                                                         else swing.build_scene(p, robot_dir, scene_path)
                                                         if isinstance(p, (swing.Params, swing.SliderFriction)) else _build(p, robot_dir, scene_path))
_modify_robot = dk.modify_robot
def _modify_robot_tip(spec, p):                   # the tip as it is (front part on a bigger base), when given
    spec = _modify_robot(spec, p)
    if isinstance(p, flexdock.FlexDock) and len(p.tip_front) == 3: flexdock.shape_tip(spec, p)
    return spec
dk.modify_robot = _modify_robot_tip
P = [None]
_magnet = dk.magnet
def _magnet_and_cable(model, data):
    r = _magnet(model, data)
    if P[0] is not None: swing.cable_force(model, data, P[0])
    return r
dk.magnet = _magnet_and_cable
TOUCH = {}


def touch_probe(model, data):
    """The tip against the adapter at its first touch (and 2 mm before): sideways (mm, adapter frame), tilt between the
    two faces' axes and roll about them (deg). The real adapter's rim needs these small (the simulator's own test,
    dk.connected, doesn't look at them)."""
    e = dk.tip_error(model, data)
    for key, gap in (("at2mm", 2.0), ("touch", 0.3)):
        if key in TOUCH or not (e[0] <= gap and abs(e[1]) < 4.0 and abs(e[2]) < 4.0): continue
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "adapter_face"); t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_face")
        Ra, Rt = data.site_xmat[a].reshape(3, 3), data.site_xmat[t].reshape(3, 3)
        xa, xt = Ra[:, 0], Rt[:, 0]
        tilt = math.degrees(math.acos(float(np.clip(abs(xa @ xt), -1.0, 1.0))))
        yp = Rt[:, 1] - (Rt[:, 1] @ xa)*xa
        roll = math.degrees(math.atan2(float(yp @ Ra[:, 2]), float(yp @ Ra[:, 1])))
        TOUCH[key] = dict(y=round(float(e[1]), 2), z=round(float(e[2]), 2), tilt=round(tilt, 2), roll=round(roll, 2))
_mc = dk.magnet
def _magnet_cable_touch(model, data):
    r = _mc(model, data); touch_probe(model, data); return r
dk.magnet = _magnet_cable_touch


def tip_kw(spec):
    """--tip out,front_w,front_h,front_len,base_w,base_h -> FlexDock's tip fields."""
    out, fw, fh, fl, bw, bh = (float(v) for v in spec.split(","))
    return dict(tip_out=out, tip_w=bw, tip_h=bh, tip_front=(fw, fh, fl))


def robot_vertices(model):
    """The robot's mesh geoms (everything but the dock and the hood)."""
    hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood"); out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_bodyid[g] in (0, hb): continue
        mid = model.geom_dataid[g]; a0 = model.mesh_vertadr[mid]; n = model.mesh_vertnum[mid]
        out.append((g, model.mesh_vert[a0:a0+n].copy()))
    return out


def clearance(model, data, verts, boxes):
    """The robot's closest approach to the hanger's boxes (mm; negative = inside one)."""
    best = 1e9
    for g, V in verts:
        W = (data.geom_xpos[g] + V @ data.geom_xmat[g].reshape(3, 3).T)*1000
        for lo, hi in boxes:
            near = np.all((W > lo - 30) & (W < hi + 30), axis=1)
            if not near.any(): continue
            Q = W[near]; d_out = np.linalg.norm(np.maximum(np.maximum(lo - Q, Q - hi), 0), axis=1)
            inside = np.all((Q > lo) & (Q < hi), axis=1)
            d = np.where(inside, -np.min(np.minimum(Q - lo, hi - Q), axis=1), d_out); best = min(best, float(d.min()))
    return best


def bar_edge_clearance(model, data, p, hb):
    """Bars: the closest any bar comes to the hood's top edge (the side walls' outer top corner at the bar's x), in the
    plane the bar swings in (mm; the bar's half width taken off). The simulator's bars are tendons that don't collide."""
    R = data.xmat[hb].reshape(3, 3); best = 1e9; top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    for name, ph, pw in p.cord_sites():
        a = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name + "_h")]*1000
        b = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name + "_w")]*1000
        c = data.xpos[hb]*1000 + R @ np.array([ph[0], np.sign(ph[1])*yo, top1])
        a2, b2, c2 = a[1:], b[1:], c[1:]; d = b2 - a2
        u = np.clip(np.dot(c2 - a2, d)/np.dot(d, d), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(c2 - (a2 + u*d))) - (p.bar_hw if isinstance(p, swing.Bars) else 1.5))
    return best


def _seg_dist(P, a, b):
    """Distances from points P (n, 3) to the segment a-b."""
    d = b - a; u = np.clip((P - a) @ d/np.dot(d, d), 0.0, 1.0)
    return np.linalg.norm(P - (a + u[:, None]*d), axis=1)


def rod_metrics(model, data, p, hb, verts):
    """Rods: where the printed rods really are. A rod hangs from its top ball through its tab's hole (the simulator's
    tendon rises with the hood; the real rod stays put while the tab slides up it, the bead hanging below the tab).
    Returns the rods' largest tilt in the hanger's seats and through the tabs (deg), their closest approach to the hood's
    top edges (mm, rod surface), and the duck's closest approach to a rod, bead or top ball (mm)."""
    R = data.xmat[hb].reshape(3, 3); o = data.xpos[hb]*1000; L = p.cord_len(); top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    seat = tab = 0.0; edge = robot = 1e9
    W = np.vstack([(data.geom_xpos[g] + V @ data.geom_xmat[g].reshape(3, 3).T)*1000 for g, V in verts])
    for name, ph, pw in p.cord_sites():
        A = np.asarray(pw, float); T = o + R @ (ph + np.array([0.0, 0.0, p.seat_dz]))
        u = (T - A)/np.linalg.norm(T - A); tip = A + (L + swing.BALL_D/2 + 0.5)*u; B = A + L*u
        seat = max(seat, math.degrees(math.acos(-u[2]))); tab = max(tab, math.degrees(math.acos(float(np.clip(-u @ R[:, 2], -1, 1)))))
        xs = ph[0] + np.arange(-12.0, 12.5, 1.0)
        E = o + (R @ np.stack([xs, np.full_like(xs, np.sign(ph[1])*yo), np.full_like(xs, top1)])).T
        edge = min(edge, float(_seg_dist(E, A, T).min()) - swing.ROD_D/2)
        lo, hi = np.minimum(A, tip) - 15, np.maximum(A, tip) + 15
        Q = W[np.all((W > lo) & (W < hi), axis=1)]
        if len(Q):
            robot = min(robot, float(_seg_dist(Q, A, tip).min()) - swing.ROD_D/2,
                        float(np.linalg.norm(Q - B, axis=1).min()) - swing.BALL_D/2, float(np.linalg.norm(Q - A, axis=1).min()) - swing.BALL_D/2)
    return seat, tab, edge, robot


def strap_metrics(model, data, p, hb, verts):
    """Straps: each strap as the box it is (15.2 x 5 mm) between its pivots, the bar's axis on the hanger and the
    knuckle's axis on the hood. Returns its largest sideways and front-to-back tilt (deg), the most it is pushed shorter
    (mm; past its travel it lifts its bar out of the groove), the closest it comes to the hood's top edges and the duck's
    closest approach to it (mm, outside its faces). A slack strap is taken as straight: lifted off-centre it bows ~10 mm."""
    R = data.xmat[hb].reshape(3, 3); o = data.xpos[hb]*1000; s = p.strap(); L = s.length_open()
    hw, ht = s.width()/2, s.T/2; top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    W = np.vstack([(data.geom_xpos[g] + V @ data.geom_xmat[g].reshape(3, 3).T)*1000 for g, V in verts])
    side = front = short = 0.0; edge = robot = 1e9
    for name, ph, pw in p.strap_sites():
        A = np.asarray(pw, float); B = o + R @ ph; d = B - A; n = float(np.linalg.norm(d)); u = d/n
        side = max(side, math.degrees(math.atan2(abs(d[1]), -d[2]))); front = max(front, math.degrees(math.atan2(abs(d[0]), -d[2])))
        short = max(short, L - n)
        ew = np.array([1.0, 0.0, 0.0]) - u*u[0]; ew /= np.linalg.norm(ew); et = np.cross(u, ew)
        def box_dist(Q):
            q = Q - A; t = q @ u; r = q - np.outer(t, u)
            dw = np.abs(r @ ew) - hw; dt = np.abs(r @ et) - ht; du = np.maximum(np.maximum(-t, t - n), 0.0)
            outside = np.sqrt(np.maximum(dw, 0)**2 + np.maximum(dt, 0)**2 + du**2)
            return np.where((dw < 0) & (dt < 0) & (du == 0), np.maximum(dw, dt), outside)
        xs = ph[0] + np.arange(-12.0, 12.5, 1.0)
        E = o + (R @ np.stack([xs, np.full_like(xs, np.sign(ph[1])*yo), np.full_like(xs, top1)])).T
        edge = min(edge, float(box_dist(E).min()))
        lo, hi = np.minimum(A, B) - 20, np.maximum(A, B) + 20
        Q = W[np.all((W > lo) & (W < hi), axis=1)]
        if len(Q): robot = min(robot, float(box_dist(Q).min()))
    return side, front, short, edge, robot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", choices=["swing", "rods", "straps", "bars", "slider", "flex"], default="swing")
    ap.add_argument("--k_y", type=float, default=40.0, help="flex: the stage's sideways spring (N/m)")
    ap.add_argument("--k_z", type=float, default=40.0, help="flex: the stage's lift spring (N/m)")
    ap.add_argument("--k_x", type=float, default=3000.0, help="flex: the stage toward the board / out (N/m)")
    ap.add_argument("--z_travel", type=float, default=14.0, help="flex: the up-stop above rest (mm)")
    ap.add_argument("--damping", type=float, default=0.03, help="flex: the stage's damping ratio")
    ap.add_argument("--k_rot", type=float, default=0.0, help="flex: the hood's tilt/turn springs (N m/rad; 0: it can't turn)")
    ap.add_argument("--hood_mass", type=float, default=0.065, help="flex: what moves with the hood (kg): the hood, its plug and the stage's carriages")
    ap.add_argument("--k_rxyz", default="", help="flex: the hood's roll,pitch,yaw springs (N m/rad, e.g. 92,12.5,55): it tilts as the stage lets it")
    ap.add_argument("--rot_pivot", default="", help="flex: world mm the hood turns about (the stage), e.g. =-12.5,0,120")
    ap.add_argument("--rot_solref", default="", help="flex: the tilt springs as soft equalities: roll,pitch,yaw time constants (s)")
    ap.add_argument("--guide", default="", help="flex: the adapter's lead-in: narrow w,h, gap to the face, narrow depth, flare (mm), e.g. 12.3,6.5,0.9,0.4,1.4")
    ap.add_argument("--tip", default="", help="flex: the magnetic tip as it is: out of the shell, front part w,h,length, base w,h (mm), e.g. 3.2,11.2,6.8,1.65,11.85,7.6")
    ap.add_argument("--face_offset", type=float, default=None, help="flex: the adapter's face from the docked tip's face (mm, +: toward the duck; default -0.1)")
    ap.add_argument("--plug_len", type=float, default=0.0, help="flex: the adapter's length (mm; 0: dock.Params' 14)")
    ap.add_argument("--plug", default="", help="flex: the adapter's height, width, length (mm), e.g. 8.11,12.91,14.27 (overrides --plug_len)")
    ap.add_argument("--cable_out", default="", help="flex: where the cable leaves the hood's parts (hood frame mm), e.g. -72.02,47.15,26.5")
    ap.add_argument("--x_travel", default="", help="flex: the hood's play toward the board and out (mm), e.g. 0.5,0.5")
    ap.add_argument("--x_solref", type=float, default=0.0, help="flex: toward the board / out as a stiff spring (equality time constant, s)")
    ap.add_argument("--z_neutral", type=float, default=0.0, help="flex: where the lift springs are straight, above rest (mm)")
    ap.add_argument("--rest_drop", type=float, default=0.0, help="flex: the hood rests this far below docked (mm; 0: the strap dock's)")
    ap.add_argument("--friction", type=float, default=0.5, help="slider: Coulomb friction on each slide (N)")
    ap.add_argument("--start_y", type=float, default=15.0, help="slider: the hood starts up to this far off-centre (mm), where the last undocking left it")
    ap.add_argument("--cable_k", type=float, default=3.0, help="the cable's spring at its exit (N/m)")
    ap.add_argument("--cable_pre", type=float, default=0.1, help="the cable's preload (N), in a random direction across the board")
    ap.add_argument("--out_mm", type=float, default=70.0, help="how far out the duck scoots before it stands up (mm)")
    ap.add_argument("--mag", type=float, default=4.0); ap.add_argument("--seeds", default="41"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=""); ap.add_argument("--save", default="", help="pickle the first trial's qpos at 50 Hz (docking and undocking)")
    ap.add_argument("--save_t", type=int, default=0, help="--save records this trial of the first seed instead")
    ap.add_argument("--chains", action="store_true", help="straps: each strap as its chain of joints with their limits (else a tendon)")
    ap.add_argument("--chain_scale", type=float, default=1.0, help="straps --chains: the joints' ranges times this (all but bending through the thickness)")
    ap.add_argument("--rod_x", default="", help="rods: the rear and front pairs' world x at rest (mm), e.g. 9,46")
    ap.add_argument("--rod_y", type=float, default=0.0, help="rods: the rods' |y| (mm; the hanger plate widens with them)")
    ap.add_argument("--tab_z", type=float, default=0.0, help="rods: the tabs' height on the hood (hood frame mm)")
    ap.add_argument("--trace", default="", help="write each control step's hood-duck contacts and hood pose (JSON lines)")
    a = ap.parse_args()
    dk.MAG_F = a.mag; res = []; TR = []; REC = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        p = (swing.Params(cable_k=a.cable_k) if a.design == "swing" else swing.Bars(cable_k=a.cable_k) if a.design == "bars"
             else swing.Rods(cable_k=a.cable_k) if a.design == "rods" else strapswing.Straps(cable_k=a.cable_k) if a.design == "straps"
             else flexdock.FlexDock(cable_k=a.cable_k, k_y=a.k_y, k_z=a.k_z, k_x=a.k_x, z_travel=a.z_travel, damping_ratio=a.damping, k_rot=a.k_rot, x_solref=a.x_solref, z_neutral=a.z_neutral, hood_mass=a.hood_mass,
                                    **(dict(zip(("k_rx", "k_ry", "k_rz"), (float(v) for v in a.k_rxyz.split(",")))) if a.k_rxyz else {}),
                                    **({"rot_pivot": tuple(float(v) for v in a.rot_pivot.split(","))} if a.rot_pivot else {}),
                                    **({"rot_solref": tuple(float(v) for v in a.rot_solref.split(","))} if a.rot_solref else {}),
                                    **({"guide": tuple(float(v) for v in a.guide.split(","))} if a.guide else {}),
                                    **({"x_travel": tuple(float(v) for v in a.x_travel.split(","))} if a.x_travel else {}),
                                    **(tip_kw(a.tip) if a.tip else {}),
                                    **({"face_offset": a.face_offset} if a.face_offset is not None else {}),
                                    **({"plug": (8.0, 12.0, a.plug_len)} if a.plug_len else {}),
                                    **({"plug": tuple(float(v) for v in a.plug.split(","))} if a.plug else {}),
                                    **({"cable_out": tuple(float(v) for v in a.cable_out.split(","))} if a.cable_out else {}),
                                    **({"rest_drop": a.rest_drop} if a.rest_drop else {})) if a.design == "flex"
             else swing.SliderFriction(friction_n=a.friction, cable_k=a.cable_k))
        if a.design == "straps" and a.chains:
            p.strap_chains = True
            for k in ("chain_in", "chain_twist", "chain_top_tilt", "chain_knuckle_tilt"): setattr(p, k, getattr(p, k)*a.chain_scale)
        if a.design == "rods":
            if a.rod_x: p.rod_xw = tuple(float(v) for v in a.rod_x.split(","))
            if a.rod_y: p.rod_y = a.rod_y; p.bar_half_w = a.rod_y + 9.0
            if a.tab_z: p.tab_z = a.tab_z
        P[0] = p
        model, data, bc, pol = ds.make(p, seat_mu=us.DOCK.seat_mu)
        rng = np.random.default_rng(seed); rng_h = np.random.default_rng(seed + 1000)
        J = {n: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("hood_y", "hood_z")}
        start = {}
        _reset = dg.reset
        def reset(*args, **kw):                      # the hood starts where the last undocking left it (slider) or hanging (swing)
            _reset(*args, **kw)
            if a.design == "slider":
                data.qpos[J["hood_y"]] = start["y"]/1e3; data.qpos[J["hood_z"]] = -p.hood_drop/1e3; mujoco.mj_forward(model, data)
        dg.reset = reset
        latency.enable(pol, bc, seed=seed, official=False, scoot=True)
        sess = ort.InferenceSession(a.undock if hasattr(a, "undock") else us.UNDOCK); pol.undock_session = sess
        verts = robot_vertices(model)
        if a.design == "swing": boxes = [(np.min(B, 0), np.max(B, 0)) for B in swing.hanger_boxes(p).values()]
        elif a.design in ("rods", "straps"):      # the hanger (the rods and straps themselves: rod_metrics, strap_metrics)
            boxes = [(np.min(B, 0), np.max(B, 0)) for B in swing.hanger_boxes(p).values()]
        elif a.design == "bars":                  # the hanger, its clevises, and the space each bar sweeps as the hood swings
            boxes = [(np.min(B, 0), np.max(B, 0)) for B in swing.bars_hanger_boxes(p).values()]
            for name, ph, pw in p.cord_sites():
                sy = np.sign(pw[1]); boxes.append((np.array([pw[0] - 2.5, min(sy*36.0, sy*60.0), 66.0]), np.array([pw[0] + 2.5, max(sy*36.0, sy*60.0), p.bar_z[0]])))
        else: boxes = []
        gid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("dock_board", "hood_rear")}
        stop_g = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("dock_bar", "dock_bar_l", "dock_bar_r")} - {-1}
        hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
        stats = {}
        CJ = {}                                   # strap chains: in-plane, twist, top-tilt joints (qpos address, limit)
        if a.design == "straps" and a.chains:
            for j in range(model.njnt):
                nm = model.joint(j).name
                if nm.startswith("sc_"):
                    kind = nm.split("_")[-1]
                    key = "in" if kind[0] == "i" else "tw" if kind[0] == "w" else "lift" if kind == "lift" else "top_y" if kind == "ty" else None
                    if key: CJ.setdefault(key, []).append((model.jnt_qposadr[j], model.jnt_range[j][1]))
        _step = dg.step_control
        def step(model_, data_, bc_, pol_, frame_cb=None):
            _step(model_, data_, bc_, pol_, frame_cb)
            stats["k"] += 1
            if stats.get("rec"): REC.append(data_.qpos.copy())
            if a.trace:
                pairs = {}
                f6 = np.zeros(6)
                for i in range(data_.ncon):
                    c = data_.contact[i]; b1, b2 = model_.geom_bodyid[c.geom1], model_.geom_bodyid[c.geom2]
                    if (b1 == hb) == (b2 == hb) or 0 in (b1, b2): continue
                    mujoco.mj_contactForce(model_, data_, i, f6)
                    key = model_.geom(c.geom1).name + "|" + model_.geom(c.geom2).name
                    pairs[key] = pairs.get(key, 0.0) + abs(f6[0])
                q = data_.qpos; jx = model_.jnt_qposadr[mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_JOINT, "hood_x")] if a.design in ("swing", "rods", "straps") or (a.design == "flex" and (a.k_x > 0 or a.x_solref > 0)) else None
                TR.append(dict(trial=stats["trial"], k=stats["k"], hood_duck_N=round(sum(pairs.values()), 2),
                               top=sorted(pairs.items(), key=lambda kv: -kv[1])[:3], hood_x=None if jx is None else round(q[jx]*1e3, 2),
                               hood_z=round(q[J["hood_z"]]*1e3, 2), hood_y=round(q[J["hood_y"]]*1e3, 2),
                               trunk_x=round(q[0]*1e3, 1), lip=any({c.geom1, c.geom2} & stop_g for c in data_.contact[:data_.ncon])))
            if stats["k"] % 2 == 0 and a.design == "bars":
                stats["edge"] = min(stats["edge"], bar_edge_clearance(model_, data_, p, hb))
            if CJ:
                q = data_.qpos
                fr_in = max(abs(q[ad])/lim for ad, lim in CJ["in"]); fr_tw = max(abs(q[ad])/lim for ad, lim in CJ["tw"])
                stats["in_max"] = max(stats.get("in_max", 0.0), fr_in); stats["tw_max"] = max(stats.get("tw_max", 0.0), fr_tw)
                stats["in_lim"] = stats.get("in_lim", 0) + (fr_in > 0.95); stats["n"] = stats.get("n", 0) + 1
                stats["lift"] = max(stats.get("lift", 0.0), max(q[ad] for ad, _ in CJ["lift"])*1e3)
                stats["topy"] = max(stats.get("topy", 0.0), max(abs(q[ad])/lim for ad, lim in CJ["top_y"]))
            if stats["k"] % 2 == 0 and a.design == "straps" and not CJ:
                sd, fr, sh, ed, rb = strap_metrics(model_, data_, p, hb, verts)
                stats.update(seat=max(stats["seat"], sd), tab=max(stats["tab"], fr), short=max(stats["short"], sh), edge=min(stats["edge"], ed), rodclr=min(stats["rodclr"], rb))
            if stats["k"] % 2 == 0 and a.design == "rods":
                sd, tb, ed, rb = rod_metrics(model_, data_, p, hb, verts)
                stats.update(seat=max(stats["seat"], sd), tab=max(stats["tab"], tb), edge=min(stats["edge"], ed), rodclr=min(stats["rodclr"], rb))
            if stats["k"] % 2 == 0:
                if boxes: stats["clear"] = min(stats["clear"], clearance(model_, data_, verts, boxes))
                stats["hy"] = max(stats["hy"], abs(data_.qpos[J["hood_y"]])*1e3); stats["hz"] = max(stats["hz"], data_.qpos[J["hood_z"]]*1e3)
                if a.design == "flex" and (a.k_x > 0 or a.x_solref > 0):
                    hx = data_.qpos[model_.jnt_qposadr[mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_JOINT, "hood_x")]]*1e3
                    stats["hx0"] = min(stats.get("hx0", 0.0), hx); stats["hx1"] = max(stats.get("hx1", 0.0), hx)
                for c in data_.contact[:data_.ncon]:
                    pair = {c.geom1, c.geom2}
                    if pair & stop_g and any(model_.geom_bodyid[g] == hb for g in pair - stop_g): stats["lip"] += 1
                    if gid["dock_board"] in pair and model_.geom_bodyid[(pair - {gid["dock_board"]}).pop()] == hb: stats["board"] += 1
        dg.step_control = step
        for t in range(a.n):
            x0, y0, yaw0 = rng.uniform(380, 460), rng.uniform(-40, 40), 180 + rng.uniform(-12, 12)
            nx, ny, nyaw = rng.normal(0, 5), rng.normal(0, 3), rng.normal(0, 1)
            th = rng_h.uniform(0, 2*np.pi); p.cable_pre = (0.0, a.cable_pre*math.cos(th), a.cable_pre*math.sin(th))
            start["y"] = rng_h.uniform(-a.start_y, a.start_y)
            stats.pop("hx0", None); stats.pop("hx1", None)
            stats.update(k=0, clear=1e9, hy=0.0, hz=-1e9, lip=0, board=0, edge=1e9, seat=0.0, tab=0.0, rodclr=1e9, short=0.0, in_max=0.0, tw_max=0.0, in_lim=0, n=0, lift=0.0, topy=0.0, trial=f"{seed}-{t}", rec=bool(a.save and t == a.save_t and seed == int(a.seeds.split(",")[0])))
            TOUCH.clear()
            r = ds.run(model, data, bc, pol, p, us.DOCK, x0, y0, yaw0, nx, ny, nyaw)
            touch = dict(TOUCH)
            dock_stats = dict(stats)
            flex_docked = dict(flex_docked_lift_mm=round(float(data.qpos[J["hood_z"]]*1e3), 2), flex_docked_y_mm=round(float(data.qpos[J["hood_y"]]*1e3), 2)) \
                if a.design == "flex" and r["ok"] else {}
            u = us.undock(model, data, bc, pol, sess, out_mm=a.out_mm) if r["ok"] else {}
            row = dict(seed=seed, t=t, docked=bool(r["ok"]), usb_drops=r["usb_drops"], err=r["err"], after_sit=r["after_sit"], touch=touch,
                       start=[round(x0), round(y0), round(yaw0, 1)], cable_pre=[round(v, 3) for v in p.cable_pre],
                       hood_start_y=round(start["y"], 1) if a.design == "slider" else 0.0,
                       dock_hood_max_y=round(dock_stats["hy"], 1), dock_stop_contacts=dock_stats["lip"], dock_board_contacts=dock_stats["board"],
                       hood_max_y=round(stats["hy"], 1), stop_contacts=stats["lip"], board_contacts=stats["board"],
                       hanger_clearance_mm=None if not boxes else round(stats["clear"], 1), hood_max_lift_mm=round(stats["hz"], 1),
                       bar_edge_clearance_mm=round(stats["edge"], 1) if a.design in ("bars", "rods", "straps") else None,
                       **(dict(strap_side_tilt_deg=round(stats["seat"], 1), strap_front_tilt_deg=round(stats["tab"], 1),
                               strap_max_shortening_mm=round(stats["short"], 1), strap_robot_clearance_mm=round(stats["rodclr"], 1))
                          if a.design == "straps" and not CJ else {}),
                       **(dict(chain_in_plane_max=round(stats["in_max"], 2), chain_in_plane_at_limit_s=round(stats["in_lim"]/50, 2),
                               chain_twist_max=round(stats["tw_max"], 2), chain_top_tilt_max=round(stats["topy"], 2), chain_bar_lift_mm=round(stats["lift"], 1))
                          if CJ else {}),
                       **(dict(rod_seat_tilt_deg=round(stats["seat"], 1), rod_tab_tilt_deg=round(stats["tab"], 1), rod_robot_clearance_mm=round(stats["rodclr"], 1))
                          if a.design == "rods" else {}),
                       **(dict(flex_x_range_mm=[round(stats.get("hx0", 0.0), 2), round(stats.get("hx1", 0.0), 2)], **flex_docked) if a.design == "flex" else {}), **u)
            res.append(row); print(json.dumps(row, default=float), flush=True)
            if stats["rec"]:
                import pickle; pickle.dump(dict(qpos=np.array(REC), p=p.__dict__, design=a.design), open(a.save, "wb")); stats["rec"] = False
        dg.reset, dg.step_control = _reset, _step
    ok = [r for r in res if r["docked"]]
    summ = dict(design=a.design, out_mm=a.out_mm, friction_N=a.friction if a.design == "slider" else 0.0, cable_k=a.cable_k, cable_pre_N=a.cable_pre, mag_N=a.mag,
                runs=len(res), docked=len(ok), usb_drops=sum(r["usb_drops"] > 0 for r in ok), released=sum(r.get("release_s") is not None for r in ok),
                stood=sum(bool(r.get("stood")) for r in ok), walked=sum(bool(r.get("walked")) for r in ok),
                release_s_max=max((r["release_s"] for r in ok if r.get("release_s") is not None), default=None),
                hood_max_y=max((r["hood_max_y"] for r in res), default=None),
                min_hanger_clearance_mm=min((r["hanger_clearance_mm"] for r in res if r["hanger_clearance_mm"] is not None), default=None))
    print("summary", json.dumps(summ, default=float), flush=True)
    if a.trace:
        with open(a.trace, "w") as f:
            for row in TR: f.write(json.dumps(row) + "\n")
    if a.out: json.dump(dict(cmd=" ".join(["swing_trials.py"] + sys.argv[1:]), summary=summ, results=res), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

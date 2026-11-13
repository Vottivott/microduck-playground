# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "manifold3d>=3.2",
#   "numpy>=2.0",
#   "pillow>=10",
#   "scipy>=1.10",
#   "shapely>=2.0",
#   "trimesh>=4.8",
# ]
# ///
"""MicroDock v3, the egg dock (flexdock_c.py): its parts, its print set on seven P1S plates, and the checks.
  1  the base with its rounded corners, as it sits (the egg's foot stands in its socket)
  2  the egg's front plate, face down (the tag pocket's floor on supports from the plate; nothing else is supported)
  3  the dome, standing on its floor (its outside all walls; brim)
  4  the spring stage, flat, half-lifted (Arachne walls for the 0.4 mm blades, no supports)
  5  the hood with its arm, upside down (supports from the plate)
  6  the tag: black base, then the white cells
  7  (optional) the hood's slice round the magnetic adapter, printed as the hood: to try the adapter, the angle
     adapter and the duck's tip in first (fit_test_c)
Fully printed: no screws or glue (the dome snaps on, the hood's arm sits in the carriage on crush ribs, locked by a
1.75 mm filament pin; the post sits on pegs, pinned through the front plate's ears by a second pin; the tag presses in).

  uv run hardware/microdock-v3/source/flexdock_c_build.py [--out DIR] [--check] [--bambu /path/to/bambu-studio]

writes stl/ (the parts as designed) and print/ (each plate's STL and top view, and with --bambu the Bambu Studio
project, built with Bambu Studio's command-line slicer as ../../microdock/source/bambu_project.py: P1S 0.4 nozzle,
0.20 mm Standard, Bambu PLA Basic white and black, textured PEI plate). --check fails unless every part is one
watertight body (the tag: its base and the white cells), the parts fit together as assembled, the stage clears the
dome and the egg's wall over its travel, the post's pin has its clear run, the lead-in and the adapter's stops are
where the duck's tip needs them, the angle adapter and the cable's plug fit the hood and its clips, and (with --bambu)
every plate slices without a warning.
"""
import argparse, dataclasses, json, os, shutil, sys, tempfile
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "microdock", "source"))
# MicroDock v3 builds on the v2 (seated docking) and v1 (dock, CAD, plates) modules beside it
_HW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("microdock-v2/source", "microdock-v2/sim", "microdock/source", "microdock/sim"):
    if os.path.join(_HW, _p) not in sys.path: sys.path.append(os.path.join(_HW, _p))
import flexdock as fd, flexdock_c as fc, strapdock_build as sb, plates, strapswing as ss, cad

V4 = dict(k_y=67.0, k_z=97.0, z_neutral=5.0, z_travel=15.0, x_solref=0.02, hood_mass=0.083)
# z_neutral 5 (v4: 3.5): the lift here also carries the hanger and the hanging blades (83 g against v4's 68): printed
# 1.5 mm higher, the springs take v4's share again (0.33 N on the stop, 1.12 N on the duck docked); hood_mass: what
# the lift carries here (the hood, its plug, the carriage, crossbar, hanging blades and hanger)
OUT = os.path.dirname(HERE)
TAG = "microdock_v3"
PLATES = {1: ("1 Base", "base"), 2: ("2 Front plate", "front"), 3: ("3 Dome", "dome"), 4: ("4 Spring stage", "stage"),
          5: ("5 Hood", "hood"), 6: ("6 Tag (black, then white)", "tag"), 7: ("7 Fit test (optional)", "test")}
ORIENT = {"base": Rot.identity(),                                # as it sits
          "tag": Rot.from_euler("y", -90, degrees=True),        # face up: black base, white cells on top
          "front": Rot.from_euler("y", 90, degrees=True),       # face down (world +x -> down)
          "dome": Rot.identity(),                               # standing on its floor
          "stage": Rot.identity(),                              # stage_c_print's plate frame
          "hood": Rot.from_euler("x", 180, degrees=True),       # upside down
          "test": Rot.from_euler("x", 180, degrees=True)}
PARAMS = {"base": {}, "tag": {}, "front": dict(sb.SUPPORT),
          "dome": {"enable_support": "0", "brim_type": "outer_only", "brim_width": "8", "layer_height": "0.16"},
          "stage": {"wall_generator": "arachne", "enable_support": "0"},
          "hood": dict(sb.SUPPORT), "test": dict(sb.SUPPORT)}
LABELS = {"base": "Base", "front": "Front plate", "dome": "Dome", "stage": "Spring stage", "hood": "Hood", "test": "Hood fit test",
          "front_blocker": "edge support blocker"}


def tm(man):
    """manifold -> trimesh, without the stray zero-volume fragments booleans can leave (separate bodies of a few
    triangles). Sliver triangles inside the solid stay: deleting them would open it."""
    q = man.to_mesh(); m = trimesh.Trimesh(np.asarray(q.vert_properties)[:, :3], np.asarray(q.tri_verts))
    parts = m.split(only_watertight=False)
    if len(parts) > 1: m = trimesh.util.concatenate([b for b in parts if len(b.faces) > 20 or abs(b.volume) > 1.0])
    return m


def posed_check(p, inner):
    """The stage posed over the travel's corners and middle: its points to the dome's inside (the eroded wall, CLEAR
    kept) and to the egg's side wall (RIM, CLEAR). Returns the smallest margins (mm)."""
    from shapely.geometry import Point
    from shapely.prepared import prep
    wall = prep(ss.egg_outline(p).buffer(-(fd.RIM + fc.CLEAR)))
    worst_back, worst_side = 1e9, 1e9
    for dy in (-p.y_travel, 0.0, p.y_travel):
        for dz in (0.0, p.z_travel/2, p.z_travel):
            V = np.vstack([m.vertices for k, m in fc.stage_c_parts(p, dy, dz).items() if k != "post"])
            back = V[V[:, 0] < fd.STAGE_X[0] + 0.01]
            worst_back = min(worst_back, float(np.min(inner(back[:, 1], back[:, 2]) - fc.CLEAR + back[:, 0])))
            sub = V[::7]
            worst_side = min(worst_side, min(0.0 if wall.contains(Point(y, z)) else -1.0 for _, y, z in sub))
    return dict(back_to_dome=round(worst_back, 2), inside_side_wall=worst_side == 0.0)


def dome_post_check(p):
    """The post's pin and the egg's lip: the stage's moving parts against the front plate's ears over the travel's
    corners (min gap, mm; the post itself excluded), the straight run along the pin's line from the middle of the egg
    to the -y ear (the filament goes in from there; mm of it clear of the stage), and the first C domes (clamp 0.3,
    the old cable channel) on the new front plate (overlap mm3)."""
    import dataclasses, cad, manifold3d as m3
    P = fc.POST_PIN; Lo = fc.layout_c(p); (py0, py1), _ = Lo["post"]; g, t = P["ear_gap"], P["ear_t"]
    ears = [((P["ear_x"], -fc.FW), (py0 - g - t, py0 - g), P["ear_z"]), ((P["ear_x"], -fc.FW), (py1 + g, py1 + g + t), P["ear_z"])]
    def box_dist(V, b):
        lo = np.array([b[0][0], b[1][0], b[2][0]]); hi = np.array([b[0][1], b[1][1], b[2][1]])
        return np.linalg.norm(np.maximum(np.maximum(lo - V, V - hi), 0), axis=1)
    worst = 1e9
    for dy in (-p.y_travel, 0.0, p.y_travel):
        for dz in (0.0, p.z_travel/2, p.z_travel):
            for k, m in fc.stage_c_parts(p, dy, dz).items():
                if k == "post": continue
                for b in ears: worst = min(worst, float(box_dist(np.asarray(m.vertices), b).min()))
    # the pin's line, from y -15 to the -y ear: clear of every stage part at rest (sampled every 0.25 mm, 1.2 mm round it)
    line = np.array([[P["x"], y, P["z"]] for y in np.arange(-15.0, py0 - g - t, 0.25)])
    parts0 = fc.stage_c_parts(p, 0.0, 0.0); free = None
    for k, m in parts0.items():
        if k == "post": continue
        man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
        for y in np.arange(-15.0, py0 - g - t, 0.25):
            probe = m3.Manifold.cylinder(0.25, 1.2, 1.2, 16).rotate((-90, 0, 0)).translate((P["x"], y, P["z"]))
            if (probe ^ man).volume() > 1e-6: free = y if free is None else max(free, y)
    run = (py0 - g - t) - (free if free is not None else -15.0)
    front = fc.front_plate_c(p); old = fc.CLAMP_GAP, fc.CABLE
    try:
        fc.CLAMP_GAP, fc.CABLE = 0.3, dict(d=4.0 - 0.1, mouth=3.5)        # the first C domes: clamp 0.3, hole 4.1, mouth 3.5
        dome_old = fc.dome_c(p)
    finally:
        fc.CLAMP_GAP, fc.CABLE = old
    return dict(ears_to_stage_mm=round(worst, 2), pin_run_mm=round(run, 1), old_dome_on_new_front_mm3=round(float((dome_old ^ front).volume()), 3))


def assembly_check(p, parts):
    """Overlaps (mm3) of the parts as assembled, at rest: dome-front plate (the snaps' barbs in their notches), dome
    and front plate against the stage, the hood's arm in the carriage (its barb in the window); and the snaps'
    numbers (engagement, the fingers' strain as they pass the catch)."""
    import manifold3d as m3
    def man(t_): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t_.vertices, np.float32), tri_verts=np.asarray(t_.faces, np.uint32)))
    stage_parts = fc.stage_c_parts(p, 0.0, 0.0)
    uz = 0.6*p.z_neutral**2/fd.Z_BLADES["L"]                        # at rest the bent lift blades pull it all this far
    hood = parts["hood"].copy(); hood.apply_translation(p.hood_home() + np.array([0.0, uz, 0.0]))   # toward the post
    F, Dm, Hd = man(parts["front"]), man(parts["dome"]), man(hood)
    St = m3.Manifold.batch_boolean([man(v) for k, v in stage_parts.items() if "blade" not in k], m3.OpType.Add)
    # (the 0.4 mm blades' meshes don't pass as manifolds; fit() and posed_check() cover their clearances)
    car = man(stage_parts["carriage"])
    for nm, mm in (("front", F), ("dome", Dm), ("stage", St), ("hood", Hd), ("carriage", car)):   # a broken mesh is empty
        assert mm.volume() > 1.0, f"{nm}: not a closed solid (the overlap checks would pass falsely)"
    ov = lambda a, b: round(float((a ^ b).volume()), 3)
    S = fc.SNAP; inner = fc._inner_depth(); strain = []
    for ys, zs, side in fc.SNAPS:
        L = (-fc.FW - 0.4) - (-float(inner([ys], [zs])[0]))           # the finger's free length
        strain.append(round(150*S["t"]*(S["barb"] - S["gap"])/L**2, 2))
    J = fc.ARM_JOINT; inter = J["rib_h"] - J["fit"]                  # the crush ribs' interference with the arm
    base = J["rib_w"]*inter/J["rib_h"]; ribs_mm3 = 8*0.5*base*inter*((fd.STAGE_X[1] - J["lead"] - 0.3) - (J["floor"] + 0.5))
    return dict(dome_front=ov(Dm, F), dome_stage=ov(Dm, St), front_stage=ov(F, St), hood_carriage=ov(Hd, car),
                hood_carriage_ribs_expected=round(ribs_mm3, 2), snap_engagement_mm=round(S["barb"] - S["gap"], 2),
                finger_strain_pct=strain, rib_interference_mm=round(inter, 2), arm_depth_mm=round(fd.STAGE_X[1] - J["floor"], 1))


def head_fit_c(p, length=None, hood_full=None):
    """cad.head_fit for the egg-C hood: the cable head and the adapter (cad.adapter_solid) at rest (on the stop rib, the
    spring's leaf left out: it presses the head by design), rising in from below past the catch's bent fingers, how far
    they drop before the catch holds them, how far forward the rib lets them go and how far back the duck can push them
    before the hard stop. `length`: the elbow + adapter (default PLUG_C's 2.7 + 13.4); the adapter's length follows."""
    import dataclasses, cad, manifold3d as m3
    p = dataclasses.replace(p, **fc.PLUG_C); pc = dataclasses.replace(p, face_offset=p.face_offset - 0.15)
    leaf = fc._spring(p); hood = (hood_full or fc.hood_c_cad(p)) - leaf
    if length is not None:                                     # the same rib, an adapter of length - elbow
        ed = p.angle_elbow[0]; rib = fc.face_window(p)["rib"]
        p = dataclasses.replace(p, plug=(p.plug[0], p.plug[1], length - ed), face_offset=rib + length - p.tip_face_x)
    _, _, fingers = cad.catch(pc); _, _, bent = cad.catch(pc, flex=cad.CATCH_HOOK - 0.2 + 0.05); flexed = hood - fingers + bent
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug; ed, el, xe = p.angle_elbow
    x0 = p.adapter_face_x - pl; r = at/2; xa = x0 - (aw - r)
    def cyl_y(r0, r1, y0, y1, x, z): return m3.Manifold.cylinder(y1 - y0, r0, r1, 48).rotate((-90, 0, 0)).translate((x, y0, z))
    def pair(dx=0.0, dz=0.0, cable=True):
        z = p.adapter_z + dz
        h = cad.hull(np.array([[x0 + dx + xx, y, z + zz] for xx, y in ((0, -xe), (0, el - xe), (-(ed + 2.1), el - xe + 1.8), (-(ed + 2.1), -xe)) for zz in (-r, r)]))
        h = h + cyl_y(r, r, -xe, al - 8.8 - xe, xa + dx, z) + cyl_y(r, 2.9, al - 8.8 - xe, al - xe, xa + dx, z)
        if cable: h = h + cyl_y(2.0, 2.0, al - xe, p.side_in + p.wall_t + 3.0, xa + dx, z)
        return h + cad.adapter_solid(p).translate((dx, 0, dz))
    v = lambda a, b: float((a ^ b).volume())
    first = lambda xs, f: next((round(float(x), 2) for x in xs if f(x) > 0.01), None)
    return dict(nominal_mm3=round(v(pair(), hood), 4), leaf_preload_mm3=round(v(pair(), leaf), 2),
                insertion_mm3=round(max(v(pair(dz=dz), flexed) for dz in np.arange(-45.0, 0.01, 0.5)), 4),
                drop_mm=first(np.arange(0.05, 3.0, 0.05), lambda d: v(pair(dz=-d, cable=False), hood)),
                forward_travel_mm=first(np.arange(0.0, 3.0, 0.02), lambda d: v(pair(dx=d, cable=False), hood)),
                back_travel_mm=first(np.arange(0.0, 3.0, 0.05), lambda d: v(pair(dx=-d, cable=False), hood)))


def goobay_fit_c(p, hood_full=None):
    """The goobay stack (fc.goobay_stack: the adapter, goobay and cable plug) in the egg-C hood, with the adapter pressed
    against its pocket's roof (the clamp, fc.PLUG_C catch_lift/pocket_top: the adapter's catch is left out of the rest
    checks, it presses by design): clear at its two stops and in between; how far the catch's fingers must bend aside
    to let the adapter sit there (the clamp's interference); rising in from below past every bent finger (the adapter's,
    the goobay's GCATCH, the moulding's CSEAT), held forward and as far back as the rear wall lets it; how far it drops
    with the adapter's catch taken out (the goobay's and the moulding's hooks carry it); how far it goes past each
    stop; how far the two new hooks reach under it at each stop; whether their fingers have room to bend aside."""
    import dataclasses, cad
    p = dataclasses.replace(p, **fc.PLUG_C); X = fc.float_x(p); hood = hood_full or fc.hood_c_cad(p); C = fc.GCATCH
    _, _, fingers = cad.catch(p); _, _, bent = cad.catch(p, flex=cad.CATCH_HOOK - 0.2 + 0.05)
    g0, gbent = fc._goobay_catch(p), fc._goobay_catch(p, flex=C["hook"] - 0.1)
    c0, cbent = fc._cord_seat(p), fc._cord_seat(p, flex=C["hook"] - 0.1)
    flexed = hood - fingers + bent - g0 + gbent - c0 + cbent; no_catch = hood - fingers
    f1, f0 = X["face_fwd"], X["face_back"]; zc = p.adapter_z; up = p.pocket_top - 0.01        # pressed to the roof
    below_top = cad.cube(-200, 200, -200, 200, -50, p.trunk_top - 0.05)
    v = lambda a, b: float((a ^ b).volume())
    first = lambda xs, f: next((round(float(x), 2) for x in xs if f(x) > 0.01), None)
    first_clear = lambda xs, f: next((round(float(x), 2) for x in xs if f(x) <= 0.005), None)
    stack = lambda fx, dz=0.0, cable=True: fc.goobay_stack(p, fx, dz, cable)
    adapter = lambda fx: cad.adapter_solid(dataclasses.replace(p, face_offset=fx - p.tip_face_x)).translate((0, 0, up))
    tip = X["front_fwd"] + 0.15 - C["hook"]
    return dict(rest_mm3=round(max(v(stack(f, up), no_catch) for f in (f1, 0.5*(f1 + f0), f0)), 4),
                roof_gap_mm=round(p.pocket_top - up, 3),
                clamp_mm=first_clear(np.arange(0.0, 1.0, 0.01), lambda fl: v(adapter(f1), cad.catch(p, flex=fl)[2])),
                insertion_fwd_mm3=round(max(v(stack(f1, dz), flexed) for dz in np.arange(-45.0, 0.01, 0.5)), 4),
                insertion_back_mm3=round(max(v(stack(f1 - X["rise_gap"] + 0.02, dz), flexed) for dz in np.arange(-45.0, 0.01, 0.5)), 4),
                drop_no_catch_fwd_mm=first(np.arange(0.05, 3.0, 0.05), lambda d: v(stack(f1, up - d, False), no_catch)),
                drop_no_catch_back_mm=first(np.arange(0.05, 3.0, 0.05), lambda d: v(stack(f0, up - d, False), no_catch)),
                past_front_mm=first(np.arange(0.0, 3.0, 0.02), lambda d: v(stack(f1 + d, up, False), no_catch)),
                past_back_mm=first(np.arange(0.0, 3.0, 0.02), lambda d: v(stack(f0 - d, up, False), no_catch)),
                hook_under_fwd_mm=round(X["front_fwd"] - tip, 2), hook_under_back_mm=round(X["front_fwd"] - (f1 - f0) - tip, 2),
                hook_room_mm3=round(v(gbent ^ below_top, hood - g0) + v(cbent ^ below_top, hood - c0), 4),
                float_mm=round(f1 - f0, 2), rise_gap_mm=round(X["rise_gap"], 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT, help="output folder (default: hardware/microdock-v3, which holds stl/ and print/)")
    ap.add_argument("--bambu", default="", help="Bambu Studio binary: also build print/microdock_v3_P1S.3mf (and slice it with --check)")
    ap.add_argument("--check", action="store_true", help="fail if a check fails (see above)")
    a = ap.parse_args(); STL, PRINT = os.path.join(a.out, "stl"), os.path.join(a.out, "print")
    os.makedirs(STL, exist_ok=True); os.makedirs(PRINT, exist_ok=True)
    p = fd.FlexDock(**V4, **fc.PLUG_C)
    stage, _ = fc.stage_c_print(p)
    parts = {"base": sb.mesh(ss.base_cad(p)), "front": tm(fc.front_plate_c(p)), "dome": tm(fc.dome_c(p)), "stage": tm(stage),
             "hood": tm(fc.hood_c_cad(p)), "test": tm(fc.fit_test_c(p))}
    tag_black, tag_white = (sb.mesh(m_) for m_ in cad.tag_insert(p))          # the tag: two colours, two bodies
    parts["tag"] = trimesh.util.concatenate([tag_black, tag_white])
    info, bad = {}, []
    for k, m in parts.items():
        m.export(os.path.join(STL, f"{TAG}_{k}.stl"))
        info[k] = dict(extents_mm=[round(float(v), 1) for v in m.extents], volume_cm3=round(float(m.volume)/1e3, 1),
                       watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)))
        print(f"{TAG}_{k}.stl", json.dumps(info[k]))
        if not info[k]["watertight"] or (info[k]["bodies"] != 1 and k != "tag"): bad.append(f"{k}: watertight {info[k]['watertight']}, bodies {info[k]['bodies']}")   # (the tag: its base and the white cells)
    fit = fc.fit(p); pc = posed_check(p, fc._inner_depth()); sp = fc.springs_c(p); asm = assembly_check(p, parts)
    fw = fc.face_window(p); hood_m = fc.hood_c_cad(p); dp = dome_post_check(p)
    print("post pin and lip", json.dumps(dp))
    if dp["ears_to_stage_mm"] < 1.0 or dp["pin_run_mm"] < 15.0 or dp["old_dome_on_new_front_mm3"] > 0.5:
        bad.append(f"the post's ears or the lip don't clear: {dp}")
    print("fit", json.dumps(fit), "posed", json.dumps(pc), "springs", json.dumps(sp)); print("assembly", json.dumps(asm))
    print("face window", json.dumps(fw))
    if not (fw["base_margin"] >= 0.05 and fw["mouth"] <= fw["shell_at_stop"] - 0.25):
        bad.append(f"the lead-in is off: {fw}")
    if fc.CABLE_HEAD == "goobay":                              # the stack floats between its stops
        X = fc.float_x(dataclasses.replace(p, **fc.PLUG_C)); gf = goobay_fit_c(p, hood_m)
        print("goobay stack", json.dumps({k: round(float(v), 3) for k, v in X.items()}), json.dumps(gf))
        if not (fw["window"][0] - 0.35 <= X["face_back"] <= fw["tip_at_stop"] - 0.2 and fw["tip_at_stop"] + 0.4 <= X["face_fwd"] <= fw["stadium"][0] - 0.2):
            bad.append(f"the stack's stops are off: {X} {fw}")
        if gf["rest_mm3"] > 0.01 or gf["insertion_fwd_mm3"] > 0.01 or gf["insertion_back_mm3"] > 0.01 \
                or not 0.08 <= (gf["clamp_mm"] or 0) <= 0.3 or not (gf["past_front_mm"] or 9) <= 0.06 or not (gf["past_back_mm"] or 9) <= 0.06 \
                or gf["rise_gap_mm"] < 0.25 or gf["hook_under_fwd_mm"] < 1.5 or gf["hook_under_back_mm"] < 0.5 or gf["hook_room_mm3"] > 0.01 \
                or not (gf["drop_no_catch_fwd_mm"] or 9) <= 0.35 or not (gf["drop_no_catch_back_mm"] or 9) <= 0.35:
            bad.append(f"the goobay stack doesn't fit: {gf}")
    else:
        if not (fw["lengths"][0] <= 15.8 and fw["lengths"][1] >= 17.0):
            bad.append(f"the adapter's face window is off: {fw}")
        for L_ in (15.8, 16.4, 17.0):                          # the elbow + adapter: the photos' 16.1, the first hoods' 16.7
            hf = head_fit_c(p, L_, hood_m); print(f"cable head, elbow + adapter {L_} mm", json.dumps(hf))
            if hf["nominal_mm3"] > 0.01 or hf["insertion_mm3"] > 0.01 or not (hf["drop_mm"] or 9) <= 0.3 or not (hf["forward_travel_mm"] or 9) <= 0.1 \
                    or not 1.2 <= (hf["back_travel_mm"] or 0) <= 1.6 or not hf["leaf_preload_mm3"] > 1.0:
                bad.append(f"the cable head doesn't fit at {L_} mm: {hf}")
    if max(asm["dome_front"], asm["dome_stage"], asm["front_stage"]) > 0.5 or asm["hood_carriage"] > 1.3*asm["hood_carriage_ribs_expected"] + 0.5:
        bad.append(f"parts overlap: {asm}")
    if fit["dome_margin"] < 0.0 or fit["blade_to_post_tab"] < 1.0 or pc["back_to_dome"] < 0.0 or not pc["inside_side_wall"]:
        bad.append(f"the stage doesn't clear: {fit} {pc}")
    laid = {}
    for k, (title, name) in PLATES.items():                           # each part alone, centred, clear of the corner
        m = parts[name].copy(); T = np.eye(4); T[:3, :3] = ORIENT[name].as_matrix(); m.apply_transform(T)
        lo, hi = m.bounds; ext = hi - lo
        x0 = max((plates.BED[0] - ext[0])/2, 3.0); y0 = max((plates.BED[1] - ext[1])/2, plates.EXCLUDE[1] + 2.0)
        sh_ = -lo + np.array([x0, y0, 0.0]); m.apply_translation(sh_); lo, hi = m.bounds
        if hi[0] > plates.BED[0] or hi[1] > plates.BED[1] or hi[2] > 256.0: bad.append(f"plate {k} {name} off the bed {lo} {hi}")
        pcs = [(m, plates.WHITE, 1, "normal_part")]
        if name == "front":                                           # no supports under the front edge's round:
            (wy0, wy1) = (-p.tag_size/2 - 2.15 - 1.0, p.tag_size/2 + 2.15 + 1.0)   # blockers everywhere but the pocket
            cz = ss.EGG_TAG_CZ; tz0, tz1 = cz + wy0, cz + wy1
            for blo, bhi in (((-1, -1, -0.1), (lo[0] + (tz0 - 3.0) - 1, 300, 2.5)),
                             ((lo[0] + (tz1 - 3.0) + 1, -1, -0.1), (300, 300, 2.5)),
                             ((-1, -1, -0.1), (300, lo[1] + (wy0 + 90.3) - 1, 2.5)),
                             ((-1, lo[1] + (wy1 + 90.3) + 1, -0.1), (300, 300, 2.5))):
                bm = sb.box_mesh(np.array(blo, float), np.array(bhi, float)); pcs.append((bm, None, 1, "support_blocker"))
        if name == "tag":                                             # black base (extruder 2), then the white cells
            tb, tw = (q_.copy() for q_ in (tag_black, tag_white))
            for q_ in (tb, tw): q_.apply_transform(T); q_.apply_translation(sh_)
            pcs = [(tb, plates.BLACK, 2, "normal_part"), (tw, plates.WHITE, 1, "normal_part")]
        laid[k] = [(name, pcs)]
        stem = f"{TAG}_plate{k}_{name}"; m.export(os.path.join(PRINT, stem + ".stl"))
        plates.top_view(os.path.join(PRINT, stem + ".png"), [(name, [(q_, c_) for q_, c_, _, sub_ in pcs if sub_ == "normal_part"])])
    if a.bambu:
        work = tempfile.mkdtemp(prefix="microdock_v3_bambu_")
        try:
            out = os.path.join(PRINT, f"{TAG}_P1S.3mf")
            sb.bambu_project(a.bambu, out, work, laid, plate_names={k: v[0] for k, v in PLATES.items()}, labels=LABELS,
                             params=PARAMS, group_names={2: "Front plate", 6: "Tag"})
            print("wrote", out)
            if a.check:
                for r in sb.slice_check(a.bambu, out, work):
                    print("sliced", json.dumps(r)); bad += [f"plate {r['plate']}: {r['warning']}"] if r["warning"] else []
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if bad:
        if a.check: sys.exit(f"check failed: {bad}")
        print(f"warning: {bad}")
    elif a.check: print("check passed")


if __name__ == "__main__":
    main()

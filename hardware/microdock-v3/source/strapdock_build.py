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
"""The strap dock (experiment, before the egg dock; flexdock_c_build.py uses its plate and Bambu Studio helpers): its print set, on four P1S plates. The hood hangs on four flex straps
(strapswing.py; the joint of the test print's strap B, flexstrap.py) from a hanger on the egg board.
  1  the egg board, flat on its back (250 x 181 mm, a plate of its own)
  2  the base with its rounded corners
  3  the hood (upside down, supports from the plate, a support blocker in each socket), the hanger and the four straps
     (100 % infill and no brim, as the test print's)
  4  the tag (the MicroDock's): black base, then the white cells
The hanger's two pins are 1.75 mm filament, cut to length; v2's keeper isn't used.

  uv run hardware/microdock-v3/source/strapdock_build.py [--out DIR] [--check] [--bambu /path/to/bambu-studio] [--extra egg_straps]

--bambu also builds the Bambu Studio project with Bambu Studio's command-line slicer (as ../../microdock/source/
bambu_project.py: P1S 0.4 nozzle, 0.20 mm Standard, Bambu PLA Basic white and black, textured PEI plate) and, with
--check, slices every plate and reports its time and filament."""
import argparse, json, math, os, re, shutil, subprocess, sys, tempfile, zipfile
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "microdock", "source")); sys.path.insert(0, HERE)
import cad
import plates
import swing
import strapswing as ss
import flexstrap as fs

OUT = os.path.join(os.path.dirname(HERE), "straps")
TAG = "microdock_v3s"
ORIENT = {"board": Rot.from_euler("y", -90, degrees=True),     # on its back, the tag pocket up
          "base": Rot.identity(),                              # as it sits
          "hood": Rot.from_euler("x", 180, degrees=True),      # upside down, as the MicroDock's (supports from the plate)
          "hanger": Rot.identity(),                            # flat, as it sits
          "strap": Rot.identity(),                             # flat, as modelled (closed; the joints print in place)
          "tag": Rot.from_euler("y", -90, degrees=True)}       # face up: black base, white cells on top
# plate: (name, [(part, lower-left corner on the plate)]): clear of the P1S's front-left corner, 8 mm between parts
PLATES = {1: ("1 Egg board", [("board", 3.0, 37.7)]),
          2: ("2 Base", [("base", 30.5, 28.0)]),
          3: ("3 Hood, hanger, straps", [("hood", 9.9, 69.9), ("hanger", 117.0, 60.0)] +
              [(f"strap{k}", 200.0, 74.0 + (k - 1)*29.0) for k in range(1, 5)]),
          4: ("4 Tag (black, then white)", [("tag", 98.0, 98.0)])}
SLUG = {1: "egg_board", 2: "base", 3: "hood_hanger_straps", 4: "tag"}
# one-off single-plate projects (--extra NAME): e.g. a reprint of the straps together with the egg board (the straps in
# a row in front of it, the egg moved back for a 15 mm gap)
EXTRA = {"egg_straps": ("Egg board + 4 straps", [("board", 3.0, 44.0)] + [(f"strap{k}", 23.65 + (k - 1)*54.2, 8.0) for k in range(1, 5)]),
         "base_straps": ("Base + 4 straps", [("base", 30.5, 53.0)] + [(f"strap{k}", 23.65 + (k - 1)*54.2, 14.0) for k in range(1, 5)]),
         "hanger": ("Hanger", [("hanger", 90.5, 60.0)]),
         "straps": ("4 straps", [(f"strap{k}", 23.65 + (k - 1)*54.2, 117.5) for k in range(1, 5)])}
SUPPORT = {"enable_support": "1", "support_type": "normal(auto)", "support_on_build_plate_only": "1"}
STRAP = {"sparse_infill_density": "100%", "brim_type": "no_brim"}          # as the test print, whose strap B was fluid


def mesh(man):
    m = man.to_mesh(); return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def man(t):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))


def box_mesh(lo, hi):
    return trimesh.creation.box(bounds=np.array([lo, hi], float))


def checks(p):
    """Fit checks in the assembly (mm3 of overlap; 0 means clear)."""
    import manifold3d as m3
    s = p.strap(); out = {}
    H, D, B = ss.hanger_cad(p), ss.hood_cad(p), ss.egg_board(p); h0 = p.hood_home(); L = s.length_open()
    out["strap_rows_min_gap_mm"] = round(fs.min_gap(s), 3)
    # the straps with the hood lifted (closing the straps, then lifting the bars out of their grooves) and swung sideways
    worst = {}
    for lift in (0.0, s.R*s.play/2, s.R*s.play, s.R*s.play + 7.0):
        hood = D.translate(tuple(h0 + [0, 0, lift])); w = 0.0
        for site in p.strap_sites():
            for row in ss.strap_world(p, site, max(0.0, s.play - lift/s.R)):
                r = man(row).translate((0.0, 0.0, max(0.0, lift - s.R*s.play)))
                w = max(w, (r ^ hood).volume(), (r ^ H).volume(), (r ^ B).volume())
        worst[f"lift_{lift:.1f}"] = round(w, 4)
    for dy in (-20.0, 20.0):                                   # a pendulum swing: the straps turn about their bars
        th = np.arcsin(dy/L); rise = L*(1 - np.cos(th)); hood = D.translate(tuple(h0 + [0, dy, rise])); w = 0.0
        for site in p.strap_sites():
            pw = site[2]
            for row in ss.strap_world(p, site, s.play):
                r = man(row).translate(tuple(-pw)).rotate((float(np.degrees(th)), 0.0, 0.0)).translate(tuple(pw))
                w = max(w, (r ^ hood).volume(), (r ^ H).volume())
        worst[f"swing_{dy:+.0f}"] = round(w, 4)
    out["strap_overlap_mm3"] = worst
    # the cable: out of the left wall's slot and straight down beside the wall (4 mm, bent at 8 mm radius)
    ce = swing.cable_exit(p); r = cad.CABLE_D/2; R = 8.0
    path = [np.array([ce[0], p.side_in - 1.0, ce[2]]), np.array([ce[0], ce[1] + 1.0, ce[2]])]
    for a in np.linspace(0, np.pi/2, 10)[1:]:
        path.append(np.array([ce[0], ce[1] + 1.0 + R*np.sin(a), ce[2] - R + R*np.cos(a)]))
    path.append(path[-1] + [0, 0, -(ce[2] - R + 10.0)])
    tube = None
    for a_, b_ in zip(path[:-1], path[1:]):
        seg = m3.Manifold.cylinder(float(np.linalg.norm(b_ - a_)), r, r, 24)
        d = (b_ - a_)/np.linalg.norm(b_ - a_); ax = np.cross([0, 0, 1], d); ang = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
        if np.linalg.norm(ax) > 1e-9:
            rv = Rot.from_rotvec(ax/np.linalg.norm(ax)*np.radians(ang)).as_euler("xyz", degrees=True)
            seg = seg.rotate(tuple(float(v) for v in rv))
        seg = seg.translate(tuple(a_)) + m3.Manifold.sphere(r, 24).translate(tuple(a_))
        tube = seg if tube is None else tube + seg
    out["cable_overlap_with_hood_mm3"] = round((tube ^ D).volume(), 4)
    # the knuckle's click: how far the ridge stands into the knuckle's path at the socket's entrance
    rb = s.T/2 + 0.15; out["click_interference_mm"] = round((-rb + 0.5) - (-s.T/2), 2)
    out["cable_clip_world_yz"] = [round(v, 1) for v in ss.egg_clip(p)]
    # the hanger in the board: its tongues clear of the slots' walls (the slots: TONGUE_Y +- 0.15, bar_z +- 0.1)
    ds, dt, _ = ss.TONGUE_FIT
    out["hanger_in_board_overlap_mm3"] = round((H ^ B).volume(), 4)
    out["tongue_clearance_mm"] = dict(each_side=round(0.15 + ds, 2), top=round(0.1 + dt, 2), bottom=0.1)
    # the knuckle's flat: still wider than the socket's throat, still bearing on the throat's edge on its flat side
    out["knuckle_mm"] = dict(across_throat=round(s.T - s.knuckle_flat, 2), throat=round(2*p.throat, 2),
                             bearing_flat_side=round(s.T/2 - s.knuckle_flat - p.throat, 2),
                             first_layer_width=round(2*math.sqrt(max((s.T/2)**2 - (s.T/2 - s.knuckle_flat - 0.1)**2, 0.0)), 2))
    return out


def layout(p, meshes, plate_set=None):
    """Every plate's parts placed (plate mm): {plate: [(name, [(mesh, colour, extruder, subtype)])]} and the problems."""
    out, bad = {}, []
    for k, (_, items) in (plate_set or PLATES).items():
        placed, boxes = [], []
        for name, x, y in items:
            kind = "strap" if name.startswith("strap") else name
            T = np.eye(4); T[:3, :3] = ORIENT[kind].as_matrix()
            pcs = [(m.copy(), c, e) for m, c, e in meshes[kind]]
            for m, _, _ in pcs: m.apply_transform(T)
            lo = np.min([m.bounds[0] for m, _, _ in pcs], 0); shift = -lo + np.array([x, y, 0.0])
            for m, _, _ in pcs: m.apply_translation(shift)
            lo, hi = lo + shift, np.max([m.bounds[1] for m, _, _ in pcs], 0)
            if not (lo[0] >= 0 and lo[1] >= 0 and hi[0] <= plates.BED[0] and hi[1] <= plates.BED[1]): bad.append(f"plate {k} {name} off the bed {lo} {hi}")
            if lo[0] < plates.EXCLUDE[0] and lo[1] < plates.EXCLUDE[1]: bad.append(f"plate {k} {name} in the front-left corner")
            for o in boxes:
                if max(o[0][0] - hi[0], lo[0] - o[1][0], o[0][1] - hi[1], lo[1] - o[1][1]) < 8.0: bad.append(f"plate {k} {name} too close")
            boxes.append((lo, hi))
            parts = [(m, c, e, "normal_part") for m, c, e in pcs]
            if kind == "hood":                                 # the sockets' blockers, placed with the hood
                for blo, bhi in ss.socket_blockers(p):
                    bm = box_mesh(blo, bhi); bm.apply_transform(T); bm.apply_translation(shift); parts.append((bm, None, 1, "support_blocker"))
            placed.append((name, parts))
        out[k] = placed
    return out, bad


def bambu_project(bambu, out, work, laid, plate_names=None, labels=None, params=None, group_names=None):
    """The Bambu Studio project, made by its command-line slicer from the plates' parts (assemble list): the parts as
    placed, the hood and its blockers one object, the tag's two colours one object (black: filament 2). A part list of
    more than one part is one object (at most one such per plate), named by group_names {plate: name}."""
    import bambu_project as bp
    resources = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(bambu))), "resources")
    parts_dir = os.path.join(work, "parts"); os.makedirs(parts_dir, exist_ok=True)
    cfg = {kind: bp.flatten(resources, kind, name) for kind, name in bp.PRESETS}
    cfg["process"].update(bp.BED)
    paths = {}
    for kind in ("machine", "process"):
        paths[kind] = os.path.join(work, kind + ".json"); json.dump(cfg[kind], open(paths[kind], "w"), indent=1)
    for col, hexc in (("white", "#FFFFFF"), ("black", "#000000")):
        paths[col] = os.path.join(work, f"filament_{col}.json"); json.dump(dict(cfg["filament"], filament_colour=[hexc]), open(paths[col], "w"), indent=1)
    names, plates_json = {}, []
    for k, placed in laid.items():
        objs, asm = [], []
        for name, parts in placed:
            group = len(parts) > 1
            for j, (m, c, e, sub) in enumerate(parts):
                stem = f"{TAG}_{name}" + ("" if not group else f"_{'blocker' + str(j) if sub != 'normal_part' else ('black' if e == 2 else 'white' if name == 'tag' else 'body')}")
                path = os.path.join(parts_dir, stem + ".stl"); m.export(path)
                o = {"path": path, "count": 1, "filaments": [e], "pos_x": [0.0], "pos_y": [0.0], "pos_z": [0.0]}
                if sub != "normal_part": o["subtype"] = sub
                if group: o["assemble_index"] = [1]
                elif params and name in params: o["print_params"] = params[name]
                elif name.startswith("strap"): o["print_params"] = STRAP
                objs.append(o)
                if sub != "normal_part": label = (labels or {}).get(f"{name}_blocker", f"socket blocker {j}")
                elif labels and name in labels: label = labels[name]
                elif name == "hood": label = "Hood"
                elif name == "tag": label = "tag black base" if e == 2 else "tag white cells"
                else: label = {"board": "Egg board", "base": "Base", "hanger": "Hanger"}.get(name, f"Strap {name[5:]}")
                names[f"{stem}_1"] = label
            if group: asm.append({"assemble_index": 1, "print_params": SUPPORT} if name == "hood" else
                                 {"assemble_index": 1, "print_params": params[name]} if params and name in params else {"assemble_index": 1})
        pj = {"plate_name": (plate_names or {k_: v[0] for k_, v in PLATES.items()})[k], "need_arrange": False, "plate_params": bp.BED, "objects": objs}
        if asm: pj["assembled_params"] = asm
        plates_json.append(pj)
    json.dump({"plates": plates_json}, open(os.path.join(work, "assemble.json"), "w"), indent=1)
    raw = os.path.join(work, "raw.3mf")
    r = subprocess.run([bambu, "--datadir", os.path.join(work, "data"), "--load-settings", f"{paths['machine']};{paths['process']}",
                        "--load-filaments", f"{paths['white']};{paths['black']}", "--load-assemble-list", os.path.join(work, "assemble.json"),
                        "--outputdir", work, "--export-3mf", os.path.basename(raw)], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(raw): sys.exit(f"bambu-studio failed ({r.returncode}): {r.stderr[-2000:]}")
    group_names = group_names or {3: "Hood", 4: "Tag"}              # the merged objects (each plate's "assemble_1")
    with zipfile.ZipFile(raw) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:     # readable object names
        for info in zi.infolist():
            data = zi.read(info.filename)
            if info.filename == "Metadata/model_settings.config" or info.filename.startswith("3D/"):
                t = data.decode("utf-8")
                for a, b in names.items(): t = t.replace(f'"{a}"', f'"{b}"')
                if info.filename == "Metadata/model_settings.config":
                    on_plate = {oid: int(pl) for blk in t.split("<plate>")[1:]
                                for pl in re.findall(r'key="plater_id" value="(\d+)"', blk)[:1]
                                for oid in re.findall(r'key="object_id" value="(\d+)"', blk)}
                    def rename(m_):
                        nm = group_names.get(on_plate.get(m_.group(1))) if m_.group(2).startswith("assemble_") else None
                        return m_.group(0) if nm is None else m_.group(0).replace(f'"{m_.group(2)}"', f'"{nm}"')
                    t = re.sub(r'<object id="(\d+)">\s*<metadata key="name" value="([^"]*)"/>', rename, t)
                    assert "assemble_" not in t, "a merged object kept its generated name"
                data = t.encode("utf-8")
            zo.writestr(info, data)


def slice_check(bambu, project, work):
    """Slice every plate from the project's own settings: time, filament, warnings."""
    out = os.path.join(work, "check"); os.makedirs(out, exist_ok=True)
    subprocess.run([bambu, "--datadir", os.path.join(work, "data"), "--slice", "0", "--outputdir", out, project], check=True, capture_output=True)
    r = json.load(open(os.path.join(out, "result.json")))
    if r["return_code"] != 0: sys.exit(f"slicing failed: {r['error_string']}")
    rep = []
    for pl in r["sliced_plates"]:
        g = {f["id"]: round(f["total_used_g"], 1) for f in pl["filaments"]}
        rep.append(dict(plate=pl["id"], minutes=round(pl["total_predication"]/60), filament_g=g, changes=pl["filament_change_times"], warning=pl["warning_message"]))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT); ap.add_argument("--check", action="store_true")
    ap.add_argument("--bambu", default="", help="Bambu Studio binary: also build the Bambu Studio project (and slice it with --check)")
    ap.add_argument("--extra", default="", help=f"also these one-off single-plate projects (comma list of {sorted(EXTRA)})")
    a = ap.parse_args()
    p = ss.Straps(); s = p.strap(); os.makedirs(a.out, exist_ok=True)
    build = {"board": ss.egg_board, "base": ss.base_cad, "hood": ss.hood_cad, "hanger": ss.hanger_cad, "strap": lambda p_: fs.strap_cad(p_.strap())}
    info, meshes = {}, {}
    for name, f in build.items():
        m = mesh(f(p)); meshes[name] = [(m, plates.WHITE, 1)]; m.export(os.path.join(a.out, f"{TAG}_{name}.stl"))
        info[name] = dict(extents_mm=[round(float(v), 1) for v in m.extents], volume_cm3=round(float(m.volume)/1000, 1),
                          watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)))
        print(f"{TAG}_{name}.stl", json.dumps(info[name]))
    blk, white = (mesh(m_) for m_ in cad.tag_insert(p)); meshes["tag"] = [(blk, plates.BLACK, 2), (white, plates.WHITE, 1)]
    laid, bad = layout(p, meshes)
    for k, placed in laid.items():
        stem = f"{TAG}_plate{k}_{SLUG[k]}"
        trimesh.util.concatenate([m for _, pcs in placed for m, _, _, sub in pcs if sub == "normal_part"]).export(os.path.join(a.out, stem + ".stl"))
        plates.top_view(os.path.join(a.out, stem + ".png"), [(n, [(m, c) for m, c, _, sub in pcs if sub == "normal_part"]) for n, pcs in placed])
    fit = checks(p); print("checks", json.dumps(fit))
    if a.bambu:
        work = tempfile.mkdtemp(prefix="microdock_v3s_bambu_")
        try:
            out = os.path.join(a.out, f"{TAG}_P1S.3mf"); bambu_project(a.bambu, out, work, laid); print("wrote", out)
            if a.check:
                rep = slice_check(a.bambu, out, work)
                for r_ in rep: print("sliced", json.dumps(r_))
                bad += [f"plate {r_['plate']}: {r_['warning']}" for r_ in rep if r_["warning"]]
        finally:
            shutil.rmtree(work, ignore_errors=True)
    for name in [n for n in a.extra.split(",") if n]:                 # the one-off plates
        laid_x, bad_x = layout(p, meshes, {1: EXTRA[name]}); bad += bad_x
        stem = f"{TAG}_{name}"; placed = laid_x[1]
        trimesh.util.concatenate([m for _, pcs in placed for m, _, _, sub in pcs if sub == "normal_part"]).export(os.path.join(a.out, stem + ".stl"))
        plates.top_view(os.path.join(a.out, stem + ".png"), [(n, [(m, c) for m, c, _, sub in pcs if sub == "normal_part"]) for n, pcs in placed])
        if a.bambu:
            work = tempfile.mkdtemp(prefix="microdock_v3s_bambu_")
            try:
                out = os.path.join(a.out, f"{TAG}_P1S_{name}.3mf"); bambu_project(a.bambu, out, work, laid_x, plate_names={1: EXTRA[name][0]}); print("wrote", out)
                if a.check:
                    for r_ in slice_check(a.bambu, out, work):
                        print("sliced", name, json.dumps(r_)); bad += [f"{name}: {r_['warning']}"] if r_["warning"] else []
            finally:
                shutil.rmtree(work, ignore_errors=True)
    if a.check:
        bad += [k for k, v in info.items() if not v["watertight"]]
        bad += [k for k, v in info.items() if v["bodies"] != (s.R + 1 if k == "strap" else 1)]
        if fit["strap_rows_min_gap_mm"] < min(s.cxy, s.cz) - 1e-6: bad.append("strap rows closer than the clearances")
        bad += [f"strap overlap {k}" for k, v in fit["strap_overlap_mm3"].items() if v > 0.01]
        if fit["cable_overlap_with_hood_mm3"] > 0.01: bad.append("cable path hits the hood")
        if fit["hanger_in_board_overlap_mm3"] > 0.0: bad.append("the hanger's tongues hit the board")
        if min(fit["tongue_clearance_mm"]["each_side"], fit["tongue_clearance_mm"]["top"]) < 0.3: bad.append("tongues too tight in their slots")
        if fit["knuckle_mm"]["bearing_flat_side"] < 0.1: bad.append("the knuckle's flat side no longer bears on the socket's throat")
        if bad: sys.exit(f"check failed: {bad}")
        print(f"check passed: {len(info)} parts watertight (the strap's {s.R + 1} rows print linked, apart), plates within the bed, "
              "straps clear of hood, hanger and board, cable path clear")

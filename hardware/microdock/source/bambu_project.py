# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "manifold3d>=3.2",
#   "numpy>=2.0",
#   "pillow>=10",
#   "scipy>=1.10",
#   "trimesh>=4.8",
# ]
# ///
"""Bambu Studio project for a Bambu Lab P1S: the three plates from plates.py with their settings, ready to slice.
P1S 0.4 nozzle, the 0.20 mm Standard process, Bambu PLA Basic in white (filament 1) and black (filament 2), the
textured PEI plate. The hood carries its own support setting (grid supports, on the build plate only); the tag's black
base and white cells are one object with a part per filament.

Made with Bambu Studio's command-line slicer: its system presets are flattened the way Bambu Studio loads them (parent,
then the included templates, then the preset's own keys) and the plates are built with its assemble list.

  uv run hardware/microdock/source/bambu_project.py --bambu /path/to/bambu-studio [--check]
  (--resources defaults to the resources folder next to the binary, as in the AppImage)
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plates
from dock import Params

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS = (("machine", "Bambu Lab P1S 0.4 nozzle"), ("process", "0.20mm Standard @BBL X1C"), ("filament", "Bambu PLA Basic @BBL P1S 0.4 nozzle"))
META = {"inherits", "include", "name", "from", "instantiation", "setting_id", "type", "description"}
SUPPORT = {"enable_support": "1", "support_type": "normal(auto)", "support_on_build_plate_only": "1"}   # tree supports leave the pegs' flats bare
BED = {"curr_bed_type": "Textured PEI Plate"}
NAMES = {"assemble_1": "microdock_tag", "microdock_tag_black_1": "tag black base", "microdock_tag_white_1": "tag white cells",
         "microdock_base_1": "microdock_base", "microdock_board_1": "microdock_board", "microdock_hood_1": "microdock_hood",
         "microdock_keeper_1": "microdock_keeper"}


def flatten(resources, kind, name):
    db = {}
    for f in glob.glob(os.path.join(resources, "profiles", "BBL", kind, "**", "*.json"), recursive=True):
        try: j = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError): continue
        if isinstance(j, dict) and "name" in j: db[j["name"]] = j
    def includes(j):
        inc = j.get("include", [])
        if isinstance(inc, str):
            inc = json.loads(inc.replace("'", '"')) if inc.strip().startswith("[") else [inc.strip().strip('"')]
        return inc
    def rec(n):
        j = db[n]; cfg = dict(rec(j["inherits"])) if j.get("inherits") else {}
        for inc in includes(j): cfg.update({k: v for k, v in db[inc].items() if k not in META})
        cfg.update({k: v for k, v in j.items() if k not in META})
        return cfg
    cfg = rec(name); cfg.update({"name": name, "from": "system", "type": kind, "instantiation": "true", "inherits": ""})
    if "setting_id" in db[name]: cfg["setting_id"] = db[name]["setting_id"]
    return cfg


def build(bambu, resources, out, work):
    parts = os.path.join(work, "parts"); os.makedirs(parts, exist_ok=True)
    files = {}
    for k, placed in plates.layout(Params()).items():
        for name, pcs in placed:
            for m, c in pcs:
                stem = f"microdock_{name}" + ("" if len(pcs) == 1 else ("_black" if c == plates.BLACK else "_white"))
                m.export(os.path.join(parts, stem + ".stl")); files[stem] = os.path.join(parts, stem + ".stl")
    cfg = {kind: flatten(resources, kind, name) for kind, name in PRESETS}
    cfg["process"].update(BED)
    paths = {}
    for kind in ("machine", "process"):
        paths[kind] = os.path.join(work, kind + ".json"); json.dump(cfg[kind], open(paths[kind], "w"), indent=1)
    for col, hexc in (("white", "#FFFFFF"), ("black", "#000000")):
        paths[col] = os.path.join(work, f"filament_{col}.json"); json.dump(dict(cfg["filament"], filament_colour=[hexc]), open(paths[col], "w"), indent=1)
    def obj(stem, fil, **kw): return dict({"path": files[stem], "count": 1, "filaments": [fil], "pos_x": [0.0], "pos_y": [0.0], "pos_z": [0.0]}, **kw)
    assemble = {"plates": [
        {"plate_name": "1 Base", "need_arrange": False, "plate_params": BED, "objects": [obj("microdock_base", 1)]},
        {"plate_name": "2 Board, hood, keeper", "need_arrange": False, "plate_params": BED,
         "objects": [obj("microdock_board", 1), obj("microdock_hood", 1, print_params=SUPPORT), obj("microdock_keeper", 1)]},
        {"plate_name": "3 Tag (black, then white)", "need_arrange": False, "plate_params": BED,
         "objects": [obj("microdock_tag_black", 2, assemble_index=[1]), obj("microdock_tag_white", 1, assemble_index=[1])]}]}
    json.dump(assemble, open(os.path.join(work, "assemble.json"), "w"), indent=1)
    raw = os.path.join(work, "raw.3mf")
    subprocess.run([bambu, "--datadir", os.path.join(work, "data"), "--load-settings", f"{paths['machine']};{paths['process']}",
                    "--load-filaments", f"{paths['white']};{paths['black']}", "--load-assemble-list", os.path.join(work, "assemble.json"),
                    "--outputdir", work, "--export-3mf", os.path.basename(raw)], check=True, capture_output=True)
    with zipfile.ZipFile(raw) as zi, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:     # readable object names
        for info in zi.infolist():
            data = zi.read(info.filename)
            if info.filename == "Metadata/model_settings.config" or info.filename.startswith("3D/"):
                t = data.decode("utf-8")
                for a, b in NAMES.items(): t = t.replace(f'"{a}"', f'"{b}"')
                data = t.encode("utf-8")
            zo.writestr(info, data)


def check(bambu, project, work):
    """Slice every plate from the project's own settings; fail on any slicing warning."""
    out = os.path.join(work, "check"); os.makedirs(out, exist_ok=True)
    subprocess.run([bambu, "--datadir", os.path.join(work, "data"), "--slice", "0", "--outputdir", out, project], check=True, capture_output=True)
    r = json.load(open(os.path.join(out, "result.json")))
    if r["return_code"] != 0: sys.exit(f"slicing failed: {r['error_string']}")
    for p in r["sliced_plates"]:
        g = {f["id"]: round(f["total_used_g"], 1) for f in p["filaments"]}
        print(f"plate {p['id']}: {round(p['total_predication']/60)} min, filament g {g}, filament changes {p['filament_change_times']}")
        if p["warning_message"]: sys.exit(f"plate {p['id']}: {p['warning_message']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bambu", required=True, help="Bambu Studio binary (command-line use)")
    ap.add_argument("--resources", default=None, help="its resources folder (default: next to the binary)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "print", "microdock_P1S.3mf"))
    ap.add_argument("--check", action="store_true", help="slice the result and fail on warnings")
    a = ap.parse_args()
    resources = a.resources or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.bambu))), "resources")
    work = tempfile.mkdtemp(prefix="microdock_bambu_")
    try:
        build(a.bambu, resources, os.path.abspath(a.out), work); print("wrote", os.path.abspath(a.out))
        if a.check: check(a.bambu, os.path.abspath(a.out), work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

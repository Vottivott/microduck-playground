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
"""MicroDock v2 parts: the seated dock's new board, the P1S print plates and, with --bambu, the Bambu Studio project.
The hood, base, keeper and tag are the MicroDock's (hardware/microdock/stl); --check confirms that they come out
unchanged with the seated parameters, that the cable head fits the hood and its catch, and that the keeper holds the
hood wherever it floats on the new board.

uv run hardware/microdock-v2/source/build.py [--out DIR] [--check] [--bambu /path/to/bambu-studio]
"""
import argparse, json, os, shutil, sys, tempfile
import numpy as np
import trimesh
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "microdock", "source")
sys.path.insert(0, V1); sys.path.insert(0, HERE)
import cad
import dock
import plates
import seated

OUT = os.path.dirname(HERE)


def mesh(man):
    m = man.to_mesh(); return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def same(a, b):
    """Two parts are the same solid: nothing of either lies outside the other (manifold3d: - is the difference)."""
    return (a - b).volume() + (b - a).volume() < 1e-3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="fail unless the board is one watertight body within the bed, the cable head fits the hood and its catch, the keeper holds the hood wherever it is, and the other parts are the MicroDock's")
    ap.add_argument("--out", default=OUT, help="output folder (default: hardware/microdock-v2, which holds stl/ and print/)")
    ap.add_argument("--bambu", default="", help="Bambu Studio binary: also build print/microdock_v2_P1S.3mf (and slice it with --check)")
    a = ap.parse_args()
    OUT = a.out
    p, p1 = seated.Params(), dock.Params()
    os.makedirs(os.path.join(OUT, "stl"), exist_ok=True); os.makedirs(os.path.join(OUT, "print"), exist_ok=True)
    info = {}
    for name in ("board",):
        m = mesh(getattr(cad, name)(p)); m.export(os.path.join(OUT, "stl", f"microdock_v2_{name}.stl"))
        info[name] = dict(extents_mm=[round(float(v), 1) for v in m.extents], volume_cm3=round(float(m.volume)/1000, 1),
                          watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)))
        print(f"microdock_v2_{name}.stl", json.dumps(info[name]))
    # the print plates (the MicroDock's layout, with this board)
    names = {1: "base", 2: "board_hood_keeper", 3: "tag"}
    for k, placed in plates.layout(p).items():
        stem = f"microdock_v2_plate{k}_{names[k]}"
        trimesh.util.concatenate([m for _, pcs in placed for m, _ in pcs]).export(os.path.join(OUT, "print", stem + ".stl"))
        plates.top_view(os.path.join(OUT, "print", stem + ".png"), placed)
    if a.check:
        bad = [k for k, v in info.items() if not (v["watertight"] and v["bodies"] == 1 and max(v["extents_mm"]) <= cad.BED_MM)]
        fit = cad.head_fit(p); print("cable head fit:", json.dumps(fit))
        bad += cad.fit_problems(fit)
        kf = cad.keeper_fit(p); print("keeper:", json.dumps(kf))
        bad += cad.keeper_problems(kf)
        for name in ("hood", "base", "keeper"):
            if not same(getattr(cad, name)(p), getattr(cad, name)(p1)): bad.append(f"{name} differs from the MicroDock's")
        if not all(same(x, y) for x, y in zip(cad.tag_insert(p), cad.tag_insert(p1))): bad.append("tag differs from the MicroDock's")
        if bad:
            sys.exit(f"check failed: {bad}")
        print("check passed: the board is one watertight body within the bed, the cable head fits the hood and its catch; "
              f"the keeper bears at least {kf['min_bearing_mm2']} mm2 on the board wherever the hood is; hood, base, keeper and tag are the MicroDock's")
    if a.bambu:
        import bambu_project as bp
        resources = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.bambu))), "resources")
        out = os.path.join(OUT, "print", "microdock_v2_P1S.3mf"); work = tempfile.mkdtemp(prefix="microdock_v2_bambu_")
        try:
            bp.build(a.bambu, resources, out, work, p=p); print("wrote", out)
            if a.check: bp.check(a.bambu, out, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)

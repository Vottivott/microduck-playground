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
"""Print plates for a Bambu Lab P1S (256 x 256 mm): every part turned to its print orientation and laid out.
The parts don't fit one plate (the base and the board together are too big), so there are three:
  1  base
  2  board (on its back), hood (upside down), keeper
  3  tag: black base with the white cells on top (its two colours as two parts)
For each plate: an STL of the whole plate and a top view. bambu_project.py builds the Bambu Studio project from the
same layout.

uv run hardware/microdock/source/plates.py [--out DIR]
"""
import argparse, json, os, sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cad
from dock import Params

HERE = os.path.dirname(os.path.abspath(__file__))
BED = (256.0, 256.0)                          # P1S build plate
EXCLUDE = (18.0, 28.0)                        # P1S: keep clear of the front-left corner (bed_exclude_area)
WHITE, BLACK = "#F2F2EEFF", "#1A1A1AFF"

# print orientation: rotation applied to each part in its assembly frame (see cad.py)
ORIENT = {"base": Rot.identity(),
          "board": Rot.from_euler("y", -90, degrees=True),     # on its back, the tag pocket up
          "hood": Rot.from_euler("x", 180, degrees=True),      # upside down: needs supports, see the README
          "keeper": Rot.from_euler("y", -90, degrees=True),    # flat
          "tag": Rot.from_euler("y", -90, degrees=True)}       # face up: black base, white cells on top
# plate -> [(part, x, y of the part's front-left corner on the plate)]
PLATES = {1: [("base", 30.5, 28.0)],
          2: [("board", 4.5, 140.0), ("hood", 25.0, 15.0), ("keeper", 160.0, 60.0)],
          3: [("tag", 98.0, 98.0)]}


def mesh(man):
    m = man.to_mesh(); return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def parts(p: Params):
    """name -> list of (mesh, colour) in print orientation, resting on z = 0 with the footprint at x, y >= 0."""
    raw = {"base": [(mesh(cad.base(p)), WHITE)], "board": [(mesh(cad.board(p)), WHITE)], "hood": [(mesh(cad.hood(p)), WHITE)],
           "keeper": [(mesh(cad.keeper(p)), WHITE)]}
    blk, white = cad.tag_insert(p)
    raw["tag"] = [(mesh(blk), BLACK), (mesh(white), WHITE)]
    out = {}
    for name, pieces in raw.items():
        T = np.eye(4); T[:3, :3] = ORIENT[name].as_matrix()
        ms = [m.copy().apply_transform(T) for m, _ in pieces]
        lo = np.min([m.bounds[0] for m in ms], axis=0)
        out[name] = [(m.apply_translation(-lo), c) for m, (_, c) in zip(ms, pieces)]
    return out


def layout(p: Params):
    """plate -> [(part name, [(mesh, colour)])] positioned on the plate; checks bed, corner and spacing."""
    P = parts(p); plates = {}
    for k, items in PLATES.items():
        placed, boxes = [], []
        for name, x, y in items:
            pcs = [(m.copy().apply_translation((x, y, 0)), c) for m, c in P[name]]
            lo = np.min([m.bounds[0] for m, _ in pcs], axis=0); hi = np.max([m.bounds[1] for m, _ in pcs], axis=0)
            assert lo[0] >= 0 and lo[1] >= 0 and hi[0] <= BED[0] and hi[1] <= BED[1], (k, name, lo, hi)
            assert not (lo[0] < EXCLUDE[0] and lo[1] < EXCLUDE[1]), (k, name, "front-left corner")
            for o in boxes:
                gap = max(o[0][0] - hi[0], lo[0] - o[1][0], o[0][1] - hi[1], lo[1] - o[1][1])
                assert gap >= 8.0, (k, name, gap)
            boxes.append((lo, hi)); placed.append((name, pcs))
        plates[k] = placed
    return plates


def top_view(path, placed, px=4):
    """Top view of the plate: each part's height as shading, the excluded corner hatched."""
    from PIL import Image, ImageDraw
    W, H = int(BED[0]*px), int(BED[1]*px)
    img = Image.new("RGB", (W, H), (44, 46, 52)); d = ImageDraw.Draw(img)
    for i in range(0, W + H, 12):
        d.line([(i, H), (i - H, 0)], fill=(52, 54, 60))
    d.rectangle([0, H - EXCLUDE[1]*px, EXCLUDE[0]*px, H], fill=(70, 50, 50))
    zmax = max(m.bounds[1][2] for _, pcs in placed for m, _ in pcs)
    for _, pcs in placed:
        for m, c in pcs:
            tri = m.triangles; up = m.face_normals[:, 2] > 0.2
            order = np.argsort(tri[up][:, :, 2].mean(1))
            base = (242, 242, 238) if c == WHITE else (40, 40, 40)
            for t in tri[up][order]:
                k = 0.55 + 0.45*t[:, 2].mean()/zmax
                d.polygon([(x*px, H - y*px) for x, y, _ in t], fill=tuple(int(v*k) for v in base))
    d.rectangle([0, 0, W - 1, H - 1], outline=(120, 124, 132), width=2)
    img.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "print"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    names = {1: "base", 2: "board_hood_keeper", 3: "tag"}
    report = {}
    for k, placed in layout(Params()).items():
        stem = f"microdock_plate{k}_{names[k]}"
        trimesh.util.concatenate([m for _, pcs in placed for m, _ in pcs]).export(os.path.join(a.out, stem + ".stl"))
        top_view(os.path.join(a.out, stem + ".png"), placed)
        report[stem] = {name: [round(float(v), 1) for v in np.max([m.bounds[1] for m, _ in pcs], axis=0) - np.min([m.bounds[0] for m, _ in pcs], axis=0)]
                        for name, pcs in placed}
    print(json.dumps(report, indent=1))

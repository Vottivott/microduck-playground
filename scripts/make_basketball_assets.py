#!/usr/bin/env python3
"""Generate the basketball look as real geometry + a colour texture.

  uv run --with pillow --with scipy python scripts/make_basketball_assets.py

Seam centre-lines (user's idealized eight-panel model, 2026-09-06): on the
unit sphere G1(t) = (cos t, 0, sin t), G2(t) = (0, cos t, sin t) and the
closed non-planar seam C(t) = (a cos t - b cos 3t, a sin t + b sin 3t,
2 sqrt(ab) cos 2t) with a = 5/8, b = 3/8.  Seam bands use the geodesic
distance to dense samples of the curves (KD-tree on chord distance).

Geometry: a lat-long sphere mesh (SEGMENTS x RINGS) displaced radially by
hex-packed pebble bumps (+PEBBLE_MM at a crown) and by the seam channels
(-SEAM_DEPTH_MM inside the band, smooth edges).  Texture: orange with pebble
shading and dark channels.  Writes robot/assets/basketball/basketball.{obj,png}.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "robot" / "assets" / "basketball"
A, B = 5 / 8, 3 / 8
SEAM_HALF_WIDTH_RAD = 0.024  # ~2.9 mm each side on a 120 mm radius: 6 mm channel
SEAM_DEPTH_MM = 1.0
PEBBLE_MM = 0.55
PEBBLES_AROUND = 340  # ~2.2 mm pebbles
ORANGE = np.array([0.90, 0.38, 0.09])
CHANNEL = np.array([0.07, 0.05, 0.04])


def seam_samples(n: int = 24000) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    zero = np.zeros_like(t)
    g1 = np.column_stack((np.cos(t), zero, np.sin(t)))
    g2 = np.column_stack((zero, np.cos(t), np.sin(t)))
    wave = np.column_stack((A * np.cos(t) - B * np.cos(3 * t), A * np.sin(t) + B * np.sin(3 * t), 2.0 * math.sqrt(A * B) * np.cos(2 * t)))
    return np.concatenate((g1, g2, wave), axis=0)


def seam_band(dirs: np.ndarray, tree: cKDTree) -> np.ndarray:
    """0..1 coverage of the seam channel for unit directions (n, 3)."""
    d, _ = tree.query(dirs, k=1, workers=-1)
    ang = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
    return np.clip((SEAM_HALF_WIDTH_RAD - ang) / (0.35 * SEAM_HALF_WIDTH_RAD) + 1.0, 0.0, 1.0)


def pebble_lookup(cells_x: int = PEBBLES_AROUND, seed: int = 7):
    """Hex-packed jittered pebble centres in (u, v) texture space; returns a
    function mapping (u, v) arrays to -1 (gap) .. +1 (crown)."""
    rng = np.random.default_rng(seed)
    cell = 1.0 / cells_x
    row_h = cell * math.sqrt(3) / 2
    ny = int(round(0.5 / row_h))
    cx, cy = [], []
    for row in range(-1, ny + 2):
        y0 = row * row_h
        offset = 0.5 * cell if row % 2 else 0.0
        xs = np.arange(0, cells_x) * cell + offset + rng.uniform(-0.18, 0.18, cells_x) * cell
        ys = y0 + rng.uniform(-0.18, 0.18, cells_x) * cell
        for shift in (-1.0, 0.0, 1.0):
            cx.append(xs + shift)
            cy.append(ys)
    tree = cKDTree(np.stack((np.concatenate(cx), np.concatenate(cy)), axis=-1))

    def field(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # v in 0..0.5 lattice (equirect rows are stretched toward the poles; keep the
        # lattice in a 2:1 aspect by using v/2 with cell = 1/cells_x)
        d, _ = tree.query(np.stack((u, v * 0.5), axis=-1), k=1, workers=-1)
        r = d / (0.55 * cell)
        return np.clip(1.0 - r * r, -1.0, 1.0)

    return field


def make_texture(tree: cKDTree, field, w: int = 2048, h: int = 1024) -> np.ndarray:
    v = (np.arange(h) + 0.5) / h
    u = (np.arange(w) + 0.5) / w
    lat = (v - 0.5) * math.pi
    lon = (u - 0.5) * 2 * math.pi
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dirs = np.stack((np.cos(LAT) * np.cos(LON), np.cos(LAT) * np.sin(LON), np.sin(LAT)), axis=-1).reshape(-1, 3)
    seam = seam_band(dirs, tree).reshape(h, w)
    U, V = np.meshgrid(u, v)
    bumps = field(U.ravel(), V.ravel()).reshape(h, w)
    speck = np.random.default_rng(3).random((h, w)) - 0.5
    # the equirectangular pebble field compresses into radial streaks near the
    # poles: fade the grain out within 8 deg of them (smooth orange there)
    polar_fade = np.clip((math.radians(90.0) - np.abs(LAT)) / math.radians(8.0), 0.0, 1.0)
    shade = 1.0 + (0.10 * bumps + 0.02 * speck) * polar_fade
    rgb = ORANGE[None, None, :] * shade[..., None]
    # polar caps: every seam meets at the poles, so the last 1.5 deg are solid
    # channel colour (the mesh's pole fans otherwise sample a pinprick of orange)
    cap = (np.abs(LAT) > math.radians(85.5)).astype(np.float32)  # 4.5 deg = 9 mm radius: the channel junction  # 3.5 deg = 7 mm radius, the channel junction
    seam = np.maximum(seam, cap)
    rgb = rgb * (1.0 - seam[..., None]) + CHANNEL[None, None, :] * seam[..., None]
    return np.clip(rgb, 0.0, 1.0)


def make_mesh(radius: float, tree: cKDTree, field, segments: int = 400, rings: int = 200) -> str:
    us = np.arange(segments + 1) / segments
    vs = np.arange(rings + 1) / rings
    U, V = np.meshgrid(us, vs)  # (rings+1, segments+1)
    lat = (V - 0.5) * math.pi
    lon = (U - 0.5) * 2 * math.pi
    n = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1).reshape(-1, 3)
    seam = seam_band(n, tree)
    bumps = field(U.ravel(), V.ravel())
    disp = (PEBBLE_MM * 0.5 * (bumps + 1.0) - SEAM_DEPTH_MM * seam) * 1e-3
    disp = np.where(seam > 0.5, -SEAM_DEPTH_MM * 1e-3 * seam, disp)  # channels are smooth-bottomed
    # Poles (user, 2026-09-07: a speckled "hole" where the seams meet): every
    # seam runs through the poles, so the channel depth would dig a pit there
    # and the lat-long fan gives garbage finite-difference normals.  Fade the
    # displacement out within 4 deg of the poles (paint stays) and use the
    # radial normal there.
    polar = np.abs(lat.reshape(-1))
    pole_fade = np.clip((math.radians(90.0) - polar) / math.radians(8.0), 0.0, 1.0)
    disp = disp * pole_fade
    r = radius + disp
    verts = n * r[:, None]
    lines = ["# basketball: pebbled, channelled sphere"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    # MuJoCo samples OBJ texcoords with v flipped and, on this lat-long mapping,
    # u mirrored (diagnostic render 2026-09-06): write (1-u, 1-v) so the painted
    # seams land on the geometric grooves.
    lines += [f"vt {1.0 - u:.6f} {1.0 - v:.6f}" for u, v in zip(U.ravel(), V.ravel())]
    # normals from the displaced surface (finite differences on the grid)
    P = verts.reshape(rings + 1, segments + 1, 3)
    du = np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)
    dv = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    nn = np.cross(du, dv)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-12
    flip = np.sign((nn * P).sum(-1, keepdims=True))
    nn = (nn * flip).reshape(-1, 3)
    w = pole_fade[:, None]
    nn = w * nn + (1.0 - w) * n
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True) + 1e-12
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in nn]
    # Pole fans: one constant texcoord per fan (the cap colour), otherwise the
    # fan triangles span the whole u range and the texture mip filter smears
    # orange into the cap (user 2026-09-07: "hole" at the poles).
    n_vt = len(U.ravel())
    lines.append("vt 0.500000 1.000000")  # south cap (v = 0 -> flipped 1)
    lines.append("vt 0.500000 0.000000")  # north cap
    vt_s, vt_n = n_vt + 1, n_vt + 2
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j + 1
            b = a + 1
            c = a + segments + 1
            d = c + 1
            if i > 0:
                if i == rings - 1:
                    lines.append(f"f {a}/{vt_n}/{a} {b}/{vt_n}/{b} {d}/{vt_n}/{d}")
                else:
                    lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {d}/{d}/{d}")
            if i < rings - 1:
                if i == 0:
                    lines.append(f"f {a}/{vt_s}/{a} {d}/{vt_s}/{d} {c}/{vt_s}/{c}")
                else:
                    lines.append(f"f {a}/{a}/{a} {d}/{d}/{d} {c}/{c}/{c}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=0.1207)
    ap.add_argument("--segments", type=int, default=400)
    ap.add_argument("--rings", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tree = cKDTree(seam_samples())
    field = pebble_lookup()
    rgb = make_texture(tree, field)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(OUT / "basketball.png")
    Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 256)).save(OUT / "basketball_preview.png")
    (OUT / "basketball.obj").write_text(make_mesh(args.radius, tree, field, args.segments, args.rings))
    print("BASKETBALL_ASSETS", OUT)


if __name__ == "__main__":
    main()

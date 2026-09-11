"""Generate convex collision pieces for the concave retained swing seat."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import trimesh

HULL_COUNT = 64
VHACD_OPTIONS = {
    "maxConvexHulls": HULL_COUNT,
    "resolution": 300_000,
    "minimumVolumePercentErrorAllowed": 0.25,
    "maxRecursionDepth": 12,
    "shrinkWrap": True,
    "fillMode": "flood",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_dir = root / "src/mjlab_microduck/robot/microduck/assets"
    source = asset_dir / "swing_seat_retained.stl"
    mesh = trimesh.load_mesh(source, process=True)
    hulls = mesh.convex_decomposition(**VHACD_OPTIONS)
    if isinstance(hulls, trimesh.Trimesh):
        hulls = [hulls]
    hulls = sorted(
        hulls,
        key=lambda hull: (
            round(float(hull.centroid[2]), 9),
            round(float(hull.centroid[1]), 9),
            round(float(hull.centroid[0]), 9),
            round(float(hull.volume), 12),
        ),
    )
    if len(hulls) != HULL_COUNT:
        raise RuntimeError(f"expected {HULL_COUNT} hulls, got {len(hulls)}")

    records = []
    for index, hull in enumerate(hulls):
        path = asset_dir / f"swing_seat_collision_hull_{index:02d}.stl"
        hull.export(path)
        records.append(
            {
                "index": index,
                "file": path.name,
                "faces": len(hull.faces),
                "volume_m3": float(hull.volume),
                "bounds_m": hull.bounds.tolist(),
            }
        )

    manifest = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "generator": "trimesh.convex_decomposition via vhacdx",
        "options": VHACD_OPTIONS,
        "hull_count": len(hulls),
        "source_volume_m3": float(mesh.volume),
        "hull_volume_m3": float(sum(hull.volume for hull in hulls)),
        "hulls": records,
    }
    manifest_path = asset_dir / "swing_seat_collision_hulls.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "hulls"}, indent=2
        )
    )


if __name__ == "__main__":
    main()

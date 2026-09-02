# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fast-simplification>=0.1.12",
#   "manifold3d>=3.2",
#   "matplotlib>=3.9",
#   "mujoco>=3.3",
#   "numpy>=2.0",
#   "trimesh>=4.8",
# ]
# ///
"""Generate Microduck replacement-sole stilt parts and a visual preview.

The existing left/right sole STL is retained as the exact foot interface.  A
flat underside and two captive-M3-nut pockets are added.  Height cartridges
then bolt on without modifying the robot.

Run from the repository root:

    uv run hardware/stilts/generate_stilts.py
    uv run hardware/stilts/generate_stilts.py --heights 5 8 12 16 20
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "src/mjlab_microduck/robot/microduck/assets"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"

# All dimensions below are millimetres.
SOLE_PLATE_WIDTH = 39.5
SOLE_PLATE_LENGTH = 52.5
SOLE_PLATE_CORNER_RADIUS = 6.5
SOLE_PLATE_EXTENSION = 3.0
SOLE_PLATE_OVERLAP = 2.0

MOUNT_SPACING = 30.0
M3_CLEARANCE_DIAMETER = 3.4
M3_NUT_ACROSS_FLATS = 5.8
M3_NUT_POCKET_BOTTOM = 2.0
M3_NUT_POCKET_TOP = 6.6

POD_FLANGE_WIDTH = 38.0
POD_FLANGE_LENGTH = 50.0
POD_FLANGE_THICKNESS = 3.0
POD_FLANGE_RADIUS = 5.5
POD_BODY_TOP_WIDTH = 23.0
POD_BODY_TOP_LENGTH = 41.0
SCREW_HEAD_DIAMETER = 6.2
SCREW_HEAD_DEPTH = 2.0


def _rounded_rectangle_loop(
    width: float, length: float, radius: float, points_per_corner: int = 10
) -> np.ndarray:
    """Return a counter-clockwise rounded-rectangle perimeter."""
    radius = min(radius, width / 2.0 - 0.01, length / 2.0 - 0.01)
    centers = [
        (width / 2 - radius, length / 2 - radius, 0.0),
        (-width / 2 + radius, length / 2 - radius, 90.0),
        (-width / 2 + radius, -length / 2 + radius, 180.0),
        (width / 2 - radius, -length / 2 + radius, 270.0),
    ]
    points: list[tuple[float, float]] = []
    for cx, cy, start_deg in centers:
        for angle in np.linspace(start_deg, start_deg + 90.0, points_per_corner, endpoint=False):
            radians = math.radians(float(angle))
            points.append((cx + radius * math.cos(radians), cy + radius * math.sin(radians)))
    return np.asarray(points)


def _rounded_loft(
    bottom_width: float,
    bottom_length: float,
    bottom_radius: float,
    top_width: float,
    top_length: float,
    top_radius: float,
    z_bottom: float,
    z_top: float,
) -> trimesh.Trimesh:
    """Create a watertight loft between two rounded rectangles."""
    bottom = _rounded_rectangle_loop(bottom_width, bottom_length, bottom_radius)
    top = _rounded_rectangle_loop(top_width, top_length, top_radius)
    count = len(bottom)
    vertices = np.vstack(
        [
            np.column_stack([bottom, np.full(count, z_bottom)]),
            np.column_stack([top, np.full(count, z_top)]),
            [[0.0, 0.0, z_bottom], [0.0, 0.0, z_top]],
        ]
    )
    bottom_center = 2 * count
    top_center = bottom_center + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        nxt = (index + 1) % count
        faces.extend(
            [
                (index, nxt, count + nxt),
                (index, count + nxt, count + index),
                (bottom_center, nxt, index),
                (top_center, count + index, count + nxt),
            ]
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    if not mesh.is_watertight:
        raise RuntimeError("Rounded loft unexpectedly produced a non-watertight mesh")
    return mesh


def _circular_frustum(
    bottom_diameter: float,
    top_diameter: float,
    z_bottom: float,
    z_top: float,
    sections: int = 64,
) -> trimesh.Trimesh:
    """Create a watertight circular frustum aligned with the Z axis."""
    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    bottom = np.column_stack(
        [
            bottom_diameter / 2.0 * np.cos(angles),
            bottom_diameter / 2.0 * np.sin(angles),
            np.full(sections, z_bottom),
        ]
    )
    top = np.column_stack(
        [
            top_diameter / 2.0 * np.cos(angles),
            top_diameter / 2.0 * np.sin(angles),
            np.full(sections, z_top),
        ]
    )
    vertices = np.vstack([bottom, top, [[0.0, 0.0, z_bottom], [0.0, 0.0, z_top]]])
    bottom_center = 2 * sections
    top_center = bottom_center + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(sections):
        nxt = (index + 1) % sections
        faces.extend(
            [
                (index, nxt, sections + nxt),
                (index, sections + nxt, sections + index),
                (bottom_center, nxt, index),
                (top_center, sections + index, sections + nxt),
            ]
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    if not mesh.is_watertight:
        raise RuntimeError("Circular frustum unexpectedly produced a non-watertight mesh")
    return mesh


def _cylinder(radius: float, height: float, center: tuple[float, float, float], sections: int = 48) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections, transform=transform)


def _boolean_union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    result = trimesh.boolean.union(meshes, engine="manifold", check_volume=True)
    if not isinstance(result, trimesh.Trimesh):
        raise TypeError("Boolean union did not produce a single mesh")
    return result


def _boolean_difference(mesh: trimesh.Trimesh, cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    cutter = _boolean_union(cutters)
    result = trimesh.boolean.difference([mesh, cutter], engine="manifold", check_volume=True)
    if not isinstance(result, trimesh.Trimesh):
        raise TypeError("Boolean difference did not produce a single mesh")
    return result


def _load_normalized_part(name: str, xy_center: np.ndarray, plate_bottom_raw: float) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(ASSET_DIR / f"{name}.stl")
    mesh.apply_scale(1000.0)
    mesh.apply_translation([-xy_center[0], -xy_center[1], -plate_bottom_raw])
    return mesh


def make_carrier(
    side: str,
    *,
    cartridge_interface: bool = True,
) -> tuple[trimesh.Trimesh, dict[str, np.ndarray | float]]:
    """Make one exact-fit replacement sole, optionally with cartridge holes."""
    source = trimesh.load_mesh(ASSET_DIR / f"sole_{side}.stl")
    source.apply_scale(1000.0)
    bounds = source.bounds
    xy_center = bounds.mean(axis=0)[:2]
    plate_bottom_raw = float(bounds[0, 2] - SOLE_PLATE_EXTENSION)
    plate_top_raw = float(bounds[0, 2] + SOLE_PLATE_OVERLAP)

    sole = source.copy()
    sole.apply_translation([-xy_center[0], -xy_center[1], -plate_bottom_raw])
    plate = _rounded_loft(
        SOLE_PLATE_WIDTH,
        SOLE_PLATE_LENGTH,
        SOLE_PLATE_CORNER_RADIUS,
        SOLE_PLATE_WIDTH,
        SOLE_PLATE_LENGTH,
        SOLE_PLATE_CORNER_RADIUS,
        0.0,
        plate_top_raw - plate_bottom_raw,
    )
    carrier = _boolean_union([sole, plate])

    if cartridge_interface:
        cutters: list[trimesh.Trimesh] = []
        for x in (-MOUNT_SPACING / 2.0, MOUNT_SPACING / 2.0):
            cutters.append(
                _cylinder(M3_CLEARANCE_DIAMETER / 2.0, 10.0, (x, 0.0, 4.0))
            )
            # Circumradius corresponding to the specified hexagon across-flats size.
            hex_radius = M3_NUT_ACROSS_FLATS / math.sqrt(3.0)
            pocket_height = M3_NUT_POCKET_TOP - M3_NUT_POCKET_BOTTOM
            cutters.append(
                _cylinder(
                    hex_radius,
                    pocket_height,
                    (x, 0.0, M3_NUT_POCKET_BOTTOM + pocket_height / 2.0),
                    sections=6,
                )
            )
        carrier = _boolean_difference(carrier, cutters)
    carrier.remove_unreferenced_vertices()
    metadata: dict[str, np.ndarray | float] = {
        "xy_center": xy_center,
        "plate_bottom_raw": plate_bottom_raw,
    }
    return carrier, metadata


def _make_mounting_hole_cutters(height: float) -> list[trimesh.Trimesh]:
    cutters: list[trimesh.Trimesh] = []
    flange_bottom = height - POD_FLANGE_THICKNESS
    for x in (-MOUNT_SPACING / 2.0, MOUNT_SPACING / 2.0):
        cutters.append(_cylinder(M3_CLEARANCE_DIAMETER / 2.0, POD_FLANGE_THICKNESS + 2.0, (x, 0.0, height - POD_FLANGE_THICKNESS / 2.0)))
        cutters.append(
            _cylinder(
                SCREW_HEAD_DIAMETER / 2.0,
                SCREW_HEAD_DEPTH + 0.2,
                (x, 0.0, flange_bottom + SCREW_HEAD_DEPTH / 2.0 - 0.05),
            )
        )
    return cutters


def make_pod(height: float, tip_width: float, tip_length: float) -> trimesh.Trimesh:
    """Make a fixed-height stilt cartridge; height is sole-to-ground distance."""
    if height < 5.0:
        raise ValueError("Pod height must be at least 5 mm")
    if not 8.0 <= tip_width <= POD_BODY_TOP_WIDTH:
        raise ValueError(f"Tip width must be between 8 and {POD_BODY_TOP_WIDTH:g} mm")
    if not 12.0 <= tip_length <= POD_BODY_TOP_LENGTH:
        raise ValueError(f"Tip length must be between 12 and {POD_BODY_TOP_LENGTH:g} mm")

    flange_bottom = height - POD_FLANGE_THICKNESS
    flange = _rounded_loft(
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        flange_bottom,
        height,
    )
    body = _rounded_loft(
        tip_width,
        tip_length,
        min(3.5, tip_width / 2.0 - 0.2, tip_length / 2.0 - 0.2),
        POD_BODY_TOP_WIDTH,
        POD_BODY_TOP_LENGTH,
        4.5,
        0.0,
        flange_bottom + 0.25,
    )
    pod = _boolean_union([flange, body])
    pod = _boolean_difference(pod, _make_mounting_hole_cutters(height))
    pod.remove_unreferenced_vertices()
    return pod


def make_peg_pod(height: float, peg_diameter: float, root_diameter: float) -> trimesh.Trimesh:
    """Make a narrow, circular peg-style cartridge."""
    if height < 8.0:
        raise ValueError("Peg height must be at least 8 mm")
    if not 8.0 <= peg_diameter <= 16.0:
        raise ValueError("Peg diameter must be between 8 and 16 mm")
    minimum_root = peg_diameter + 4.0
    if not minimum_root <= root_diameter <= POD_BODY_TOP_WIDTH:
        raise ValueError(f"Peg root diameter must be between {minimum_root:g} and {POD_BODY_TOP_WIDTH:g} mm")

    flange_bottom = height - POD_FLANGE_THICKNESS
    flange = _rounded_loft(
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        flange_bottom,
        height,
    )
    peg = _circular_frustum(
        peg_diameter,
        root_diameter,
        0.0,
        flange_bottom + 0.25,
    )
    pod = _boolean_union([flange, peg])
    pod = _boolean_difference(pod, _make_mounting_hole_cutters(height))
    pod.remove_unreferenced_vertices()
    return pod


def make_transition_pod(
    height: float,
    blend: float,
    wide_tip_width: float,
    wide_tip_length: float,
    peg_diameter: float,
    peg_root_diameter: float,
    *,
    mounting_holes: bool = True,
) -> trimesh.Trimesh:
    """Morph continuously from the rounded platform pod to the circular peg."""
    if height < 8.0:
        raise ValueError("Transition pod height must be at least 8 mm")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Transition blend must be between 0 (platform) and 1 (peg)")

    def lerp(start: float, end: float) -> float:
        return start + blend * (end - start)

    tip_width = lerp(wide_tip_width, peg_diameter)
    tip_length = lerp(wide_tip_length, peg_diameter)
    tip_radius = lerp(min(3.5, wide_tip_width / 2.0 - 0.2), peg_diameter / 2.0)
    root_width = lerp(POD_BODY_TOP_WIDTH, peg_root_diameter)
    root_length = lerp(POD_BODY_TOP_LENGTH, peg_root_diameter)
    root_radius = lerp(4.5, peg_root_diameter / 2.0)

    flange_bottom = height - POD_FLANGE_THICKNESS
    flange = _rounded_loft(
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        flange_bottom,
        height,
    )
    body = _rounded_loft(
        tip_width,
        tip_length,
        tip_radius,
        root_width,
        root_length,
        root_radius,
        0.0,
        flange_bottom + 0.25,
    )
    pod = _boolean_union([flange, body])
    if mounting_holes:
        pod = _boolean_difference(pod, _make_mounting_hole_cutters(height))
    pod.remove_unreferenced_vertices()
    return pod


def make_direct_transition_stilt(
    side: str,
    height: float,
    blend: float,
    wide_tip_width: float = 22.0,
    wide_tip_length: float = 32.0,
    peg_diameter: float = 12.0,
    peg_root_diameter: float = 22.0,
    support_overhang_degrees: float | None = None,
) -> trimesh.Trimesh:
    """Fuse a side-specific replacement sole directly to one transition stilt."""
    carrier, _ = make_carrier(side, cartridge_interface=False)
    carrier.apply_translation((0.0, 0.0, height))
    if support_overhang_degrees is None:
        pod = make_transition_pod(
            height,
            blend,
            wide_tip_width,
            wide_tip_length,
            peg_diameter,
            peg_root_diameter,
            mounting_holes=False,
        )
    else:
        if not 0.0 < support_overhang_degrees <= 60.0:
            raise ValueError("Support overhang must be in (0, 60] degrees")

        def lerp(start: float, end: float) -> float:
            return start + blend * (end - start)

        tip_width = lerp(wide_tip_width, peg_diameter)
        tip_length = lerp(wide_tip_length, peg_diameter)
        tip_radius = lerp(
            min(3.5, wide_tip_width / 2.0 - 0.2),
            peg_diameter / 2.0,
        )
        root_width = lerp(POD_BODY_TOP_WIDTH, peg_root_diameter)
        root_length = lerp(POD_BODY_TOP_LENGTH, peg_root_diameter)
        root_radius = lerp(4.5, peg_root_diameter / 2.0)

        root_loop = _rounded_rectangle_loop(root_width, root_length, root_radius)
        sole_loop = _rounded_rectangle_loop(
            SOLE_PLATE_WIDTH,
            SOLE_PLATE_LENGTH,
            SOLE_PLATE_CORNER_RADIUS,
        )
        segment_start = root_loop
        segment_end = np.roll(root_loop, -1, axis=0)
        segment_vector = segment_end - segment_start
        segment_length_sq = np.sum(segment_vector * segment_vector, axis=1)
        max_horizontal_offset = 0.0
        for point in sole_loop:
            projection = np.clip(
                np.sum((point - segment_start) * segment_vector, axis=1)
                / segment_length_sq,
                0.0,
                1.0,
            )
            closest = segment_start + projection[:, None] * segment_vector
            distance = np.sqrt(np.sum((point - closest) ** 2, axis=1)).min()
            max_horizontal_offset = max(max_horizontal_offset, float(distance))

        chamfer_rise = max_horizontal_offset / math.tan(
            math.radians(support_overhang_degrees)
        )
        # Give the generated facets a small margin below the requested limit
        # and overlap adjacent solids so the manifold union is unambiguous.
        chamfer_rise += 0.25
        if height <= chamfer_rise + 1.0:
            raise ValueError("Stilt is too short for the requested support-free chamfer")
        chamfer_bottom = height - chamfer_rise
        body = _rounded_loft(
            tip_width,
            tip_length,
            tip_radius,
            root_width,
            root_length,
            root_radius,
            0.0,
            chamfer_bottom,
        )
        shoulder = _rounded_loft(
            root_width,
            root_length,
            root_radius,
            SOLE_PLATE_WIDTH,
            SOLE_PLATE_LENGTH,
            SOLE_PLATE_CORNER_RADIUS,
            chamfer_bottom,
            height,
        )
        pod = _boolean_union([body, shoulder])

    direct_stilt = _boolean_union([pod, carrier])
    direct_stilt.remove_unreferenced_vertices()
    return direct_stilt


def make_bumper_adapter(height: float = 5.0) -> trimesh.Trimesh:
    """Make a cartridge accepting one central M3 male rubber bobbin."""
    flange = _rounded_loft(
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        POD_FLANGE_WIDTH,
        POD_FLANGE_LENGTH,
        POD_FLANGE_RADIUS,
        height - POD_FLANGE_THICKNESS,
        height,
    )
    boss = _rounded_loft(18.0, 24.0, 4.0, 18.0, 24.0, 4.0, 0.0, height - POD_FLANGE_THICKNESS + 0.25)
    adapter = _boolean_union([flange, boss])
    cutters = _make_mounting_hole_cutters(height)
    # 4.2 mm is a conservative pilot for a common M3 heat-set insert. Measure
    # the chosen insert and override this in source before production printing.
    cutters.append(_cylinder(2.1, 4.2, (0.0, 0.0, 2.0)))
    adapter = _boolean_difference(adapter, cutters)
    adapter.remove_unreferenced_vertices()
    return adapter


def _add_mesh(ax, mesh: trimesh.Trimesh, color: str, alpha: float = 1.0) -> None:
    collection = Poly3DCollection(mesh.triangles, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(collection)


def _set_equal_bounds(ax, meshes: list[trimesh.Trimesh], pad: float = 1.12) -> None:
    bounds = np.vstack([mesh.bounds for mesh in meshes])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    center = (low + high) / 2.0
    half_span = (high - low).max() * pad / 2.0
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def render_preview(
    output_path: Path,
    carrier: trimesh.Trimesh,
    carrier_meta: dict[str, np.ndarray | float],
    pods: list[tuple[float, trimesh.Trimesh]],
    *,
    concept_title: str = "Microduck modular stilt concept",
    progression_title: str = "Same mount and footprint, increasing height",
    contact_note: str = "two captive M3 nuts per foot",
    preview_target_height: float = 15.0,
) -> None:
    """Render an assembled foot and the generated height progression."""
    preview_height, preview_pod = min(pods, key=lambda item: abs(item[0] - preview_target_height))
    xy_center = np.asarray(carrier_meta["xy_center"])
    plate_bottom_raw = float(carrier_meta["plate_bottom_raw"])
    foot = _load_normalized_part("foot_left", xy_center, plate_bottom_raw)
    ankle = _load_normalized_part("ankle_left", xy_center, plate_bottom_raw)

    assembled_carrier = carrier.copy()
    assembled_carrier.apply_translation([0.0, 0.0, preview_height])
    foot.apply_translation([0.0, 0.0, preview_height])
    ankle.apply_translation([0.0, 0.0, preview_height])

    fig = plt.figure(figsize=(14, 8), dpi=180, facecolor="#f6f7f9")
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#f6f7f9")
    _add_mesh(ax, preview_pod, "#7357d8")
    _add_mesh(ax, assembled_carrier, "#55cec3", 0.92)
    _add_mesh(ax, foot, "#f4b51d", 0.9)
    _add_mesh(ax, ankle, "#f4b51d", 0.9)
    scene_meshes = [preview_pod, assembled_carrier, foot, ankle]
    _set_equal_bounds(ax, scene_meshes)
    ax.view_init(elev=24, azim=-43)
    ax.set_title(f"Assembled left foot — {preview_height:g} mm cartridge", fontsize=15, pad=12)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#f6f7f9")
    spaced: list[trimesh.Trimesh] = []
    x_step = 48.0
    x_origin = -x_step * (len(pods) - 1) / 2.0
    for index, (height, pod) in enumerate(pods):
        shifted = pod.copy()
        shifted.apply_translation([x_origin + index * x_step, 0.0, 0.0])
        spaced.append(shifted)
        _add_mesh(ax2, shifted, "#7357d8")
        ax2.text(x_origin + index * x_step, -31.0, 0.0, f"{height:g} mm", ha="center", fontsize=10)
    bounds = np.vstack([mesh.bounds for mesh in spaced])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    ax2.set_xlim(low[0] - 6.0, high[0] + 6.0)
    ax2.set_ylim(low[1] - 5.0, high[1] + 5.0)
    ax2.set_zlim(-1.0, high[2] + 3.0)
    ax2.set_box_aspect((2.4, 1.0, 0.75))
    ax2.set_proj_type("ortho")
    ax2.set_axis_off()
    ax2.view_init(elev=18, azim=-58)
    ax2.set_title(progression_title, fontsize=15, pad=12)

    fig.suptitle(concept_title, fontsize=20, y=0.97)
    fig.text(
        0.5,
        0.035,
        f"cyan = exact replacement sole  •  purple = rigid replaceable cartridge  •  {contact_note}",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_transition_preview(
    output_path: Path,
    carrier: trimesh.Trimesh,
    carrier_meta: dict[str, np.ndarray | float],
    transition_pods: list[tuple[float, trimesh.Trimesh]],
    height: float,
    wide_tip_width: float,
    wide_tip_length: float,
    peg_diameter: float,
) -> None:
    """Render the fixed-height platform-to-peg support-shape progression."""
    middle_blend, middle_pod = min(transition_pods, key=lambda item: abs(item[0] - 0.5))
    xy_center = np.asarray(carrier_meta["xy_center"])
    plate_bottom_raw = float(carrier_meta["plate_bottom_raw"])
    foot = _load_normalized_part("foot_left", xy_center, plate_bottom_raw)
    ankle = _load_normalized_part("ankle_left", xy_center, plate_bottom_raw)
    assembled_carrier = carrier.copy()
    for mesh in (assembled_carrier, foot, ankle):
        mesh.apply_translation([0.0, 0.0, height])

    fig = plt.figure(figsize=(15, 8), dpi=180, facecolor="#f6f7f9")
    ax = fig.add_subplot(1, 2, 1, projection="3d", facecolor="#f6f7f9")
    _add_mesh(ax, middle_pod, "#7357d8")
    _add_mesh(ax, assembled_carrier, "#55cec3", 0.92)
    _add_mesh(ax, foot, "#f4b51d", 0.9)
    _add_mesh(ax, ankle, "#f4b51d", 0.9)
    _set_equal_bounds(ax, [middle_pod, assembled_carrier, foot, ankle])
    ax.view_init(elev=24, azim=-43)
    ax.set_title(f"Mid-transition support — {middle_blend:.0%} blend", fontsize=15, pad=12)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#f6f7f9")
    spaced: list[trimesh.Trimesh] = []
    x_step = 43.0
    x_origin = -x_step * (len(transition_pods) - 1) / 2.0
    for index, (blend, pod) in enumerate(transition_pods):
        shifted = pod.copy()
        shifted.apply_translation([x_origin + index * x_step, 0.0, 0.0])
        spaced.append(shifted)
        _add_mesh(ax2, shifted, "#7357d8")
        tip_width = wide_tip_width + blend * (peg_diameter - wide_tip_width)
        tip_length = wide_tip_length + blend * (peg_diameter - wide_tip_length)
        shape = f"Ø{peg_diameter:g}" if math.isclose(blend, 1.0) else f"{tip_width:g}×{tip_length:g}"
        ax2.text(
            x_origin + index * x_step,
            -31.0,
            0.0,
            f"{blend:.0%}\n{shape} mm",
            ha="center",
            fontsize=9,
        )
    bounds = np.vstack([mesh.bounds for mesh in spaced])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    ax2.set_xlim(low[0] - 5.0, high[0] + 5.0)
    ax2.set_ylim(low[1] - 5.0, high[1] + 5.0)
    ax2.set_zlim(-1.0, high[2] + 3.0)
    ax2.set_box_aspect((2.7, 1.0, 0.75))
    ax2.set_proj_type("ortho")
    ax2.set_axis_off()
    ax2.view_init(elev=18, azim=-58)
    ax2.set_title(f"Support shape changes; height stays fixed at {height:g} mm", fontsize=15, pad=12)

    fig.suptitle("Microduck platform-to-peg curriculum cartridges", fontsize=20, y=0.97)
    fig.text(
        0.5,
        0.035,
        "one mounting interface  •  fixed height  •  progressively smaller and rounder support polygon",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_robot_preview(
    output_path: Path,
    pod: trimesh.Trimesh,
    height: float,
    carrier_metadata: dict[str, dict[str, np.ndarray | float]],
    *,
    title: str | None = None,
    contact_note: str = "22 × 32 mm contact patch shown",
) -> None:
    """Render the selected cartridge on the complete robot in its stand pose."""
    robot_dir = ASSET_DIR.parent
    with tempfile.TemporaryDirectory(prefix="microduck_stilt_preview_") as temp_name:
        temp_dir = Path(temp_name)
        tree = ET.parse(robot_dir / "robot_walk.xml")
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            raise RuntimeError("robot_walk.xml has no compiler element")
        compiler.set("meshdir", str(ASSET_DIR.resolve()))
        asset = root.find("asset")
        if asset is None:
            raise RuntimeError("robot_walk.xml has no asset element")
        ET.SubElement(asset, "material", name="stilt_preview_material", rgba="0.45 0.34 0.85 1")

        for side in ("left", "right"):
            raw_pod = pod.copy()
            raw_pod.apply_scale(0.001)
            metadata = carrier_metadata[side]
            xy_center = np.asarray(metadata["xy_center"])
            plate_bottom_raw = float(metadata["plate_bottom_raw"])
            raw_pod.apply_translation(
                [xy_center[0] / 1000.0, xy_center[1] / 1000.0, (plate_bottom_raw - height) / 1000.0]
            )
            mesh_path = temp_dir / f"stilt_{side}.stl"
            raw_pod.export(mesh_path)
            ET.SubElement(asset, "mesh", name=f"stilt_{side}_preview", file=str(mesh_path))

            ankle = next(element for element in root.iter("body") if element.get("name") == f"ankle_{side}")
            sole = next(
                element
                for element in ankle.findall("geom")
                if element.get("mesh") == f"sole_{side}" and element.get("class") == "visual"
            )
            ET.SubElement(
                ankle,
                "geom",
                type="mesh",
                **{"class": "visual"},
                pos=sole.get("pos", "0 0 0"),
                quat=sole.get("quat", "1 0 0 0"),
                mesh=f"stilt_{side}_preview",
                material="stilt_preview_material",
            )

        robot_path = temp_dir / "robot_walk.xml"
        tree.write(robot_path)
        shutil.copy(robot_dir / "scene_walk.xml", temp_dir / "scene_walk.xml")
        model = mujoco.MjModel.from_xml_path(str(temp_dir / "scene_walk.xml"))
        data = mujoco.MjData(model)
        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        # The replacement plate adds about 3 mm below the original sole.
        data.qpos[2] += (height + SOLE_PLATE_EXTENSION) / 1000.0
        mujoco.mj_forward(model, data)

        fig = plt.figure(figsize=(8, 10), dpi=180, facecolor="#f6f7f9")
        ax = fig.add_subplot(111, projection="3d", facecolor="#f6f7f9")
        world_meshes: list[trimesh.Trimesh] = []
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[geom_id] != 2:
                continue
            mesh_id = model.geom_dataid[geom_id]
            vertex_start = model.mesh_vertadr[mesh_id]
            vertex_count = model.mesh_vertnum[mesh_id]
            face_start = model.mesh_faceadr[mesh_id]
            face_count = model.mesh_facenum[mesh_id]
            vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count].copy()
            faces = model.mesh_face[face_start : face_start + face_count].copy()
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if len(mesh.faces) > 2600:
                mesh = mesh.simplify_quadric_decimation(face_count=1300)
            world_meshes.append(mesh)
            material_id = model.geom_matid[geom_id]
            rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
            _add_mesh(ax, mesh, rgba)

        bounds = np.vstack([mesh.bounds for mesh in world_meshes])
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        span = (high - low) * 1.16
        center = (low + high) / 2.0
        ax.set_xlim(center[0] - span[0] / 2.0, center[0] + span[0] / 2.0)
        ax.set_ylim(center[1] - span[1] / 2.0, center[1] + span[1] / 2.0)
        z_low = min(0.0, low[2] - 0.01)
        z_high = high[2] + 0.01
        ax.set_zlim(z_low, z_high)
        ax.set_box_aspect((span[0], span[1], z_high - z_low))
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.view_init(elev=8, azim=145)
        ax.set_title(title or f"Microduck with {height:g} mm stilts", fontsize=19, pad=15)
        ax.set_position([0.06, 0.11, 0.88, 0.80])
        fig.text(0.5, 0.04, contact_note, ha="center", fontsize=11, color="#374151")
        fig.savefig(output_path)
        plt.close(fig)


def _mesh_summary(mesh: trimesh.Trimesh) -> str:
    volume_cm3 = abs(float(mesh.volume)) / 1000.0
    petg_mass_g = volume_cm3 * 1.27
    return f"{volume_cm3:.2f} cm^3 (~{petg_mass_g:.1f} g solid PETG), watertight={mesh.is_watertight}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heights", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0])
    parser.add_argument("--tip-width", type=float, default=22.0)
    parser.add_argument("--tip-length", type=float, default=32.0)
    parser.add_argument(
        "--peg-heights",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0, 25.0, 50.0, 250.0],
    )
    parser.add_argument("--peg-diameter", type=float, default=12.0)
    parser.add_argument("--peg-root-diameter", type=float, default=22.0)
    parser.add_argument("--transition-height", type=float, default=20.0)
    parser.add_argument("--transition-blends", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--direct-heights",
        type=float,
        nargs="+",
        default=[100.0, 150.0, 200.0, 250.0, 500.0, 1000.0, 1400.0, 2000.0],
        help="Heights for released fused sole-and-stilt meshes (mm)",
    )
    parser.add_argument("--direct-blend", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    carriers: dict[str, trimesh.Trimesh] = {}
    carrier_metadata: dict[str, dict[str, np.ndarray | float]] = {}
    for side in ("left", "right"):
        carrier, metadata = make_carrier(side)
        carrier.export(args.output_dir / f"carrier_{side}.stl")
        carriers[side] = carrier
        carrier_metadata[side] = metadata
        print(f"carrier_{side}.stl: {_mesh_summary(carrier)}")

    pods: list[tuple[float, trimesh.Trimesh]] = []
    for height in sorted(set(args.heights)):
        pod = make_pod(height, args.tip_width, args.tip_length)
        name = f"pod_{height:04.1f}mm".replace(".", "p") + ".stl"
        pod.export(args.output_dir / name)
        pods.append((height, pod))
        print(f"{name}: {_mesh_summary(pod)}")

    bumper_adapter = make_bumper_adapter()
    bumper_adapter.export(args.output_dir / "pod_m3_rubber_bumper_adapter.stl")
    print(f"pod_m3_rubber_bumper_adapter.stl: {_mesh_summary(bumper_adapter)}")

    peg_pods: list[tuple[float, trimesh.Trimesh]] = []
    for height in sorted(set(args.peg_heights)):
        peg_pod = make_peg_pod(height, args.peg_diameter, args.peg_root_diameter)
        name = f"peg_{height:04.1f}mm_d{args.peg_diameter:04.1f}mm".replace(".", "p") + ".stl"
        peg_pod.export(args.output_dir / name)
        peg_pods.append((height, peg_pod))
        print(f"{name}: {_mesh_summary(peg_pod)}")

    transition_pods: list[tuple[float, trimesh.Trimesh]] = []
    for blend in sorted(set(args.transition_blends)):
        transition_pod = make_transition_pod(
            args.transition_height,
            blend,
            args.tip_width,
            args.tip_length,
            args.peg_diameter,
            args.peg_root_diameter,
        )
        name = (
            f"transition_b{blend:0.2f}_h{args.transition_height:04.1f}mm".replace(".", "p")
            + ".stl"
        )
        transition_pod.export(args.output_dir / name)
        transition_pods.append((blend, transition_pod))
        print(f"{name}: {_mesh_summary(transition_pod)}")

    release_dir = args.output_dir / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    for height in sorted(set(args.direct_heights)):
        released: dict[str, trimesh.Trimesh] = {}
        stem = f"b{args.direct_blend:0.2f}_h{height:04.1f}mm".replace(".", "p")
        for side in ("left", "right"):
            mesh = make_direct_transition_stilt(side, height, args.direct_blend)
            mesh.export(release_dir / f"direct_replacement_{side}_{stem}.stl")
            released[side] = mesh
        left = released["left"].copy()
        right = released["right"].copy()
        left.apply_translation((-25.0, 0.0, 0.0))
        right.apply_translation((25.0, 0.0, 0.0))
        pair = trimesh.util.concatenate([left, right])
        pair.export(release_dir / f"direct_replacement_pair_{stem}.stl")
        print(f"release/{stem}: left={_mesh_summary(released['left'])}; pair={_mesh_summary(pair)}")

    render_preview(args.output_dir / "preview.png", carriers["left"], carrier_metadata["left"], pods)
    print(f"preview.png: {args.output_dir / 'preview.png'}")
    robot_height, robot_pod = min(pods, key=lambda item: abs(item[0] - 15.0))
    render_robot_preview(args.output_dir / "robot_preview.png", robot_pod, robot_height, carrier_metadata)
    print(f"robot_preview.png: {args.output_dir / 'robot_preview.png'}")

    tallest_peg_height, tallest_peg = max(peg_pods, key=lambda item: item[0])
    render_preview(
        args.output_dir / "peg_preview.png",
        carriers["left"],
        carrier_metadata["left"],
        peg_pods,
        concept_title="Microduck narrow peg stilt concept",
        progression_title=f"Same {args.peg_diameter:g} mm circular tip, increasing height",
        contact_note=f"{args.peg_diameter:g} mm circular contact",
        preview_target_height=tallest_peg_height,
    )
    print(f"peg_preview.png: {args.output_dir / 'peg_preview.png'}")
    render_robot_preview(
        args.output_dir / "robot_peg_preview.png",
        tallest_peg,
        tallest_peg_height,
        carrier_metadata,
        title=f"Microduck wearing {tallest_peg_height:g} mm peg stilts",
        contact_note=f"Tallest generated version • {args.peg_diameter:g} mm circular contact",
    )
    print(f"robot_peg_preview.png: {args.output_dir / 'robot_peg_preview.png'}")
    render_transition_preview(
        args.output_dir / "transition_preview.png",
        carriers["left"],
        carrier_metadata["left"],
        transition_pods,
        args.transition_height,
        args.tip_width,
        args.tip_length,
        args.peg_diameter,
    )
    print(f"transition_preview.png: {args.output_dir / 'transition_preview.png'}")


if __name__ == "__main__":
    main()

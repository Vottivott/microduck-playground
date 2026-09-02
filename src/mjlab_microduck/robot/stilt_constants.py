"""Compile-time Microduck stilt morphology for locomotion training.

The CAD in ``hardware/stilts`` is the manufacturing concept.  This module
builds the corresponding support as a small, convex MuJoCo mesh so each
training stage has explicit contact geometry, mass, inertia, and foot sites.
The old sole remains visible but is removed from collision.
"""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial

import mujoco
from mjlab.entity import EntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG

MIN_STILT_HEIGHT_MM = 8.0
MAX_STILT_HEIGHT_MM = 3000.0
DEFAULT_STILT_MASS_BASE_KG = 0.012
DEFAULT_STILT_MASS_PER_MM_KG = 0.0001


def validate_stilt_morphology(height_mm: float, blend: float) -> None:
    """Validate the fixed morphology selected for one compiled task."""
    if not MIN_STILT_HEIGHT_MM <= height_mm <= MAX_STILT_HEIGHT_MM:
        raise ValueError(
            f"Stilt height must be in [{MIN_STILT_HEIGHT_MM:g}, "
            f"{MAX_STILT_HEIGHT_MM:g}] mm"
        )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Stilt blend must be between 0 (platform) and 1 (peg)")


def stilt_profile_dimensions(blend: float) -> dict[str, float]:
    """Interpolate the CAD profile from a platform to a circular peg."""
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Stilt blend must be between 0 and 1")

    def lerp(start: float, end: float) -> float:
        return start + blend * (end - start)

    return {
        "tip_width_mm": lerp(22.0, 12.0),
        "tip_length_mm": lerp(32.0, 12.0),
        "tip_radius_mm": lerp(3.5, 6.0),
        "root_width_mm": lerp(23.0, 22.0),
        "root_length_mm": lerp(41.0, 22.0),
        "root_radius_mm": lerp(4.5, 11.0),
    }


def default_stilt_mass_kg(height_mm: float) -> float:
    """Conservative stage mass: 14 g at 20 mm and 312 g at 3,000 mm."""
    return DEFAULT_STILT_MASS_BASE_KG + height_mm * DEFAULT_STILT_MASS_PER_MM_KG


def _rounded_rectangle_ring(
    width: float,
    length: float,
    radius: float,
    samples_per_corner: int = 8,
) -> list[tuple[float, float]]:
    """Return a counter-clockwise rounded-rectangle boundary in metres."""
    radius = min(radius, width / 2.0, length / 2.0)
    corners = (
        (width / 2.0 - radius, length / 2.0 - radius, 0.0),
        (-width / 2.0 + radius, length / 2.0 - radius, math.pi / 2.0),
        (-width / 2.0 + radius, -length / 2.0 + radius, math.pi),
        (width / 2.0 - radius, -length / 2.0 + radius, 3.0 * math.pi / 2.0),
    )
    ring: list[tuple[float, float]] = []
    for center_x, center_y, start_angle in corners:
        for sample in range(samples_per_corner):
            angle = start_angle + (math.pi / 2.0) * sample / samples_per_corner
            ring.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return ring


def stilt_mesh_data(
    height_mm: float,
    blend: float,
) -> tuple[list[float], list[int]]:
    """Build a watertight convex loft in the foot-site frame.

    The top is at z=0 and the support surface is at z=-height.  Values passed
    to MuJoCo are metres.
    """
    validate_stilt_morphology(height_mm, blend)
    dimensions = stilt_profile_dimensions(blend)
    millimetres = 0.001
    bottom_ring = _rounded_rectangle_ring(
        dimensions["tip_width_mm"] * millimetres,
        dimensions["tip_length_mm"] * millimetres,
        dimensions["tip_radius_mm"] * millimetres,
    )
    top_ring = _rounded_rectangle_ring(
        dimensions["root_width_mm"] * millimetres,
        dimensions["root_length_mm"] * millimetres,
        dimensions["root_radius_mm"] * millimetres,
    )
    count = len(bottom_ring)
    height = height_mm * millimetres
    vertices: list[tuple[float, float, float]] = [
        (x, y, -height) for x, y in bottom_ring
    ] + [(x, y, 0.0) for x, y in top_ring]
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))

    faces: list[tuple[int, int, int]] = []
    for current in range(count):
        following = (current + 1) % count
        faces.extend(
            (
                (current, following, count + following),
                (current, count + following, count + current),
                (bottom_center, following, current),
                (top_center, count + current, count + following),
            )
        )
    return [coordinate for vertex in vertices for coordinate in vertex], [
        index for face in faces for index in face
    ]


def get_stilt_walk_spec(
    height_mm: float = 20.0,
    blend: float = 0.0,
    mass_kg: float | None = None,
) -> mujoco.MjSpec:
    """Return the walk robot with explicit stilt contact and tip sites."""
    validate_stilt_morphology(height_mm, blend)
    if mass_kg is None:
        mass_kg = default_stilt_mass_kg(height_mm)
    if mass_kg <= 0.0:
        raise ValueError("Stilt mass must be positive")

    spec = MICRODUCK_WALK_ROBOT_CFG.spec_fn()
    vertices, faces = stilt_mesh_data(height_mm, blend)
    mesh_name = "stilt_cartridge_mesh"
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)

    for side in ("left", "right"):
        old_collision = spec.geom(f"{side}_foot_collision")
        old_collision.name = f"{side}_original_sole_disabled"

        old_site = spec.site(f"{side}_foot")
        site_pos = old_site.pos.copy()
        site_quat = old_site.quat.copy()
        old_site.name = f"{side}_original_foot_site"

        stilt_body = spec.body(f"ankle_{side}").add_body(
            name=f"stilt_{side}",
            pos=site_pos,
            quat=site_quat,
        )
        stilt_body.add_geom(
            name=f"{side}_foot_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            mass=mass_kg,
            group=3,
        )
        stilt_body.add_geom(
            name=f"{side}_stilt_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=2,
            rgba=(0.45, 0.34, 0.85, 1.0),
        )
        stilt_body.add_site(
            name=f"{side}_foot",
            pos=(0.0, 0.0, -height_mm * 0.001),
        )

    return spec


def make_stilt_walk_robot_cfg(
    height_mm: float,
    blend: float,
    mass_kg: float | None = None,
) -> EntityCfg:
    """Clone the canonical BAM walk robot and replace only its morphology."""
    validate_stilt_morphology(height_mm, blend)
    robot_cfg = deepcopy(MICRODUCK_WALK_ROBOT_CFG)
    robot_cfg.spec_fn = partial(
        get_stilt_walk_spec,
        height_mm=height_mm,
        blend=blend,
        mass_kg=mass_kg,
    )
    return robot_cfg

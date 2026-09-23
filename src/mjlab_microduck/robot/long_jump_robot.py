"""Fully collision-covered Microduck for whole-body dynamic tasks (long jump).

``robot_allcollisions.xml`` only carries collision hulls for the soles, the
shins, the hip brackets, the head shells and one chassis part.  The trunk
shells, thighs, neck, knee/hip servos and ankle brackets have NO collision
geometry at all (``scripts/audit_collision_coverage.py`` measured up to 49 mm
of visible body outside every collision hull).  A policy trained on that
model can crouch with its thighs inside the floor or land on a trunk that
does not exist for the physics engine, which is impossible on hardware.

``get_long_jump_spec`` loads the all-collisions model and adds a convex-hull
collision geom (MuJoCo collides meshes as convex hulls) for every visible
mesh that is not already covered, except the tiny bearings that sit inside
brackets and the head internals (the three head shells already cover the
head to within 2.3 mm).  Every collision geom is named ``*_collision`` so the
robot's ``CollisionCfg`` (which enables geoms by name) sees all of them.
Parent-child body pairs are excluded from contact by MuJoCo, so explicit
contact pairs are added for the leg segments that can fold onto each other
(thigh/shin, hip bracket/thigh); nested assemblies that overlap by
construction are excluded (``EXCLUDED_BODY_PAIRS``).
"""
from __future__ import annotations

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
    actuators,
)

# Meshes that never need their own hull: bearings sit inside brackets/servos.
SKIP_MESH_PREFIXES = ("seeed_bearing",)
# Bodies whose existing hulls already cover the visible parts (audit: head
# internals protrude <= 2.3 mm from the three shell hulls).
SKIP_BODIES = ("jaw_soft",)
# Parent-child body pairs whose collision geoms must collide explicitly
# (MuJoCo skips parent-child pairs by default).
EXPLICIT_PAIR_BODIES = (
    ("upper_leg_left", "leg"),
    ("upper_leg_right", "leg_2"),
    ("hip_l", "upper_leg_left"),
    ("hip_l_2", "upper_leg_right"),
)
# Non-adjacent body pairs whose hulls overlap by construction (nested
# assemblies: the neck-pitch bracket sits inside the head shells).  The
# ankle servo (on the shin body) nests inside the ankle bracket/foot, but
# shin/foot is a parent-child pair and already excluded by MuJoCo.
EXCLUDED_BODY_PAIRS = (
    ("neck_pitch", "jaw_soft"),
)


def _is_collision_geom(geom: mujoco.MjsGeom) -> bool:
    cls = geom.classname.name if geom.classname is not None else ""
    return cls in ("collision", "self_collision_only") or geom.name.endswith("_collision")


def add_full_collision_geoms(spec: mujoco.MjSpec) -> list[str]:
    """Add named convex-hull collision geoms for uncovered visible meshes.

    Returns the names of the geoms that were added.  Existing unnamed
    collision geoms are given ``<body>_<mesh>_collision`` names.
    """
    coll_default = spec.find_default("collision")
    added: list[str] = []
    for body in spec.bodies:
        if body.name in ("", "world"):
            continue
        existing = []
        for geom in body.geoms:
            if _is_collision_geom(geom):
                if not geom.name:
                    geom.name = f"{body.name}_{geom.meshname}_collision"
                existing.append((geom.meshname, tuple(round(float(p), 5) for p in geom.pos)))
        if body.name in SKIP_BODIES:
            continue
        counter = 0
        for geom in list(body.geoms):
            if _is_collision_geom(geom) or geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = geom.meshname
            if mesh.startswith(SKIP_MESH_PREFIXES):
                continue
            key = (mesh, tuple(round(float(p), 5) for p in geom.pos))
            if key in existing:
                continue
            name = f"{body.name}_{mesh}_{counter}_collision"
            counter += 1
            new = body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh,
                pos=geom.pos,
                quat=geom.quat,
            )
            new.classname = coll_default
            added.append(name)
            existing.append(key)
    return added


def add_leg_fold_pairs(spec: mujoco.MjSpec) -> int:
    """Explicit contact pairs between parent-child leg segments (see module doc)."""
    by_body: dict[str, list[str]] = {}
    for body in spec.bodies:
        by_body[body.name] = [g.name for g in body.geoms if _is_collision_geom(g) and g.name]
    count = 0
    for parent, child in EXPLICIT_PAIR_BODIES:
        for g1 in by_body.get(parent, []):
            for g2 in by_body.get(child, []):
                spec.add_pair(geomname1=g1, geomname2=g2)
                count += 1
    for b1, b2 in EXCLUDED_BODY_PAIRS:
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
    return count


def get_long_jump_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    add_full_collision_geoms(spec)
    add_leg_fold_pairs(spec)
    return spec


MICRODUCK_LONG_JUMP_ROBOT_CFG = EntityCfg(
    spec_fn=get_long_jump_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

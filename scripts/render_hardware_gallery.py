#!/usr/bin/env python3
"""Render consistent studio galleries for the swing seat and green stilts."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from mjlab_microduck.robot.microduck_constants import get_swing_spec
from mjlab_microduck.robot.stilt_constants import get_stilt_walk_spec

GREEN = (0.0, 0.6823529412, 0.2588235294, 1.0)  # #00AE42
STUDIO = (0.84, 0.87, 0.91, 1.0)
FLOOR = (0.69, 0.74, 0.80, 1.0)

SWING_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": -0.4079,
    "left_knee": 1.35,
    "left_ankle": 0.0,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.4079,
    "right_knee": -1.35,
    "right_ankle": 0.0,
}

STAND_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}


def add_studio(spec: mujoco.MjSpec, floor_z: float) -> None:
    """Add a neutral three-wall studio with soft, physically cast shadows."""
    floor_material = spec.add_material(
        name="gallery_floor_material", rgba=FLOOR, roughness=0.82, specular=0.08
    )
    wall_material = spec.add_material(
        name="gallery_wall_material", rgba=STUDIO, roughness=0.9, specular=0.02
    )
    spec.worldbody.add_geom(
        name="gallery_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0.0, 0.0, floor_z),
        size=(4.0, 4.0, 0.05),
        material=floor_material.name,
        contype=0,
        conaffinity=0,
    )
    # The camera always sees a light wall instead of MuJoCo's black void.
    for name, pos, size in (
        ("gallery_wall_x_pos", (2.5, 0.0, 1.3), (0.03, 2.5, 1.3)),
        ("gallery_wall_x_neg", (-2.5, 0.0, 1.3), (0.03, 2.5, 1.3)),
        ("gallery_wall_y_pos", (0.0, 2.5, 1.3), (2.5, 0.03, 1.3)),
        ("gallery_wall_y_neg", (0.0, -2.5, 1.3), (2.5, 0.03, 1.3)),
    ):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            material=wall_material.name,
            contype=0,
            conaffinity=0,
        )

    key = spec.worldbody.add_light(name="gallery_key")
    key.pos = (1.1, -0.9, 1.8)
    key.dir = (-0.45, 0.30, -1.0)
    key.type = mujoco.mjtLightType.mjLIGHT_SPOT
    key.castshadow = True
    key.diffuse = (0.82, 0.84, 0.88)
    key.specular = (0.18, 0.18, 0.18)

    fill = spec.worldbody.add_light(name="gallery_fill")
    fill.pos = (-1.2, 0.8, 1.1)
    fill.dir = (0.55, -0.35, -0.7)
    fill.type = mujoco.mjtLightType.mjLIGHT_SPOT
    fill.castshadow = True
    fill.diffuse = (0.46, 0.48, 0.52)
    fill.specular = (0.06, 0.06, 0.06)

    spec.visual.global_.offwidth = 1400
    spec.visual.global_.offheight = 1050
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = (0.32, 0.32, 0.34)
    spec.visual.headlight.diffuse = (0.38, 0.38, 0.40)
    spec.visual.headlight.specular = (0.08, 0.08, 0.08)
    spec.visual.map.znear = 0.002
    spec.visual.map.shadowclip = 3.0


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: tuple[float, float, float],
    joints: dict[str, float],
) -> None:
    mujoco.mj_resetData(model, data)
    root = model.joint("trunk_base_freejoint").qposadr[0]
    data.qpos[root : root + 3] = root_pos
    data.qpos[root + 3 : root + 7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in joints.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def render_views(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_dir: Path,
    prefix: str,
    lookat: tuple[float, float, float],
    distance: float,
    views: tuple[tuple[str, float, float], ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=900, width=1200)
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.geomgroup[3] = 0  # hide collision proxies
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    for name, azimuth, elevation in views:
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera, scene_option=options)
        Image.fromarray(renderer.render()).save(output_dir / f"{prefix}_{name}.png")
    renderer.close()


def swing_gallery(output_dir: Path) -> None:
    spec = get_swing_spec()
    # Keep the product shots about the retained seat, not the surrounding rig.
    for geom in spec.geoms:
        if geom.name.startswith("swing_frame_"):
            geom.rgba = (0.0, 0.0, 0.0, 0.0)
    for tendon in spec.tendons:
        if tendon.name.startswith("swing_string_"):
            tendon.width = 1e-6
    add_studio(spec, floor_z=-0.02)
    model = spec.compile()
    data = mujoco.MjData(model)
    set_pose(model, data, (0.0, 0.0, 0.28), SWING_POSE)
    render_views(
        model,
        data,
        output_dir,
        "seat",
        lookat=(0.0, 0.0, 0.29),
        distance=0.47,
        views=(
            ("front", 180.0, -8.0),
            ("three_quarter", 132.0, -12.0),
            ("side", 90.0, -7.0),
        ),
    )


def stilt_gallery(output_dir: Path, height_cm: float) -> None:
    spec = get_stilt_walk_spec(height_cm=height_cm, blend=0.5)
    # Match the demonstrated hardware: both the replacement sole and stilt are green.
    for geom in spec.geoms:
        if geom.name.endswith("_stilt_visual") or geom.meshname in {
            "sole_left",
            "sole_right",
        }:
            geom.material = ""
            geom.rgba = GREEN
    add_studio(spec, floor_z=0.0)
    model = spec.compile()
    data = mujoco.MjData(model)
    root_z = 0.12 + height_cm * 0.01
    set_pose(model, data, (0.0, 0.0, root_z), STAND_POSE)
    render_views(
        model,
        data,
        output_dir,
        "stilts",
        lookat=(0.0, 0.0, 0.20 + height_cm * 0.005),
        distance=0.58 + height_cm * 0.006,
        views=(
            ("front", 180.0, -9.0),
            ("three_quarter", 137.0, -12.0),
            ("side", 90.0, -7.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("hardware"))
    parser.add_argument("--stilt-height-cm", type=float, default=10.0)
    args = parser.parse_args()
    swing_gallery(args.output_root / "swing-seat" / "renders")
    stilt_gallery(args.output_root / "stilts" / "renders", args.stilt_height_cm)


if __name__ == "__main__":
    main()

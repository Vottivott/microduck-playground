"""Render the latest running policy colliding with an upright crash mat."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import types
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
from rsl_rl.runners import OnPolicyRunner

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab_microduck.tasks  # noqa: F401


def prepare_visual_mesh(glb_path: Path, output_dir: Path) -> tuple[Path, np.ndarray]:
    """Export the complete GLB mesh to a MuJoCo-friendly OBJ without decimation."""
    scene = trimesh.load(glb_path, force="scene")
    source = scene.to_mesh()

    # The supplied mat is long in X. Stand it on its narrow edge with the long
    # dimension spanning the lane: X=thickness, Y=width, Z=height. This makes
    # the broad padded face normal to the robot's +X running direction, as in
    # the reference photograph.
    source.apply_translation(-source.bounding_box.centroid)
    source.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2.0, (0.0, 0.0, 1.0))
    )
    source.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi / 2.0, (0.0, 1.0, 0.0))
    )
    # The first upright render put the rear support legs on the approach side.
    # Turn the visual 180 degrees around world-up so the padded face meets the
    # duck and the support frame sits behind it. This leaves the box extents and
    # therefore the physical collision plane unchanged.
    source.apply_transform(
        trimesh.transformations.rotation_matrix(math.pi, (0.0, 0.0, 1.0))
    )
    obj_path = output_dir / "crash_mat_upright_full.obj"
    obj_path.write_text(
        trimesh.exchange.obj.export_obj(
            source,
            include_color=False,
            include_texture=False,
        ),
        encoding="utf-8",
    )
    return obj_path, source.extents


def make_barrier_spec(xml_path: Path) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(xml_path))


def make_upper_body_floor_spec(barrier_x: float) -> mujoco.MjSpec:
    region_center_x = barrier_x + 2.5
    return mujoco.MjSpec.from_string(
        f"""<mujoco model="upper_body_floor">
  <worldbody>
    <geom name="upper_body_floor" type="box" size="4 4 0.05"
          pos="{region_center_x:.8f} 0 -0.052"
          contype="16" conaffinity="8" group="3" rgba="0 0 0 0"
          friction="1.0 0.005 0.0001" margin="0"
          solref="0.005 1" solimp="0.95 0.99 0.001"/>
  </worldbody>
</mujoco>"""
    )


def make_robot_with_upper_body_collision() -> mujoco.MjSpec:
    from mjlab_microduck.robot.microduck_constants import (
        get_standup_spec,
        get_walk_spec,
    )

    # Preserve the exact trained walk model. Copy the CAD-aligned external
    # collision shells from the full spec onto isolated crash bits. The special
    # floor exists only near the mat, so these shells cannot perturb the run-up.
    spec = get_walk_spec()
    full_spec = get_standup_spec()
    foot_names = {"left_foot_collision", "right_foot_collision"}
    source_geoms = [
        geom
        for geom in full_spec.geoms
        if int(geom.contype) == 1 and geom.name not in foot_names
    ]
    if len(source_geoms) != 8:
        raise RuntimeError(f"expected eight CAD crash shells, got {source_geoms}")
    for index, source in enumerate(source_geoms):
        destination_body = spec.body(source.parent.name)
        if destination_body is None:
            raise RuntimeError(f"walk spec is missing body {source.parent.name!r}")
        collision = destination_body.add_geom(
            name=f"cad_{index}_{source.meshname}_crash_collision",
            type=source.type,
            meshname=source.meshname,
            pos=tuple(source.pos),
            quat=tuple(source.quat),
        )
        collision.contype = 8
        collision.conaffinity = 16
        collision.group = 3
        collision.rgba = (0.0, 0.0, 0.0, 0.0)
        collision.margin = 0.0

    # The full model's trunk shell is mostly the rear battery. Add a conservative
    # central volume so the white torso cannot pass below the floor.
    trunk = spec.body("trunk_base")
    trunk_proxy = trunk.add_geom(
        name="trunk_crash_collision",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=(0.070, 0.052, 0.055),
        pos=(-0.012, 0.0, 0.0),
    )
    trunk_proxy.contype = 8
    trunk_proxy.conaffinity = 16
    trunk_proxy.group = 3
    trunk_proxy.rgba = (0.0, 0.0, 0.0, 0.0)

    # Close the articulated gaps between the torso and exact head shells.
    for body_name, radius in (
        ("neck", 0.018),
        ("neck_pitch", 0.018),
        ("yaw_roll_motion", 0.020),
        ("bearing_roll", 0.020),
    ):
        body = spec.body(body_name)
        proxy = body.add_geom(
            name=f"{body_name}_crash_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(radius, 0.0, 0.0),
        )
        proxy.contype = 8
        proxy.conaffinity = 16
        proxy.group = 3
        proxy.rgba = (0.0, 0.0, 0.0, 0.0)
    return spec


def compiled_collision_manifest(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """Return and validate the collision masks that survived scene compilation."""
    manifest = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not (
            name.endswith("_crash_collision")
            or name.endswith("crash_mat_collision")
            or name.endswith("upper_body_floor")
        ):
            continue
        manifest[name] = {
            "contype": int(model.geom_contype[geom_id]),
            "conaffinity": int(model.geom_conaffinity[geom_id]),
            "condim": int(model.geom_condim[geom_id]),
        }

    shell_entries = {
        name: attrs
        for name, attrs in manifest.items()
        if name.endswith("_crash_collision")
        and not name.endswith("crash_mat_collision")
    }
    if len(shell_entries) != 13:
        raise RuntimeError(
            f"expected thirteen compiled full-body crash geoms, got {shell_entries}"
        )
    for name, attrs in shell_entries.items():
        if attrs["contype"] != 8 or attrs["conaffinity"] != 16:
            raise RuntimeError(f"compiled crash geom mask was rewritten: {name}={attrs}")

    expected_static = {
        "crash_mat_collision": (17, 9),
        "upper_body_floor": (16, 8),
    }
    for suffix, (contype, conaffinity) in expected_static.items():
        matches = [attrs for name, attrs in manifest.items() if name.endswith(suffix)]
        if len(matches) != 1 or (
            matches[0]["contype"], matches[0]["conaffinity"]
        ) != (contype, conaffinity):
            raise RuntimeError(
                f"compiled static collision mask mismatch for {suffix}: {matches}"
            )
    return manifest


def spawn_spark_burst(
    emitter: np.ndarray,
    particle_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create a deterministic hot-metal burst in world coordinates."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(particle_count, 3))
    # Electrical fragments scatter irregularly, with only a mild upward bias;
    # avoiding a uniform upper hemisphere prevents a synthetic firework shape.
    direction[:, 2] = rng.normal(0.20, 0.50, particle_count)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    speed = rng.lognormal(mean=math.log(1.55), sigma=0.34, size=particle_count)
    speed = np.clip(speed, 0.55, 3.35)
    launch_delay = rng.exponential(0.026, particle_count)
    launch_delay = np.clip(launch_delay, 0.0, 0.12)
    launch_delay[: max(6, particle_count // 6)] = 0.0
    position = np.repeat(emitter[None, :], particle_count, axis=0)
    position += rng.normal(scale=(0.012, 0.010, 0.010), size=position.shape)
    return {
        "emitter": emitter.copy(),
        "position": position,
        "previous_position": position.copy(),
        "velocity": direction * speed[:, None],
        "age": -launch_delay,
        "lifetime": rng.uniform(0.16, 0.46, particle_count),
        "radius": rng.uniform(0.00055, 0.00125, particle_count),
    }


def advance_spark_burst(
    sparks: dict[str, np.ndarray],
    dt: float,
    mat_face_x: float,
    mat_half_width: float,
    mat_height: float,
) -> None:
    """Integrate sparks with gravity, drag, and floor/mat rebounds."""
    old_age = sparks["age"].copy()
    sparks["age"] += dt
    active = (sparks["age"] >= 0.0) & (sparks["age"] <= sparks["lifetime"])
    newly_launched = (old_age < 0.0) & (sparks["age"] >= 0.0)
    sparks["position"][newly_launched] = sparks["emitter"]
    sparks["previous_position"][active] = sparks["position"][active]
    if not np.any(active):
        return
    velocity = sparks["velocity"]
    velocity[active, 2] -= 9.81 * dt
    velocity[active] *= math.exp(-0.75 * dt)
    sparks["position"][active] += velocity[active] * dt

    # Tiny hot fragments skip once or twice on the physical floor.
    floor_hit = active & (sparks["position"][:, 2] < 0.004)
    sparks["position"][floor_hit, 2] = 0.004
    velocity[floor_hit, 2] = np.abs(velocity[floor_hit, 2]) * 0.24
    velocity[floor_hit, :2] *= 0.62

    # The padded face is a world-space collision plane for the particles too.
    mat_hit = (
        active
        & (sparks["previous_position"][:, 0] < mat_face_x)
        & (sparks["position"][:, 0] >= mat_face_x)
        & (np.abs(sparks["position"][:, 1]) <= mat_half_width)
        & (sparks["position"][:, 2] <= mat_height)
    )
    sparks["position"][mat_hit, 0] = mat_face_x - 0.002
    velocity[mat_hit, 0] = -np.abs(velocity[mat_hit, 0]) * 0.18
    velocity[mat_hit, 1:] *= 0.72


def render_spark_burst(visualizer, sparks: dict[str, np.ndarray] | None) -> None:
    """Add emissive streak geometry to MuJoCo's depth-buffered 3D scene."""
    if sparks is None:
        return
    burst_age = float(np.max(sparks["age"]))
    if 0.0 <= burst_age < 0.12 and visualizer.scn.nlight < len(visualizer.scn.lights):
        # A short local flash lets the effect illuminate nearby real surfaces;
        # it is a native scene light, not a screen-space bloom.
        flash = (1.0 - burst_age / 0.12) ** 2
        light = visualizer.scn.lights[visualizer.scn.nlight]
        visualizer.scn.nlight += 1
        light.type = mujoco.mjtLightType.mjLIGHT_POINT.value
        light.id = -1
        light.headlight = 0
        light.pos[:] = sparks["emitter"]
        light.dir[:] = (0.0, 0.0, -1.0)
        light.attenuation[:] = (1.0, 0.65, 2.4)
        light.cutoff = 180.0
        light.exponent = 0.0
        light.range = 1.6
        light.bulbradius = 0.012
        light.castshadow = 0
        light.ambient[:] = np.asarray((0.025, 0.010, 0.001)) * flash
        light.diffuse[:] = np.asarray((0.95, 0.42, 0.07)) * flash
        light.specular[:] = np.asarray((1.0, 0.55, 0.12)) * flash
    active_ids = np.flatnonzero(
        (sparks["age"] >= 0.0) & (sparks["age"] <= sparks["lifetime"])
    )
    for particle_id in active_ids:
        age_fraction = float(
            sparks["age"][particle_id] / sparks["lifetime"][particle_id]
        )
        fade = (1.0 - age_fraction) ** 1.35
        # White-hot at ignition, cooling through yellow into orange.
        color = (
            1.0,
            0.34 + 0.66 * fade,
            0.03 + 0.42 * fade,
            0.96 * fade,
        )
        position = sparks["position"][particle_id]
        velocity = sparks["velocity"][particle_id]
        speed = float(np.linalg.norm(velocity))
        trail_s = np.clip(0.005 + 0.0025 * speed, 0.006, 0.014)
        trail_start = position - velocity * trail_s
        radius = float(sparks["radius"][particle_id])

        visualizer.add_cylinder(trail_start, position, radius, color)
        trail_geom = visualizer.scn.geoms[visualizer.scn.ngeom - 1]
        trail_geom.emission = 1.0
        trail_geom.specular = 0.8
        trail_geom.shininess = 0.9
        visualizer.add_sphere(position, radius * 1.35, color)
        core_geom = visualizer.scn.geoms[visualizer.scn.ngeom - 1]
        core_geom.emission = 1.0


def write_barrier_xml(
    obj_path: Path,
    output_dir: Path,
    extents: np.ndarray,
) -> Path:
    half = extents / 2.0
    xml_path = output_dir / "crash_mat.xml"
    xml_path.write_text(
        f"""<mujoco model="crash_mat">
  <compiler angle="radian" meshdir="{output_dir}"/>
  <asset>
    <mesh name="crash_mat_visual" file="{obj_path.name}"/>
  </asset>
  <worldbody>
    <!-- A soft approach-side key light makes the robot cast onto both the
         ground and the vertical padded face during impact. -->
    <light name="impact_shadow_key" pos="2 -2 3" dir="1 0 -0.35"
           directional="true" castshadow="true"
           diffuse="0.28 0.28 0.28" specular="0.04 0.04 0.04"
           ambient="0 0 0"/>
    <body name="crash_mat">
      <geom name="crash_mat_visual_geom" type="mesh" mesh="crash_mat_visual"
            contype="0" conaffinity="0" group="2" rgba="0.08 0.20 0.46 1"/>
      <geom name="crash_mat_collision" type="box"
            size="{half[0]:.8f} {half[1]:.8f} {half[2]:.8f}"
            contype="17" conaffinity="9" group="3" rgba="0 0 0 0"
            friction="0.9 0.02 0.001" margin="0"
            solref="0.008 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return xml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--barrier-x", type=float, default=5.80)
    parser.add_argument("--video-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--startup-seed", type=int, default=0)
    parser.add_argument("--camera-final-distance", type=float, default=3.5)
    parser.add_argument("--camera-start-fovy", type=float, default=22.0)
    parser.add_argument("--camera-end-fovy", type=float, default=14.0)
    parser.add_argument("--camera-zoom-start-s", type=float, default=0.15)
    parser.add_argument("--camera-zoom-end-s", type=float, default=0.75)
    parser.add_argument("--camera-pan-smoothing", type=float, default=0.12)
    parser.add_argument("--camera-final-azimuth", type=float, default=35.0)
    parser.add_argument("--camera-final-elevation", type=float, default=-10.0)
    parser.add_argument("--camera-lookahead", type=float, default=0.20)
    parser.add_argument("--camera-impact-hold-s", type=float, default=0.60)
    parser.add_argument("--power-off-s", type=float, default=6.0)
    parser.add_argument("--spark-particles", type=int, default=110)
    parser.add_argument("--spark-seed", type=int, default=20260829)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    obj_path, extents = prepare_visual_mesh(args.glb, args.output_dir)
    barrier_xml = write_barrier_xml(obj_path, args.output_dir, extents)

    # Fix the camera in world space at the point that yields the accepted V5
    # angle at impact. At the start this naturally gives the mirrored view of
    # the duck coming toward the camera; as it passes, only pan/tilt/FOV change.
    collision_face_x = args.barrier_x - float(extents[0] / 2.0)
    final_lookat = np.array(
        [collision_face_x + args.camera_lookahead, 0.0, 0.22], dtype=np.float64
    )
    final_azimuth_rad = math.radians(args.camera_final_azimuth)
    final_elevation_rad = math.radians(args.camera_final_elevation)
    final_forward = np.array(
        [
            math.cos(final_elevation_rad) * math.cos(final_azimuth_rad),
            math.cos(final_elevation_rad) * math.sin(final_azimuth_rad),
            math.sin(final_elevation_rad),
        ]
    )
    fixed_camera_pos = final_lookat - args.camera_final_distance * final_forward
    initial_lookat = np.array([args.camera_lookahead, 0.0, 0.22])
    initial_aim = initial_lookat - fixed_camera_pos
    initial_distance = float(np.linalg.norm(initial_aim))
    initial_azimuth = math.degrees(math.atan2(initial_aim[1], initial_aim[0]))
    initial_elevation = math.degrees(
        math.asin(initial_aim[2] / initial_distance)
    )

    env_cfg = load_env_cfg("Mjlab-Running-Flat-MicroDuck", play=True)
    agent_cfg = load_rl_cfg("Mjlab-Running-Flat-MicroDuck")
    # Use one audited startup physical instance, then select a deterministic
    # reset realization whose unforced policy trajectory reaches the mat.
    env_cfg.seed = args.startup_seed
    env_cfg.scene.num_envs = 1
    # The stock running scene uses a 2 m model extent. MuJoCo sizes the
    # directional-light shadow box from this value, which caused the visible
    # hard shadow cutoff several metres before the mat. Compile the renderer
    # with a runway-scale extent so the shadow map covers the entire shot.
    env_cfg.scene.extent = 12.0
    env_cfg.episode_length_s = 30.0
    # A free camera lets us update pan/tilt/zoom while analytically preserving
    # one fixed world-space camera position.
    env_cfg.viewer.origin_type = env_cfg.viewer.OriginType.WORLD
    env_cfg.viewer.lookat = tuple(initial_lookat)
    env_cfg.viewer.distance = initial_distance
    env_cfg.viewer.fovy = args.camera_start_fovy
    env_cfg.viewer.elevation = initial_elevation
    env_cfg.viewer.azimuth = initial_azimuth
    env_cfg.viewer.width = 720
    env_cfg.viewer.height = 720
    env_cfg.viewer.max_extra_envs = 0

    reset_base = env_cfg.events["reset_base"].params
    reset_base["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.125, 0.125),
        "yaw": (0.0, 0.0),
    }
    reset_base["velocity_range"] = {}
    env_cfg.terminations.clear()

    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (args.speed, args.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.resampling_time_range = (30.0, 30.0)
    # Hide command targets so the shot reads like handheld footage rather than
    # a simulator debug view.
    for command_cfg in env_cfg.commands.values():
        if hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = False

    robot_cfg = deepcopy(env_cfg.scene.entities["robot"])
    robot_cfg.spec_fn = make_robot_with_upper_body_collision
    # The stock collision editor resets every matching geom to 1/1 and disables
    # non-matches. Include the crash geoms in the same exhaustive editor so the
    # dedicated reciprocal masks survive entity construction.
    robot_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(r".*_collision$",),
            contype={r".*_crash_collision$": 8, r".*_collision$": 1},
            conaffinity={r".*_crash_collision$": 16, r".*_collision$": 1},
            condim={
                r"^(left|right)_foot_collision$": 3,
                r".*_crash_collision$": 3,
                r".*_collision$": 1,
            },
            priority={r"^(left|right)_foot_collision$": 1, r".*_collision$": 0},
            friction={
                r"^(left|right)_foot_collision$": (1.0,),
                r".*_crash_collision$": (0.9, 0.02, 0.001),
            },
            margin={r".*_crash_collision$": 0.0},
        ),
    )
    env_cfg.scene.entities["robot"] = robot_cfg
    env_cfg.scene.entities["crash_mat"] = EntityCfg(
        spec_fn=lambda: make_barrier_spec(barrier_xml),
        init_state=EntityCfg.InitialStateCfg(
            pos=(args.barrier_x, 0.0, float(extents[2] / 2.0)),
        ),
    )
    env_cfg.scene.entities["upper_body_floor"] = EntityCfg(
        spec_fn=lambda: make_upper_body_floor_spec(args.barrier_x),
    )

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    video_dir = args.output_dir / "video"
    video_env = VideoRecorder(
        raw_env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.video_steps,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls("Mjlab-Running-Flat-MicroDuck") or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(args.checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = raw_env.scene["robot"]
    raw_env.reset(seed=args.seed)
    obs = env.get_observations()
    camera = raw_env._offline_renderer._cam
    render_model = raw_env._offline_renderer._model
    render_data = raw_env._offline_renderer._data
    collision_manifest = compiled_collision_manifest(render_model)
    print("COMPILED_COLLISION_MANIFEST " + json.dumps(collision_manifest, sort_keys=True))
    # The fixed tripod sees a much larger world-space extent than the original
    # robot-following camera. Expand its shadow frustum so the duck's ground
    # shadow remains present all the way to the mat.
    render_model.stat.extent = max(float(render_model.stat.extent), 12.0)
    render_model.vis.map.shadowclip = 1.5
    render_model.vis.map.shadowscale = 1.0
    smoothed_lookat = np.asarray(env_cfg.viewer.lookat, dtype=np.float64)
    head_body_id = robot.find_bodies("jaw_soft")[0][0]
    trunk_body_id = robot.find_bodies("trunk_base")[0][0]
    root_x = []
    root_y = []
    tilt_deg = []
    head_center_z = []
    trunk_center_z = []
    camera_azimuth = []
    camera_distance = []
    camera_fovy = []
    crash_contact_distances = []
    impact_camera_locked = False
    impact_lock_step = None
    camera_reacquired_robot = False
    power_off_step = round(args.power_off_s / raw_env.step_dt)
    power_was_cut = False
    zero_ctrl = torch.zeros(
        (raw_env.num_envs, robot.num_actuators), device=device, dtype=torch.float32
    )
    spark_state = {"particles": None}
    original_update_visualizers = raw_env.update_visualizers

    def update_visualizers_with_sparks(self, visualizer) -> None:
        original_update_visualizers(visualizer)
        render_spark_burst(visualizer, spark_state["particles"])

    raw_env.update_visualizers = types.MethodType(
        update_visualizers_with_sparks, raw_env
    )
    screen_track = []
    for step in range(args.video_steps):
        root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        if step == power_off_step:
            emitter = (
                robot.data.body_com_pos_w[0, trunk_body_id].detach().cpu().numpy()
            )
            spark_state["particles"] = spawn_spark_burst(
                emitter=emitter,
                particle_count=args.spark_particles,
                seed=args.spark_seed,
            )
        elif step > power_off_step and spark_state["particles"] is not None:
            advance_spark_burst(
                spark_state["particles"],
                dt=raw_env.step_dt,
                mat_face_x=collision_face_x,
                mat_half_width=float(extents[1] / 2.0),
                mat_height=float(extents[2]),
            )
        # Pan smoothly toward the duck, but derive all camera angles from the
        # fixed tripod position rather than translating the camera with it.
        if root_pos[0] >= collision_face_x - 0.18:
            impact_camera_locked = True
            if impact_lock_step is None:
                impact_lock_step = step
        impact_hold_steps = round(args.camera_impact_hold_s / raw_env.step_dt)
        holding_impact = (
            impact_camera_locked
            and impact_lock_step is not None
            and step < impact_lock_step + impact_hold_steps
        )
        if impact_camera_locked and not holding_impact:
            camera_reacquired_robot = True
        target = (
            final_lookat
            if holding_impact
            else np.array([root_pos[0] + args.camera_lookahead, root_pos[1], 0.22])
        )
        smoothed_lookat += args.camera_pan_smoothing * (target - smoothed_lookat)
        aim = smoothed_lookat - fixed_camera_pos
        distance = float(np.linalg.norm(aim))
        camera.lookat[:] = smoothed_lookat
        camera.distance = distance
        camera.azimuth = math.degrees(math.atan2(aim[1], aim[0]))
        camera.elevation = math.degrees(math.asin(aim[2] / distance))

        time_s = step * raw_env.step_dt
        zoom_phase = np.clip(
            (time_s - args.camera_zoom_start_s)
            / (args.camera_zoom_end_s - args.camera_zoom_start_s),
            0.0,
            1.0,
        )
        zoom_phase = zoom_phase * zoom_phase * (3.0 - 2.0 * zoom_phase)
        fovy = args.camera_start_fovy + zoom_phase * (
            args.camera_end_fovy - args.camera_start_fovy
        )
        render_model.vis.global_.fovy = fovy
        camera_azimuth.append(float(camera.azimuth))
        camera_distance.append(distance)
        camera_fovy.append(float(fovy))
        if step == power_off_step:
            # A failed robot still has gravity, inertia, friction, damping, and
            # full contact physics. Only its motor commands disappear on every
            # physics substep, so it goes genuinely limp instead of freezing.
            def apply_powered_off_controls(self) -> None:
                self.write_ctrl_to_sim(zero_ctrl)

            robot._apply_actuator_controls = types.MethodType(
                apply_powered_off_controls, robot
            )
            power_was_cut = True
            print(
                f"POWER_OFF frame={step} time_s={step * raw_env.step_dt:.6f} "
                "motor_ctrl=forced_zero passive_physics=enabled"
            )
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        root_x.append(float(robot.data.root_link_pos_w[0, 0]))
        root_y.append(float(robot.data.root_link_pos_w[0, 1]))
        gravity_z = float(robot.data.projected_gravity_b[0, 2])
        tilt_deg.append(math.degrees(math.acos(np.clip(-gravity_z, -1.0, 1.0))))
        head_center_z.append(
            float(robot.data.body_com_pos_w[0, head_body_id, 2])
        )
        trunk_center_z.append(
            float(robot.data.body_com_pos_w[0, trunk_body_id, 2])
        )
        # Project the torso through the fixed, no-roll tripod basis. Deriving
        # this from the same world position and look-at used above avoids an
        # mjlab/MuJoCo scene-wrapper boundary in mjv_cameraInModel().
        camera_head = fixed_camera_pos
        camera_forward = aim / distance
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        camera_right = np.cross(camera_forward, world_up)
        camera_right /= np.linalg.norm(camera_right)
        camera_up = np.cross(camera_right, camera_forward)
        camera_up /= np.linalg.norm(camera_up)
        trunk_world = (
            robot.data.body_com_pos_w[0, trunk_body_id].detach().cpu().numpy()
        )
        camera_relative = trunk_world - camera_head
        camera_depth = float(np.dot(camera_relative, camera_forward))
        focal_pixels = (env_cfg.viewer.height / 2.0) / math.tan(
            math.radians(fovy) / 2.0
        )
        screen_u = env_cfg.viewer.width / 2.0 + focal_pixels * float(
            np.dot(camera_relative, camera_right)
        ) / camera_depth
        screen_v = env_cfg.viewer.height / 2.0 - focal_pixels * float(
            np.dot(camera_relative, camera_up)
        ) / camera_depth
        screen_track.append(
            {
                "frame": step,
                "time_s": step * raw_env.step_dt,
                "screen_u_px": screen_u,
                "screen_v_px": screen_v,
                "camera_depth_m": camera_depth,
                "trunk_world_m": trunk_world.tolist(),
                "camera_world_m": camera_head.tolist(),
            }
        )
        # VideoRecorder has just synchronized the GPU simulation into its CPU
        # rendering state. Audit dedicated shell contacts numerically; negative
        # MuJoCo contact distance is geometric penetration.
        for contact_index in range(render_data.ncon):
            contact = render_data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            is_crash_contact = (
                int(render_model.geom_contype[geom1]) & 8
                and int(render_model.geom_conaffinity[geom2]) & 8
            ) or (
                int(render_model.geom_contype[geom2]) & 8
                and int(render_model.geom_conaffinity[geom1]) & 8
            )
            if is_crash_contact:
                crash_contact_distances.append(float(contact.dist))

    env.close()
    recorded = video_dir / "rl-video-step-0.mp4"
    final_video = args.output_dir / "microduck_crash_mat_upright_telephoto.mp4"
    # Cloud-backed mounts may reject timestamp updates attempted by copy2().
    shutil.copyfile(recorded, final_video)
    track_path = args.output_dir / "robot_screen_track.json"
    track_path.write_text(
        json.dumps(screen_track, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "startup_seed": args.startup_seed,
        "rollout_seed": args.seed,
        "speed_command_mps": args.speed,
        "barrier_center_x_m": args.barrier_x,
        "barrier_extents_m": extents.tolist(),
        "barrier_front_face_x_m": collision_face_x,
        "max_root_x_m": max(root_x),
        "final_root_x_m": root_x[-1],
        "maximum_absolute_root_y_m": max(abs(y) for y in root_y),
        "max_tilt_deg": max(tilt_deg),
        "min_head_center_z_m": min(head_center_z),
        "min_trunk_center_z_m": min(trunk_center_z),
        "crash_shell_contact_count": len(crash_contact_distances),
        "minimum_crash_shell_contact_distance_m": (
            min(crash_contact_distances) if crash_contact_distances else None
        ),
        "maximum_crash_shell_penetration_m": (
            max(0.0, -min(crash_contact_distances))
            if crash_contact_distances
            else None
        ),
        "compiled_collision_manifest": collision_manifest,
        "power_off": {
            "requested_time_s": args.power_off_s,
            "frame": power_off_step,
            "actual_time_s": power_off_step * raw_env.step_dt,
            "motor_ctrl": "forced_zero",
            "passive_physics_continues": True,
            "executed": power_was_cut,
        },
        "spark_particles": {
            "rendering": "native_mujoco_3d_geometry",
            "count": args.spark_particles,
            "seed": args.spark_seed,
            "spawn_frame": power_off_step,
            "spawn_time_s": power_off_step * raw_env.step_dt,
            "gravity_mps2": -9.81,
            "linear_drag_per_s": 0.75,
            "floor_collision": True,
            "mat_collision": True,
            "smoke": False,
        },
        "shadow_rendering": {
            "model_extent_m": float(render_model.stat.extent),
            "directional_shadow_half_width_m": float(
                render_model.stat.extent * render_model.vis.map.shadowclip
            ),
            "spotlight_shadow_scale": float(render_model.vis.map.shadowscale),
        },
        "camera": {
            "fixed_world_position_m": fixed_camera_pos.tolist(),
            "start_azimuth_deg": camera_azimuth[0],
            "end_azimuth_deg": camera_azimuth[-1],
            "start_distance_m": camera_distance[0],
            "end_distance_m": camera_distance[-1],
            "start_vertical_fov_deg": camera_fovy[0],
            "end_vertical_fov_deg": camera_fovy[-1],
            "pan_ema_gain_per_frame": args.camera_pan_smoothing,
            "intended_final_azimuth_deg": args.camera_final_azimuth,
            "intended_final_elevation_deg": args.camera_final_elevation,
            "lookahead_m": args.camera_lookahead,
            "impact_hold_enabled": True,
            "impact_hold_s": args.camera_impact_hold_s,
            "impact_lock_frame": impact_lock_step,
            "reacquired_robot_after_hold": camera_reacquired_robot,
        },
        "screen_track": str(track_path),
        "video": str(final_video),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

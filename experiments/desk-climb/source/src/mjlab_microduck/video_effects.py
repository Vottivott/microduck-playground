"""Presentation effects for evaluation videos.

* :func:`configure_video_cfg` — 720p frames, no command-arrow debug markers.
* :func:`fix_render_shadows` — widen the directional shadow map so ground
  shadows do not cut off in the background.
* :class:`CrashEffects` — the moment a robot hits the ground its motors are
  cut (zero ctrl on every physics substep, so it goes limp under gravity and
  friction instead of freezing) and a burst of hot sparks is emitted at the
  impact point.  Sparks are emissive geoms in the MuJoCo scene, so they are
  depth-tested against the robots and light the nearby floor.

The spark burst is ported from ``scripts/render_running_crash_video.py``.
"""

from __future__ import annotations

import math
import types
from typing import Sequence

import mujoco
import numpy as np
import torch


def configure_video_cfg(env_cfg, width: int = 1280, height: int = 720) -> None:
    """720p output and no floating command arrows / sensor markers."""
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    for command_cfg in getattr(env_cfg, "commands", {}).values():
        if hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = False
    for sensor_cfg in getattr(env_cfg.scene, "sensors", ()) or ():
        if hasattr(sensor_cfg, "debug_vis"):
            sensor_cfg.debug_vis = False


def fix_render_shadows(raw_env, min_extent: float = 4.0, light_dir: Sequence[float] | None = None) -> None:
    """MuJoCo sizes the directional-light shadow box from ``stat.extent``;
    with the default extent the shadow map ends a metre or two from the
    tracked robot and shows as a hard cut in the background.  Enlarge the
    shadow frustum; normal contact shadows are unchanged.

    ``light_dir`` re-aims every directional light (the scene sun).  The
    default sun points straight down, so tall off-screen structures (the
    upper ladder treads) cast detached rectangles onto the floor; aiming the
    sun along a structure's incline collapses those shadows at its foot.
    """
    renderer = getattr(raw_env, "_offline_renderer", None)
    if renderer is None:
        return
    model = renderer._model
    model.stat.extent = max(float(model.stat.extent), min_extent)
    model.vis.map.shadowclip = 1.5
    model.vis.map.shadowscale = 1.0
    model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize), 8192)
    if light_dir is not None:
        d = np.asarray(light_dir, dtype=np.float64)
        d /= np.linalg.norm(d)
        for i in range(model.nlight):
            if int(model.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                model.light_dir[i] = d


# --- sparks -----------------------------------------------------------------------


def spawn_spark_burst(emitter: np.ndarray, particle_count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(particle_count, 3))
    direction[:, 2] = rng.normal(0.20, 0.50, particle_count)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    speed = np.clip(rng.lognormal(mean=math.log(1.55), sigma=0.34, size=particle_count), 0.55, 3.35)
    launch_delay = np.clip(rng.exponential(0.026, particle_count), 0.0, 0.12)
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
        "radius": rng.uniform(0.0012, 0.0026, particle_count),
    }


def advance_spark_burst(sparks: dict[str, np.ndarray], dt: float, floor_z: float = 0.0) -> None:
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
    floor_hit = active & (sparks["position"][:, 2] < floor_z + 0.004)
    sparks["position"][floor_hit, 2] = floor_z + 0.004
    velocity[floor_hit, 2] = np.abs(velocity[floor_hit, 2]) * 0.24
    velocity[floor_hit, :2] *= 0.62


def spark_burst_alive(sparks: dict[str, np.ndarray]) -> bool:
    return bool(np.any(sparks["age"] <= sparks["lifetime"]))


def render_spark_burst(visualizer, sparks: dict[str, np.ndarray]) -> None:
    scn = getattr(visualizer, "scn", None)
    if scn is None:
        return
    burst_age = float(np.max(sparks["age"]))
    if 0.0 <= burst_age < 0.12 and scn.nlight < len(scn.lights):
        flash = (1.0 - burst_age / 0.12) ** 2
        light = scn.lights[scn.nlight]
        scn.nlight += 1
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
    active_ids = np.flatnonzero((sparks["age"] >= 0.0) & (sparks["age"] <= sparks["lifetime"]))
    for particle_id in active_ids:
        if scn.ngeom + 2 > len(scn.geoms):
            break
        age_fraction = float(sparks["age"][particle_id] / sparks["lifetime"][particle_id])
        fade = (1.0 - age_fraction) ** 1.35
        color = (1.0, 0.34 + 0.66 * fade, 0.03 + 0.42 * fade, 0.96 * fade)
        position = sparks["position"][particle_id]
        velocity = sparks["velocity"][particle_id]
        speed = float(np.linalg.norm(velocity))
        trail_s = float(np.clip(0.005 + 0.0025 * speed, 0.006, 0.014))
        radius = float(sparks["radius"][particle_id])
        visualizer.add_cylinder(position - velocity * trail_s, position, radius, color)
        trail_geom = scn.geoms[scn.ngeom - 1]
        trail_geom.emission = 1.0
        trail_geom.specular = 0.8
        trail_geom.shininess = 0.9
        visualizer.add_sphere(position, radius * 1.35, color)
        scn.geoms[scn.ngeom - 1].emission = 1.0


# --- crash effects ---------------------------------------------------------------


class CrashEffects:
    """Cut the motors and fire sparks the moment a robot hits the ground.

    Ground contact is detected geometrically: trunk or head centre within
    ``ground_z`` of the floor, or trunk tilt beyond ``tilt_deg``.  The dead
    mask is per environment; sparks are only spawned for the rendered
    environment (``viewer.env_idx``).
    """

    def __init__(
        self,
        raw_env,
        entity_names: Sequence[str] = ("robot",),
        trunk_body: str = "trunk_base",
        head_body_regex: str = "jaw_soft",
        ground_z: float = 0.06,
        tilt_deg: float = 80.0,
        particles: int = 160,
        seed: int = 20260903,
    ) -> None:
        self.env = raw_env
        self.ground_z = ground_z
        self.tilt_cos = math.cos(math.radians(tilt_deg))
        self.particles = particles
        self.seed = seed
        self.bursts: list[dict[str, np.ndarray]] = []
        self.dead: dict[str, torch.Tensor] = {}
        self.death_step: dict[str, int | None] = {}
        self._bodies: dict[str, tuple[int, int | None]] = {}
        self._step = 0
        renderer = getattr(raw_env, "_offline_renderer", None)
        self.render_env = int(renderer._cfg.env_idx) if renderer is not None else 0
        n, dev = raw_env.num_envs, raw_env.device
        for name in entity_names:
            entity = raw_env.scene[name]
            trunk_id = entity.find_bodies(trunk_body)[0][0]
            try:
                head_ids = entity.find_bodies(head_body_regex)[0] if head_body_regex else []
            except ValueError:
                head_ids = []
            self._bodies[name] = (trunk_id, head_ids[0] if head_ids else None)
            self.dead[name] = torch.zeros(n, dtype=torch.bool, device=dev)
            self.death_step[name] = None
            self._install_power_cut(entity, self.dead[name])
        original = raw_env.update_visualizers
        self._last_counter = -1

        def update_visualizers(_self, visualizer) -> None:
            # Render happens inside env.step(): detect the impact on the
            # post-step state here so the sparks appear in the same frame.
            self.step()
            original(visualizer)
            for burst in self.bursts:
                render_spark_burst(visualizer, burst)

        raw_env.update_visualizers = types.MethodType(update_visualizers, raw_env)

    @staticmethod
    def _install_power_cut(entity, dead: torch.Tensor) -> None:
        original = entity._apply_actuator_controls
        zeros = torch.zeros(entity.num_actuators, device=dead.device)

        def apply_controls(_self) -> None:
            original()
            ids = torch.nonzero(dead).flatten()
            if ids.numel():
                _self.write_ctrl_to_sim(zeros.expand(ids.numel(), -1), env_ids=ids)

        entity._apply_actuator_controls = types.MethodType(apply_controls, entity)

    def step(self) -> None:
        """Advance effects once per env step (idempotent within a step)."""
        counter = int(self.env.common_step_counter)
        if counter == self._last_counter:
            return
        self._last_counter = counter
        self._step += 1
        origins = self.env.scene.env_origins
        for name, (trunk_id, head_id) in self._bodies.items():
            entity = self.env.scene[name]
            com = entity.data.body_com_pos_w
            trunk = com[:, trunk_id] - origins
            lowest = trunk.clone()
            if head_id is not None:
                head = com[:, head_id] - origins
                lowest = torch.where((head[:, 2] < trunk[:, 2]).unsqueeze(-1), head, trunk)
            grounded = lowest[:, 2] < self.ground_z
            tilted = -entity.data.projected_gravity_b[:, 2] < self.tilt_cos
            newly = (grounded | tilted) & ~self.dead[name]
            self.dead[name] |= newly
            if bool(newly[self.render_env]):
                emitter = (lowest[self.render_env] + origins[self.render_env]).detach().cpu().numpy().astype(np.float64)
                emitter[2] = max(float(emitter[2]), float(origins[self.render_env, 2]) + 0.01)
                self.bursts.append(
                    spawn_spark_burst(emitter, self.particles, self.seed + 7919 * len(self.bursts))
                )
                self.death_step[name] = self._step
        floor_z = float(origins[self.render_env, 2])
        for burst in self.bursts:
            advance_spark_burst(burst, self.env.step_dt, floor_z=floor_z)
        self.bursts = [b for b in self.bursts if spark_burst_alive(b)]

    def summary(self) -> dict[str, float | None]:
        dt = self.env.step_dt
        return {f"{name}_dead_at_s": (s * dt if s is not None else None) for name, s in self.death_step.items()}

#!/usr/bin/env python3
"""Film a checkpoint of any registered task (720p, no command arrow, shadow fix).

  uv run python scripts/render_checkpoint.py --task Mjlab-Basketball-MicroDuck \
      --checkpoint-file logs/basketball/<run>/model_800.pt --out-dir eval/bb_b1 \
      --duration-s 8 [--azimuth 135 --elevation -25 --distance 1.2] [--effects]
      [--track-body trunk_base] [--num-envs 1] [--seed 0]

Records env 0 for ``duration_s`` (episodes roll over on resets) and prints the
step at which each termination fired, so clips can be trimmed.  Effects
(sparks + dead motors on ground impact) are opt-in: the presentation rule is
effects on for non-tug clips, off for tug of war.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401 - populate registry
from mjlab_microduck.video_effects import CrashEffects, configure_video_cfg, fix_render_shadows


@dataclass
class Cfg:
    task: str
    checkpoint_file: str
    out_dir: str
    duration_s: float = 8.0
    num_envs: int = 1
    seed: int = 0
    azimuth: float | None = None
    elevation: float | None = None
    distance: float | None = None
    track_body: str | None = None  # viewer body to track (default: the task's)
    effects: bool = False
    entities: str = "robot"  # comma list of duck entities for the effects
    shadowclip: float | None = 4.0
    light_dir: str | None = "auto"  # "x,y,z" or "auto": sun tilted TOWARD the camera (2026-09-07: MuJoCo's directional shadow box
    # clamps to its edge texels, so the floor on the sun's down-light side beyond the box gets a speckled band where its depth
    # crosses the edge depth; tilting the sun toward the camera puts that side behind the viewer)
    light_tilt: float = 0.43  # horizontal / vertical component ratio for "auto"
    train_cfg: bool = False  # film under the training cfg (DR, curricula at level 0)
    tag: str = "clip"
    width: int = 1280
    height: int = 720  # e.g. 720x720 for a square clip (same vertical field of view as 1280x720, centre-cropped)
    follow_entity: str | None = None  # free camera aimed at this entity's root every frame (e.g. "ball")
    follow_dz: float = 0.0  # look-at height above the followed entity's root
    follow_axis: str = "x"  # body axis whose ground heading defines the follow yaw: "x" (nose) or "z" (head; lying entities)
    shadow_extent: float = 4.0  # shadow-map box half-size (m); smaller = denser shadow texels, less far-floor speckle
    shadow_follow: bool = True  # keep the shadow box centred on the followed entity
    fog: str | None = None  # "start,end" metres: subtle depth fog blending the far floor (hides residual shadow speckle)
    follow_yaw_offset: float | None = None  # if set, camera azimuth = followed entity's yaw + this (deg), every frame
    stop_on_done: bool = False  # end a diagnostic take at its first termination


def main(cfg: Cfg) -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.task, play=not cfg.train_cfg)
    basketball_checkpoint = None
    if cfg.task == "Mjlab-Basketball-MicroDuck":
        from mjlab_microduck.basketball_distillation import make_distillation_env_cfg
        basketball_checkpoint = torch.load(cfg.checkpoint_file, map_location="cpu", weights_only=False)
        if "student_state_dict" in basketball_checkpoint:
            import os
            env_cfg = make_distillation_env_cfg(
                play=not cfg.train_cfg, command_scale=float(os.getenv("MICRODUCK_BB_CMD_SCALE", "1")),
            )
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    configure_video_cfg(env_cfg, width=cfg.width, height=cfg.height)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), cfg.duration_s + 1.0)  # no time-out reset inside the clip
    if cfg.track_body:
        env_cfg.viewer.body_name = cfg.track_body
    if cfg.azimuth is not None:
        env_cfg.viewer.azimuth = cfg.azimuth
    if cfg.elevation is not None:
        env_cfg.viewer.elevation = cfg.elevation
    if cfg.distance is not None:
        env_cfg.viewer.distance = cfg.distance
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    if cfg.light_dir == "auto":
        az = math.radians(env_cfg.viewer.azimuth if cfg.azimuth is None else cfg.azimuth)
        light = (cfg.light_tilt * math.cos(az), cfg.light_tilt * math.sin(az), -1.0)
    else:
        light = tuple(float(v) for v in cfg.light_dir.split(",")) if cfg.light_dir else None
    fix_render_shadows(raw_env, light_dir=light)
    if cfg.shadowclip is not None:
        raw_env._offline_renderer._model.vis.map.shadowclip = cfg.shadowclip
    rmodel = raw_env._offline_renderer._model
    rmodel.stat.extent = cfg.shadow_extent
    if cfg.fog:
        import mujoco as _mj

        start, end = (float(v) for v in cfg.fog.split(","))
        raw_env._offline_renderer._renderer.scene.flags[_mj.mjtRndFlag.mjRND_FOG] = 1
        rmodel.vis.map.fogstart, rmodel.vis.map.fogend = start, end
        rmodel.vis.rgba.fog[:] = [0.17, 0.23, 0.34, 1.0]  # the floor's far tone
    names = tuple(n for n in cfg.entities.split(",") if n and "trunk_base" in raw_env.scene[n].body_names)
    effects = CrashEffects(raw_env, entity_names=names, ground_only=False) if cfg.effects and names else None
    from mjlab.utils.wrappers import VideoRecorder

    steps = round(cfg.duration_s / raw_env.step_dt)
    env = VideoRecorder(raw_env, video_folder=Path(cfg.out_dir) / cfg.tag, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True)
    agent_cfg = load_rl_cfg(cfg.task)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()
    if basketball_checkpoint is not None:
        from mjlab_microduck.basketball_distillation import load_actor
        policy = load_actor(basketball_checkpoint, obs, device)
    else:
        runner_cls = load_runner_cls(cfg.task) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(cfg.checkpoint_file, map_location=device)
        policy = runner.get_inference_policy(device=device)
    cam = None
    if cfg.follow_entity:
        import mujoco

        cam = raw_env._offline_renderer._cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1
        cam.distance = env_cfg.viewer.distance
        cam.azimuth = env_cfg.viewer.azimuth
        cam.elevation = env_cfg.viewer.elevation
    events = []
    for i in range(steps):
        if cam is not None:
            ent = raw_env.scene[cfg.follow_entity]
            p = ent.data.root_link_pos_w[0]
            cam.lookat[:] = [float(p[0]), float(p[1]), float(p[2]) + cfg.follow_dz]
            if cfg.shadow_follow:
                rmodel.stat.center[:] = [float(p[0]), float(p[1]), float(p[2])]
            if cfg.follow_yaw_offset is not None:
                w, x, y, z = [float(v) for v in ent.data.root_link_quat_w[0]]
                if cfg.follow_axis == "z":
                    ax = (2.0 * (x * z + w * y), 2.0 * (y * z - w * x))
                else:
                    ax = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z))
                yaw = math.degrees(math.atan2(ax[1], ax[0]))
                cam.azimuth = yaw + cfg.follow_yaw_offset
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, extras = env.step(actions)
        if hasattr(policy, "reset"):
            policy.reset(dones)
        if effects is not None:
            effects.step()
        if bool(dones[0]):
            terms = raw_env.termination_manager
            fired = [k for k in terms.active_terms if bool(terms.get_term(k)[0])] if hasattr(terms, "get_term") else []
            events.append((round(i * raw_env.step_dt, 2), fired))
            if cfg.stop_on_done:
                break
        if (i + 1) % 100 == 0:
            print(f"RENDER_PROGRESS {i + 1}/{steps}", flush=True)
    env.close()
    print(f"RENDER_DONE {Path(cfg.out_dir) / cfg.tag} steps={i + 1} env0_events={events}")


if __name__ == "__main__":
    main(tyro.cli(Cfg))

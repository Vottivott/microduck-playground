"""Headless evaluation of a stair-ladder checkpoint, per curriculum level.

Reports, for climb and hold commands separately: trunk rise in metres and in
risers, fall fraction, reached-top fraction, and body-on-ladder contact time.
Optionally records a video of a few environments.  Example::

    MUJOCO_GL=egl uv run python scripts/evaluate_ladder_checkpoint.py \
        --checkpoint-file logs/ladder_climb/<run>/model_2000.pt --output-file eval.json \
        --video-dir eval_videos
"""

from __future__ import annotations

import os as _os
import sys as _sys

if any("StaircaseDemo" in a for a in _sys.argv):
    _os.environ.setdefault("MICRODUCK_STAIRCASE_DEMO", "1")  # registers the demo task

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401 - populate registry
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.video_effects import CrashEffects, configure_video_cfg, fix_render_shadows
from mjlab_microduck.tasks.microduck_ladder_env_cfg import LADDER_LEVELS


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    task_id: str = "Mjlab-LadderClimb-MicroDuck"
    num_envs: int = 512
    duration_s: float = 8.0
    levels: tuple[int, ...] = tuple(range(len(LADDER_LEVELS)))
    climb_speed: float = 0.04
    floor_spawn_prob: float = 0.5
    seed: int = 123
    output_file: str | None = None
    video_dir: str | None = None
    video_envs: int = 4
    video_level: int | None = None
    # Skip the hold-command battery/video (demo runs: only climbing matters).
    skip_hold: bool = False
    swing_spawn_prob: float = 0.0
    swing_check_s: float = 1.5
    trace: bool = False
    trace_every: int = 5


def _evaluate(cfg: Config, level: int, hold: bool, device: str, record: bool) -> dict:
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed + level
    env_cfg.scene.num_envs = cfg.video_envs if record else cfg.num_envs
    env_cfg.episode_length_s = cfg.duration_s + 1.0
    env_cfg.terminations.clear()
    twist = env_cfg.commands["twist"]
    speed = 0.0 if hold else cfg.climb_speed
    twist.ranges.lin_vel_x = (speed, speed)
    twist.rel_standing_envs = 0.0
    twist.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)
    spawn = env_cfg.events["reset_stair_ladder"].params
    spawn["fixed_level"] = level
    spawn["floor_spawn_prob"] = 0.0 if hold else cfg.floor_spawn_prob
    spawn["swing_spawn_prob"] = 0.0 if hold else cfg.swing_spawn_prob
    if record:
        spawn["floor_spawn_prob"] = 0.5

    if record:
        configure_video_cfg(env_cfg)  # 720p, no command arrows
    agent_cfg = load_rl_cfg(cfg.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array" if record else None)
    env: object = raw_env
    effects = None
    if record:
        from mjlab.utils.wrappers import VideoRecorder  # local import: needs EGL

        effects = CrashEffects(raw_env, entity_names=("robot",))
        tag = "hold" if hold else "climb"
        env = VideoRecorder(
            raw_env,
            video_folder=Path(cfg.video_dir) / f"level{level}_{tag}",
            step_trigger=lambda step: step == 0,
            video_length=round(cfg.duration_s / raw_env.step_dt),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    robot = raw_env.scene["robot"]
    state = raw_env._stair
    if record:
        # Aim the sun down along the ladder incline (ladder leans toward +x)
        # so the off-screen upper treads do not cast detached floor shadows.
        angle0 = float(state.angle[0])
        fix_render_shadows(raw_env, light_dir=(-math.cos(angle0), 0.15, -math.sin(angle0)))
    n = env_cfg.scene.num_envs
    spawn_z = robot.data.root_link_pos_w[:, 2].clone()
    on_floor = state.spawn_on_floor.clone()
    riser = state.riser.clone()
    top_z = state.tread_top[:, -1].clone()
    steps = round(cfg.duration_s / raw_env.step_dt)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    reached = torch.zeros_like(fell)
    body_touch = torch.zeros(n, device=device)
    max_rise = torch.zeros(n, device=device)
    start_tread = state.start_tread.clone()
    swing_is_left = ((start_tread + 1) % 2 != 0)  # lower (swing) foot is left when k+1 is odd
    landed = torch.zeros(n, dtype=torch.bool, device=device)
    retreated = torch.zeros(n, dtype=torch.bool, device=device)
    fall_gravity = torch.zeros(n, 3, device=device)  # body-frame gravity at the fall
    swing_check_step = round(cfg.swing_check_s / raw_env.step_dt)
    servo_ids = microduck_mdp._servo_joint_ids(raw_env, robot) if cfg.trace else None
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if effects is not None:
                effects.step()
        if cfg.trace and not record and step % cfg.trace_every == 0 and step < 200:
            c = microduck_mdp._stair_contacts(raw_env)
            g = robot.data.projected_gravity_b[0]
            sites = microduck_mdp._stair_foot_sites(raw_env, robot)
            fz = robot.data.site_pos_w[0, sites, 2]
            root = robot.data.root_link_pos_w[0] - raw_env.scene.env_origins[0]
            q = robot.data.joint_pos[0, servo_ids]
            print(
                f"TRACE t={step * raw_env.step_dt:.2f} root=({root[0]:+.3f},{root[2]:.3f}) "
                f"grav=({g[0]:+.2f},{g[1]:+.2f}) feet_z=({fz[0]:.3f},{fz[1]:.3f}) "
                f"tread={c['foot_tread'][0].tolist()} sup={c['foot_support'][0].int().tolist()} "
                f"touch={int(c['body_touch'][0])} knees=({q[3]:+.2f},{q[12]:+.2f}) "
                f"hips=({q[2]:+.2f},{q[11]:+.2f}) ankles=({q[4]:+.2f},{q[13]:+.2f}) vz={robot.data.root_link_lin_vel_w[0, 2]:+.3f}"
            )
        if step == swing_check_step:
            foot_tread = microduck_mdp._stair_contacts(raw_env)["foot_tread"]
            swing_tread = torch.where(swing_is_left, foot_tread[:, 0], foot_tread[:, 1])
            landed = swing_tread >= start_tread + 2
            retreated = (swing_tread >= 0) & (swing_tread <= start_tread)
        z = robot.data.root_link_pos_w[:, 2]
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        fall_now = ((tilt > math.radians(65.0)) | microduck_mdp.ladder_fallen(raw_env)) & ~fell
        fall_gravity[fall_now] = robot.data.projected_gravity_b[fall_now]
        fell |= fall_now
        reached |= z > top_z + 0.117 + 0.05
        contacts = microduck_mdp._stair_contacts(raw_env)
        body_touch += contacts["body_touch"].float()
        max_rise = torch.maximum(max_rise, z - spawn_z)
    final_rise = robot.data.root_link_pos_w[:, 2] - spawn_z

    def summarize(mask: torch.Tensor) -> dict:
        if int(mask.sum()) == 0:
            return {"count": 0}
        rise = final_rise[mask]
        return {
            "count": int(mask.sum()),
            "fall_fraction": float(fell[mask].float().mean()),
            "reached_top_fraction": float(reached[mask].float().mean()),
            "final_rise_m_mean": float(rise.mean()),
            "final_rise_m_p50": float(rise.median()),
            "max_rise_risers_p50": float((max_rise[mask] / riser[mask]).median()),
            "max_rise_risers_p90": float((max_rise[mask] / riser[mask]).quantile(0.9)),
            "body_touch_step_fraction": float(body_touch[mask].mean() / steps),
            "swing_landed_fraction": float(landed[mask].float().mean()),
            "swing_retreated_fraction": float(retreated[mask].float().mean()),
            # Fall direction: body-frame gravity x > 0 is a forward (into the
            # ladder) lean, y != 0 sideways.
            "fall_forward_fraction": float(
                (fall_gravity[mask & fell, 0] > 0.3).float().mean()
            ) if bool((mask & fell).any()) else None,
            "fall_backward_fraction": float(
                (fall_gravity[mask & fell, 0] < -0.3).float().mean()
            ) if bool((mask & fell).any()) else None,
            "fall_sideways_fraction": float(
                (fall_gravity[mask & fell, 1].abs() > 0.5).float().mean()
            ) if bool((mask & fell).any()) else None,
        }

    result = {
        "level": level,
        "command": "hold" if hold else f"climb {cfg.climb_speed} m/s",
        "riser_m": (float(riser.min()), float(riser.max())),
        "angle_deg": (float(torch.rad2deg(state.angle).min()), float(torch.rad2deg(state.angle).max())),
        "ladder_spawn": summarize(~on_floor),
        "floor_spawn": summarize(on_floor),
        "all": summarize(torch.ones_like(fell)),
    }
    if record:
        env.close()
    else:
        raw_env.close()
    return result


def main(cfg: Config) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = []
    for level in cfg.levels:
        for hold in ((False,) if cfg.skip_hold else (False, True)):
            r = _evaluate(cfg, level, hold, device, record=False)
            results.append(r)
            a = r["all"]
            print(
                f"LADDER_EVAL level={level} cmd={r['command']} fall={a['fall_fraction']:.3f} "
                f"top={a['reached_top_fraction']:.3f} rise_p50={a['final_rise_m_p50']*1000:.0f}mm "
                f"risers_p50={a['max_rise_risers_p50']:.1f} "
                f"landed={a['swing_landed_fraction']:.2f} retreated={a['swing_retreated_fraction']:.2f}"
            )
    if cfg.video_dir:
        level = cfg.video_level if cfg.video_level is not None else cfg.levels[-1]
        for hold in ((False,) if cfg.skip_hold else (False, True)):
            _evaluate(cfg, level, hold, device, record=True)
        print(f"LADDER_EVAL_VIDEOS {cfg.video_dir}")
    if cfg.output_file:
        with open(cfg.output_file, "w") as f:
            json.dump({"config": asdict(cfg), "results": results}, f, indent=1)


if __name__ == "__main__":
    main(tyro.cli(Config))

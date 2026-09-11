"""Deterministically measure a running checkpoint at an exact command speed."""

import json
import math
from dataclasses import asdict, dataclass

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    task_id: str = "Mjlab-Running-Flat-MicroDuck"
    speed: float = 0.8
    num_envs: int = 512
    duration_s: float = 8.0
    warmup_s: float = 1.0
    fall_tilt_deg: float = 70.0
    seed: int = 123
    output_file: str | None = None


def _quantile(values: torch.Tensor, q: float) -> float | None:
    return float(torch.quantile(values.float(), q)) if values.numel() else None


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return ZYX yaw for scalar-first quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(cfg: Config) -> None:
    if cfg.speed < 0.0:
        raise ValueError("speed must be non-negative")
    if not 0.0 <= cfg.warmup_s < cfg.duration_s:
        raise ValueError("warmup_s must be in [0, duration_s)")

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.duration_s + 1.0
    env_cfg.terminations.clear()
    command_cfg = env_cfg.commands["twist"]
    command_cfg.ranges.lin_vel_x = (cfg.speed, cfg.speed)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_turn_in_place_envs = 0.0
    command_cfg.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)

    agent_cfg = load_rl_cfg(cfg.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = env.get_observations()
    robot = raw_env.scene["robot"]
    num_steps = round(cfg.duration_s / raw_env.step_dt)
    warmup_steps = round(cfg.warmup_s / raw_env.step_dt)
    alive = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(alive)
    speed_sum = torch.zeros(cfg.num_envs, device=device)
    lateral_sum = torch.zeros_like(speed_sum)
    yaw_rate_sum = torch.zeros_like(speed_sum)
    heading_error_sum = torch.zeros_like(speed_sum)
    sample_count = torch.zeros_like(speed_sum)
    flight_steps = torch.zeros_like(speed_sum)
    flight_events = torch.zeros_like(speed_sum)
    airborne_streak = torch.zeros(cfg.num_envs, dtype=torch.long, device=device)
    max_tilt = torch.zeros_like(speed_sum)
    heading_ref = _yaw_from_quat(robot.data.root_link_quat_w).clone()
    position_ref = robot.data.root_link_pos_w[:, :2].clone()
    final_heading_error = torch.zeros_like(speed_sum)

    for step in range(num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        upright_cos = torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
        tilt = torch.acos(upright_cos)
        max_tilt = torch.maximum(max_tilt, tilt)
        fall_now = tilt > math.radians(cfg.fall_tilt_deg)
        fell |= fall_now
        alive &= ~fall_now

        found = raw_env.scene["feet_ground_contact"].data.found
        airborne = ~found.reshape(cfg.num_envs, -1).any(dim=-1)
        airborne_streak = torch.where(
            airborne, airborne_streak + 1, torch.zeros_like(airborne_streak)
        )
        if step >= warmup_steps:
            valid = alive
            body_vel = torch.nan_to_num(robot.data.root_link_lin_vel_b)
            heading = _yaw_from_quat(robot.data.root_link_quat_w)
            heading_error = torch.atan2(
                torch.sin(heading - heading_ref), torch.cos(heading - heading_ref)
            )
            final_heading_error = heading_error
            speed_sum += torch.where(valid, body_vel[:, 0], 0.0)
            lateral_sum += torch.where(valid, body_vel[:, 1].abs(), 0.0)
            yaw_rate_sum += torch.where(
                valid, robot.data.root_link_ang_vel_b[:, 2].abs(), 0.0
            )
            heading_error_sum += torch.where(valid, heading_error.abs(), 0.0)
            sample_count += valid.float()
            flight_steps += (valid & airborne).float()
            flight_events += (valid & (airborne_streak == 3)).float()

    has_samples = sample_count > 0
    mean_speed = speed_sum[has_samples] / sample_count[has_samples]
    displacement_w = robot.data.root_link_pos_w[:, :2] - position_ref
    initial_forward_w = torch.stack(
        (torch.cos(heading_ref), torch.sin(heading_ref)), dim=1
    )
    initial_lateral_w = torch.stack(
        (-torch.sin(heading_ref), torch.cos(heading_ref)), dim=1
    )
    forward_displacement = torch.sum(displacement_w * initial_forward_w, dim=1)
    lateral_displacement = torch.sum(displacement_w * initial_lateral_w, dim=1).abs()
    rollout_duration = num_steps * raw_env.step_dt
    result = {
        "checkpoint": cfg.checkpoint_file,
        "task_id": cfg.task_id,
        "command_speed_mps": cfg.speed,
        "num_envs": cfg.num_envs,
        "duration_s": cfg.duration_s,
        "actor_observation_dim": int(obs["actor"].shape[-1]),
        "survival_fraction": float((~fell).float().mean()),
        "forward_speed_mps": {
            "mean": float(mean_speed.mean()) if mean_speed.numel() else None,
            "p10": _quantile(mean_speed, 0.10),
            "median": _quantile(mean_speed, 0.50),
            "p90": _quantile(mean_speed, 0.90),
            "clean_survivor_mean": (
                float((speed_sum[~fell] / sample_count[~fell]).mean())
                if (~fell).any()
                else None
            ),
        },
        "initial_heading_displacement_speed_mps": {
            "mean": float((forward_displacement / rollout_duration).mean()),
            "median": float(
                torch.quantile(forward_displacement / rollout_duration, 0.5)
            ),
            "p10": float(torch.quantile(forward_displacement / rollout_duration, 0.1)),
        },
        "absolute_lateral_displacement_m": {
            "mean": float(lateral_displacement.mean()),
            "median": float(torch.quantile(lateral_displacement, 0.5)),
            "p90": float(torch.quantile(lateral_displacement, 0.9)),
        },
        "mean_absolute_lateral_speed_mps": (
            float((lateral_sum[has_samples] / sample_count[has_samples]).mean())
            if has_samples.any()
            else None
        ),
        "mean_absolute_yaw_rate_radps": (
            float((yaw_rate_sum[has_samples] / sample_count[has_samples]).mean())
            if has_samples.any()
            else None
        ),
        "mean_absolute_heading_error_deg": (
            math.degrees(
                float(
                    (heading_error_sum[has_samples] / sample_count[has_samples]).mean()
                )
            )
            if has_samples.any()
            else None
        ),
        "final_absolute_heading_error_deg": {
            "mean": math.degrees(float(final_heading_error.abs().mean())),
            "median": math.degrees(
                float(torch.quantile(final_heading_error.abs(), 0.5))
            ),
            "p90": math.degrees(float(torch.quantile(final_heading_error.abs(), 0.9))),
        },
        "flight_fraction": (
            float((flight_steps[has_samples] / sample_count[has_samples]).mean())
            if has_samples.any()
            else None
        ),
        "flight_events_per_second": float(
            flight_events.sum() / max(float(sample_count.sum()) * raw_env.step_dt, 1e-8)
        ),
        "max_tilt_deg_median": math.degrees(float(torch.quantile(max_tilt, 0.5))),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if cfg.output_file:
        with open(cfg.output_file, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))

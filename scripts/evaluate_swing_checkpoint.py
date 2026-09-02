"""Deterministic physical metrics for a learned self-pumping swing policy."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.robot.microduck_constants import (
    SWING_STRING_LENGTH,
    SWING_STRING_STIFFNESS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()

    configure_torch_backends()
    task_id = "Mjlab-SwingPump-MicroDuck"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args.duration + 1.0
    env_cfg.terminations.clear()
    # Physical evaluation deliberately runs through the full requested
    # horizon and reports geometry directly, independent of training debt.
    env_cfg.rewards.pop("invalid_episode_debt", None)
    agent_cfg = load_rl_cfg(task_id)

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    robot = raw.scene["robot"]
    servo_joint_ids, servo_joint_names = robot.find_joints(
        r"^(?!passive_).*", preserve_order=True
    )
    tendon_ids, _ = robot.find_tendons(
        ("swing_string_left", "swing_string_right"), preserve_order=True
    )
    if len(robot.actuators) != 1:
        raise RuntimeError(
            f"Expected one joint-actuator group, found {len(robot.actuators)}"
        )
    bam_actuator = robot.actuators[0]
    if not hasattr(bam_actuator, "diagnostics_enabled"):
        raise RuntimeError("Swing physical audit requires BAM diagnostics support")
    bam_actuator.diagnostics_enabled = True
    actuator_names = list(robot.actuator_names)
    actuator_force_limits = [
        float(raw.sim.mj_model.actuator_forcerange[int(ctrl_id), 1])
        for ctrl_id in robot.indexing.ctrl_ids
    ]

    samples: list[dict[str, float]] = []
    joint_position_history: list[list[float]] = []
    action_history: list[list[float]] = []
    raw_action_history: list[list[float]] = []
    actuator_torque_history: list[list[float]] = []
    duty_applied_history: list[list[float]] = []
    current_limited_history: list[list[bool]] = []
    mechanical_power_history: list[list[float]] = []
    previous_action: torch.Tensor | None = None
    steps = round(args.duration / raw.step_dt)
    for step in range(steps):
        angle, rate, lateral, slack, imbalance = microduck_mdp._swing_kinematics(raw)
        alignment = microduck_mdp.swing_attachment_alignment_penalty(raw)
        predictive_geometry_barrier = (
            microduck_mdp.swing_predictive_geometry_barrier_penalty(raw)
        )
        lengths = robot.data.tendon_len[0, tendon_ids]
        estimated_tensions = SWING_STRING_STIFFNESS * torch.clamp(
            lengths - SWING_STRING_LENGTH, min=0.0
        )
        servo_joint_vel = robot.data.joint_vel[0, servo_joint_ids]
        servo_joint_pos = robot.data.joint_pos[0, servo_joint_ids]
        joint_position_history.append(servo_joint_pos.detach().cpu().tolist())
        root_lateral_velocity = robot.data.root_link_lin_vel_w[0, 1]
        root_angular_velocity = robot.data.root_link_ang_vel_w[0]
        out_of_plane_angular_speed = torch.sqrt(
            root_angular_velocity[0].square() + root_angular_velocity[2].square()
        )
        samples.append(
            {
                "time_s": step * raw.step_dt,
                "angle_rad": float(angle[0]),
                "rate_rad_s": float(rate[0]),
                "equivalent_energy_height_m": float(
                    1.0
                    - torch.cos(angle[0])
                    + 0.5 * (SWING_STRING_LENGTH / 9.81) * rate[0].square()
                ),
                "lateral_m": float(lateral[0]) * 0.38,
                "lateral_velocity_m_s": float(root_lateral_velocity),
                "out_of_plane_angular_speed_rad_s": float(
                    out_of_plane_angular_speed
                ),
                "slack_m": float(slack[0]),
                "imbalance_m": float(imbalance[0]),
                "attachment_alignment_penalty": float(alignment[0]),
                "predictive_geometry_barrier": float(
                    predictive_geometry_barrier[0]
                ),
                "servo_joint_speed_rms_rad_s": float(
                    torch.sqrt(torch.mean(servo_joint_vel.square()))
                ),
                "string_left_m": float(lengths[0]),
                "string_right_m": float(lengths[1]),
                "estimated_string_tension_left_n": float(estimated_tensions[0]),
                "estimated_string_tension_right_n": float(estimated_tensions[1]),
            }
        )
        with torch.inference_mode():
            actions = policy(obs)
        raw_action = actions[0]
        applied_action = torch.clamp(
            raw_action, -agent_cfg.clip_actions, agent_cfg.clip_actions
        )
        raw_action_history.append(raw_action.detach().cpu().tolist())
        action_history.append(applied_action.detach().cpu().tolist())
        samples[-1]["max_abs_raw_action"] = float(raw_action.abs().max())
        samples[-1]["fraction_raw_actions_clipped"] = float(
            (raw_action.abs() > agent_cfg.clip_actions).float().mean()
        )
        samples[-1]["fraction_applied_actions_at_limit"] = float(
            (
                applied_action.abs()
                >= agent_cfg.clip_actions - 1.0e-4
            ).float().mean()
        )
        if previous_action is None:
            action_delta_rms = 0.0
        else:
            action_delta_rms = float(
                torch.sqrt(torch.mean((applied_action - previous_action).square()))
            )
        samples[-1]["action_delta_rms"] = action_delta_rms
        previous_action = applied_action.clone()
        obs, _, _, _ = env.step(actions)

        diagnostics = bam_actuator.last_diagnostics
        if diagnostics is None:
            raise RuntimeError("BAM diagnostics were not populated after stepping")
        torque = diagnostics["motor_torque"][0]
        joint_velocity = diagnostics["joint_velocity"][0]
        duty_unclipped = diagnostics["duty_unclipped"][0]
        duty_current_limited = diagnostics["duty_current_limited"][0]
        duty_applied = diagnostics["duty_applied"][0]
        effective_vin = diagnostics["effective_vin"][0]
        actuator_torque_history.append(torque.detach().cpu().tolist())
        duty_applied_history.append(duty_applied.detach().cpu().tolist())
        current_limited_history.append(
            (
                (duty_current_limited - duty_unclipped).abs() > 1.0e-5
            ).detach().cpu().tolist()
        )
        mechanical_power_history.append(
            (torque * joint_velocity).detach().cpu().tolist()
        )
        samples[-1].update(
            {
                "effective_battery_voltage_v": float(effective_vin.mean()),
                "servo_torque_rms_nm": float(torch.sqrt(torch.mean(torque.square()))),
                "max_abs_servo_torque_nm": float(torque.abs().max()),
                "fraction_servos_pwm_saturated": float(
                    (
                        duty_applied.abs()
                        >= float(bam_actuator._bam_model.actuator.max_pwm) - 1.0e-4
                    ).float().mean()
                ),
                "fraction_servos_current_limited": float(
                    (
                        (duty_current_limited - duty_unclipped).abs() > 1.0e-5
                    ).float().mean()
                ),
                "servo_signed_mechanical_power_w": float(
                    torch.sum(torque * joint_velocity)
                ),
                "servo_absolute_mechanical_power_w": float(
                    torch.sum(torch.abs(torque * joint_velocity))
                ),
            }
        )

    angles = [sample["angle_rad"] for sample in samples]
    rates = [sample["rate_rad_s"] for sample in samples]
    laterals = [sample["lateral_m"] for sample in samples]
    lateral_velocities = [sample["lateral_velocity_m_s"] for sample in samples]
    out_of_plane_angular_speeds = [
        sample["out_of_plane_angular_speed_rad_s"] for sample in samples
    ]
    alignments = [sample["attachment_alignment_penalty"] for sample in samples]
    predictive_geometry_barriers = [
        sample["predictive_geometry_barrier"] for sample in samples
    ]
    joint_speeds = [sample["servo_joint_speed_rms_rad_s"] for sample in samples]
    action_deltas = [sample["action_delta_rms"] for sample in samples[1:]]
    equivalent_energy_heights = [
        sample["equivalent_energy_height_m"] for sample in samples
    ]
    effective_battery_voltages = [
        sample["effective_battery_voltage_v"] for sample in samples
    ]
    servo_torque_rms = [sample["servo_torque_rms_nm"] for sample in samples]
    max_abs_servo_torques = [
        sample["max_abs_servo_torque_nm"] for sample in samples
    ]
    pwm_saturation_fractions = [
        sample["fraction_servos_pwm_saturated"] for sample in samples
    ]
    current_limit_fractions = [
        sample["fraction_servos_current_limited"] for sample in samples
    ]
    absolute_servo_powers = [
        sample["servo_absolute_mechanical_power_w"] for sample in samples
    ]
    signed_servo_powers = [
        sample["servo_signed_mechanical_power_w"] for sample in samples
    ]
    string_tensions = [
        tension
        for sample in samples
        for tension in (
            sample["estimated_string_tension_left_n"],
            sample["estimated_string_tension_right_n"],
        )
    ]
    minimum_string_tensions = [
        min(
            sample["estimated_string_tension_left_n"],
            sample["estimated_string_tension_right_n"],
        )
        for sample in samples
    ]
    maximum_string_tensions = [
        max(
            sample["estimated_string_tension_left_n"],
            sample["estimated_string_tension_right_n"],
        )
        for sample in samples
    ]
    all_lengths = [
        length
        for sample in samples
        for length in (sample["string_left_m"], sample["string_right_m"])
    ]
    minimum_lengths = [
        min(sample["string_left_m"], sample["string_right_m"])
        for sample in samples
    ]
    string_imbalances = [sample["imbalance_m"] for sample in samples]
    maximum_lengths = [
        max(sample["string_left_m"], sample["string_right_m"])
        for sample in samples
    ]

    # Mirror the training latch exactly: geometry debt begins after two
    # consecutive control samples outside either hard gate.  Reporting this
    # separately from violation fractions distinguishes a late, brief failure
    # from a policy that spends most of the rollout in invalid geometry.
    first_sustained_gate_violation_s: float | None = None
    consecutive_invalid_samples = 0
    for sample in samples:
        outside_exact_gate = (
            abs(sample["lateral_m"]) > 0.020
            or sample["attachment_alignment_penalty"] > 0.050
            or max(sample["string_left_m"], sample["string_right_m"]) > 0.394
            or max(sample["string_left_m"], sample["string_right_m"]) < 0.370
        )
        consecutive_invalid_samples = (
            consecutive_invalid_samples + 1 if outside_exact_gate else 0
        )
        if consecutive_invalid_samples >= 2:
            first_sustained_gate_violation_s = sample["time_s"]
            break
    debt_free_duration_s = (
        args.duration
        if first_sustained_gate_violation_s is None
        else first_sustained_gate_violation_s
    )
    crossings: dict[str, float | None] = {}
    for threshold_deg in (5, 15, 30, 45, 60, 90, 120, 150):
        threshold = math.radians(threshold_deg)
        crossing = next(
            (sample["time_s"] for sample in samples if abs(sample["angle_rad"]) >= threshold),
            None,
        )
        crossings[f"{threshold_deg}_deg"] = crossing

    def peak_to_peak_between(start_s: float, end_s: float) -> float:
        window = [
            sample["angle_rad"]
            for sample in samples
            if start_s <= sample["time_s"] < end_s
        ]
        return math.degrees(max(window) - min(window))

    window_s = min(4.0, args.duration / 2.0)
    initial_peak_to_peak = peak_to_peak_between(0.0, window_s)
    final_peak_to_peak = peak_to_peak_between(args.duration - window_s, args.duration)
    rolling_window_step_s = min(1.0, window_s)
    rolling_peak_to_peak: list[dict[str, float]] = []
    rolling_start = 0.0
    while rolling_start + window_s <= args.duration + 1e-9:
        rolling_peak_to_peak.append(
            {
                "start_s": rolling_start,
                "end_s": rolling_start + window_s,
                "peak_to_peak_deg": peak_to_peak_between(
                    rolling_start, rolling_start + window_s
                ),
            }
        )
        rolling_start += rolling_window_step_s
    post_warmup_rolling = [
        window["peak_to_peak_deg"]
        for window in rolling_peak_to_peak
        if window["start_s"] >= window_s
    ]

    turning_peaks: list[dict[str, float]] = []
    for previous, current in zip(samples, samples[1:], strict=False):
        if previous["rate_rad_s"] * current["rate_rad_s"] < 0.0:
            turning_peaks.append(
                {
                    "time_s": current["time_s"],
                    "angle_deg": math.degrees(current["angle_rad"]),
                }
            )
    first_turns = turning_peaks[: min(6, len(turning_peaks))]
    last_turns = turning_peaks[-min(6, len(turning_peaks)) :]

    # Collapse same-side rate reversals into the most extreme peak before
    # measuring a half-cycle. This prevents small high-frequency jitters near
    # one side from masquerading as additional useful swing cycles.
    alternating_peaks: list[dict[str, float]] = []
    for peak in turning_peaks:
        sign = 1 if peak["angle_deg"] >= 0.0 else -1
        if not alternating_peaks:
            alternating_peaks.append(peak)
            continue
        previous_sign = 1 if alternating_peaks[-1]["angle_deg"] >= 0.0 else -1
        if sign == previous_sign:
            if abs(peak["angle_deg"]) > abs(alternating_peaks[-1]["angle_deg"]):
                alternating_peaks[-1] = peak
        else:
            alternating_peaks.append(peak)
    half_cycles = [
        {
            "start_s": previous["time_s"],
            "end_s": current["time_s"],
            "duration_s": current["time_s"] - previous["time_s"],
            "peak_to_peak_deg": abs(current["angle_deg"] - previous["angle_deg"]),
        }
        for previous, current in zip(
            alternating_peaks, alternating_peaks[1:], strict=False
        )
    ]
    final_half_cycles = half_cycles[-min(6, len(half_cycles)) :]

    def rms(values: list[float]) -> float:
        return math.sqrt(sum(value * value for value in values) / len(values))

    raw_action_tensor = torch.tensor(raw_action_history)

    def raw_action_abs_quantile(q: float) -> dict[str, float]:
        values = torch.quantile(raw_action_tensor.abs(), q, dim=0)
        return {
            name: float(values[index])
            for index, name in enumerate(actuator_names)
        }

    joint_name_to_index = {
        name: index for index, name in enumerate(servo_joint_names)
    }
    bilateral_pairs = {
        suffix: (
            joint_name_to_index[f"left_{suffix}"],
            joint_name_to_index[f"right_{suffix}"],
        )
        for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
    }

    def bilateral_residuals(history: list[list[float]]) -> dict[str, dict[str, float]]:
        return {
            suffix: {
                "same_sign_rms": rms(
                    [row[left] - row[right] for row in history]
                ),
                "opposite_sign_rms": rms(
                    [row[left] + row[right] for row in history]
                ),
            }
            for suffix, (left, right) in bilateral_pairs.items()
        }

    def mean_abs_turn(peaks: list[dict[str, float]]) -> float:
        if not peaks:
            return 0.0
        return sum(abs(peak["angle_deg"]) for peak in peaks) / len(peaks)

    summary = {
        "checkpoint": str(args.checkpoint),
        "duration_s": args.duration,
        "control_dt_s": raw.step_dt,
        "max_positive_deg": math.degrees(max(angles)),
        "max_negative_deg": math.degrees(min(angles)),
        "max_abs_deg": math.degrees(max(abs(value) for value in angles)),
        "peak_to_peak_deg": math.degrees(max(angles) - min(angles)),
        "initial_window_s": window_s,
        "initial_peak_to_peak_deg": initial_peak_to_peak,
        "final_peak_to_peak_deg": final_peak_to_peak,
        "peak_to_peak_growth_deg": final_peak_to_peak - initial_peak_to_peak,
        "rolling_window_s": window_s,
        "rolling_window_step_s": rolling_window_step_s,
        "post_warmup_min_rolling_peak_to_peak_deg": (
            min(post_warmup_rolling) if post_warmup_rolling else final_peak_to_peak
        ),
        "post_warmup_mean_rolling_peak_to_peak_deg": (
            sum(post_warmup_rolling) / len(post_warmup_rolling)
            if post_warmup_rolling
            else final_peak_to_peak
        ),
        "rolling_peak_to_peak": rolling_peak_to_peak,
        "max_abs_rate_rad_s": max(abs(value) for value in rates),
        "max_equivalent_energy_height_m": max(equivalent_energy_heights),
        "initial_mean_equivalent_energy_height_m": sum(
            sample["equivalent_energy_height_m"]
            for sample in samples
            if sample["time_s"] < window_s
        )
        / sum(sample["time_s"] < window_s for sample in samples),
        "final_mean_equivalent_energy_height_m": sum(
            sample["equivalent_energy_height_m"]
            for sample in samples
            if sample["time_s"] >= args.duration - window_s
        )
        / sum(
            sample["time_s"] >= args.duration - window_s for sample in samples
        ),
        "max_abs_lateral_m": max(abs(value) for value in laterals),
        "rms_lateral_velocity_m_s": math.sqrt(
            sum(value * value for value in lateral_velocities)
            / len(lateral_velocities)
        ),
        "max_abs_lateral_velocity_m_s": max(
            abs(value) for value in lateral_velocities
        ),
        "fraction_abs_lateral_velocity_above_0_2_m_s": sum(
            abs(value) > 0.2 for value in lateral_velocities
        )
        / len(lateral_velocities),
        "rms_out_of_plane_angular_speed_rad_s": math.sqrt(
            sum(value * value for value in out_of_plane_angular_speeds)
            / len(out_of_plane_angular_speeds)
        ),
        "max_out_of_plane_angular_speed_rad_s": max(
            out_of_plane_angular_speeds
        ),
        "fraction_out_of_plane_angular_speed_above_1_rad_s": sum(
            value > 1.0 for value in out_of_plane_angular_speeds
        )
        / len(out_of_plane_angular_speeds),
        "mean_attachment_alignment_penalty": sum(alignments) / len(alignments),
        "max_attachment_alignment_penalty": max(alignments),
        "mean_predictive_geometry_barrier": (
            sum(predictive_geometry_barriers) / len(predictive_geometry_barriers)
        ),
        "max_predictive_geometry_barrier": max(predictive_geometry_barriers),
        "fraction_predictive_geometry_barrier_above_1": sum(
            value > 1.0 for value in predictive_geometry_barriers
        )
        / len(predictive_geometry_barriers),
        "fraction_any_string_below_0_375_m": sum(
            length < 0.375 for length in minimum_lengths
        )
        / len(minimum_lengths),
        "fraction_any_string_below_0_370_m": sum(
            length < 0.370 for length in minimum_lengths
        )
        / len(minimum_lengths),
        "fraction_both_strings_below_0_375_m": sum(
            sample["string_left_m"] < 0.375
            and sample["string_right_m"] < 0.375
            for sample in samples
        )
        / len(samples),
        "fraction_both_strings_below_0_370_m": sum(
            sample["string_left_m"] < 0.370
            and sample["string_right_m"] < 0.370
            for sample in samples
        )
        / len(samples),
        "fraction_any_string_above_0_394_m": sum(
            length > 0.394 for length in maximum_lengths
        )
        / len(maximum_lengths),
        "fraction_alignment_penalty_above_0_05": sum(
            value > 0.05 for value in alignments
        )
        / len(alignments),
        "fraction_abs_lateral_above_0_02_m": sum(
            abs(value) > 0.02 for value in laterals
        )
        / len(laterals),
        "first_sustained_exact_gate_violation_s": (
            first_sustained_gate_violation_s
        ),
        "debt_free_duration_s": debt_free_duration_s,
        "debt_free_full_horizon": first_sustained_gate_violation_s is None,
        "mean_servo_joint_speed_rms_rad_s": sum(joint_speeds) / len(joint_speeds),
        "max_servo_joint_speed_rms_rad_s": max(joint_speeds),
        "mean_action_delta_rms": sum(action_deltas) / len(action_deltas),
        "max_action_delta_rms": max(action_deltas),
        "max_abs_raw_action": max(
            abs(value) for row in raw_action_history for value in row
        ),
        "fraction_raw_actions_clipped": sum(
            abs(value) > agent_cfg.clip_actions
            for row in raw_action_history
            for value in row
        )
        / (len(raw_action_history) * len(actuator_names)),
        "fraction_applied_actions_at_limit": sum(
            abs(value) >= agent_cfg.clip_actions - 1.0e-4
            for row in action_history
            for value in row
        )
        / (len(action_history) * len(actuator_names)),
        "fraction_raw_action_clipped": {
            name: sum(
                abs(row[index]) > agent_cfg.clip_actions
                for row in raw_action_history
            )
            / len(raw_action_history)
            for index, name in enumerate(actuator_names)
        },
        "raw_action_abs_p50": raw_action_abs_quantile(0.50),
        "raw_action_abs_p90": raw_action_abs_quantile(0.90),
        "raw_action_abs_p99": raw_action_abs_quantile(0.99),
        "actuator_force_limit_nm": dict(
            zip(actuator_names, actuator_force_limits, strict=True)
        ),
        "min_effective_battery_voltage_v": min(effective_battery_voltages),
        "max_effective_battery_voltage_v": max(effective_battery_voltages),
        "mean_servo_torque_rms_nm": sum(servo_torque_rms) / len(servo_torque_rms),
        "max_abs_servo_torque_nm": max(max_abs_servo_torques),
        "fraction_control_steps_with_any_pwm_saturation": sum(
            fraction > 0.0 for fraction in pwm_saturation_fractions
        )
        / len(pwm_saturation_fractions),
        "mean_fraction_servos_pwm_saturated": sum(pwm_saturation_fractions)
        / len(pwm_saturation_fractions),
        "fraction_control_steps_with_any_current_limiting": sum(
            fraction > 0.0 for fraction in current_limit_fractions
        )
        / len(current_limit_fractions),
        "mean_fraction_servos_current_limited": sum(current_limit_fractions)
        / len(current_limit_fractions),
        "mean_absolute_servo_mechanical_power_w": sum(absolute_servo_powers)
        / len(absolute_servo_powers),
        "peak_absolute_servo_mechanical_power_w": max(absolute_servo_powers),
        "mean_signed_servo_mechanical_power_w": sum(signed_servo_powers)
        / len(signed_servo_powers),
        "max_abs_torque_nm": {
            name: max(abs(row[index]) for row in actuator_torque_history)
            for index, name in enumerate(actuator_names)
        },
        "fraction_pwm_saturated": {
            name: sum(abs(row[index]) >= 1.0 - 1.0e-4 for row in duty_applied_history)
            / len(duty_applied_history)
            for index, name in enumerate(actuator_names)
        },
        "fraction_current_limited": {
            name: sum(row[index] for row in current_limited_history)
            / len(current_limited_history)
            for index, name in enumerate(actuator_names)
        },
        "mean_signed_mechanical_power_w": {
            name: sum(row[index] for row in mechanical_power_history)
            / len(mechanical_power_history)
            for index, name in enumerate(actuator_names)
        },
        "mean_absolute_mechanical_power_w": {
            name: sum(abs(row[index]) for row in mechanical_power_history)
            / len(mechanical_power_history)
            for index, name in enumerate(actuator_names)
        },
        "rms_joint_position_rad": {
            name: rms([row[index] for row in joint_position_history])
            for index, name in enumerate(servo_joint_names)
        },
        "rms_action": {
            name: rms([row[index] for row in action_history])
            for index, name in enumerate(servo_joint_names)
        },
        "bilateral_joint_position_residual_rms_rad": bilateral_residuals(
            joint_position_history
        ),
        "bilateral_action_residual_rms": bilateral_residuals(action_history),
        "turning_peak_count": len(turning_peaks),
        "first_six_mean_abs_turning_peak_deg": mean_abs_turn(first_turns),
        "last_six_mean_abs_turning_peak_deg": mean_abs_turn(last_turns),
        "turning_peaks": turning_peaks,
        "alternating_turning_peak_count": len(alternating_peaks),
        "half_cycle_count": len(half_cycles),
        "mean_half_cycle_duration_s": (
            sum(cycle["duration_s"] for cycle in half_cycles) / len(half_cycles)
            if half_cycles
            else 0.0
        ),
        "final_six_mean_half_cycle_peak_to_peak_deg": (
            sum(cycle["peak_to_peak_deg"] for cycle in final_half_cycles)
            / len(final_half_cycles)
            if final_half_cycles
            else 0.0
        ),
        "half_cycles": half_cycles,
        "min_string_length_m": min(all_lengths),
        "max_string_length_m": max(all_lengths),
        "mean_string_length_imbalance_m": sum(string_imbalances)
        / len(string_imbalances),
        "max_string_length_imbalance_m": max(string_imbalances),
        "estimated_string_tension_model": {
            "dead_band_length_m": SWING_STRING_LENGTH,
            "stiffness_n_per_m": SWING_STRING_STIFFNESS,
            "note": "Elastic spring tension only; strict rollouts remain below the 0.395 m safety-limit constraint.",
        },
        "min_estimated_string_tension_n": min(string_tensions),
        "max_estimated_string_tension_n": max(string_tensions),
        "mean_max_string_tension_n": sum(maximum_string_tensions)
        / len(maximum_string_tensions),
        "fraction_any_string_below_1_n_tension": sum(
            value < 1.0 for value in minimum_string_tensions
        )
        / len(minimum_string_tensions),
        "fraction_both_strings_below_1_n_tension": sum(
            sample["estimated_string_tension_left_n"] < 1.0
            and sample["estimated_string_tension_right_n"] < 1.0
            for sample in samples
        )
        / len(samples),
        "first_crossings_s": crossings,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
    raw.close()


if __name__ == "__main__":
    main()

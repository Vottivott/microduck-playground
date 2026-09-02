"""Regression tests for the forward-running task."""

from itertools import pairwise

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_running_env_cfg import (
    RUNNING_CURRICULUM_START_ITERATION,
    RUNNING_ENABLE_HEADING_FEEDBACK,
    RUNNING_FINAL_ACTION_RATE_WEIGHT,
    RUNNING_FORWARD_PROGRESS_WEIGHT,
    RUNNING_HIGH_SPEED_STAGE_INTERVAL,
    RUNNING_SPEED_STAGES,
    RUNNING_TARGET_MAX_SPEED,
    MicroduckRunningFlightRlCfg,
    MicroduckRunningRlCfg,
    _running_speed_stages,
    make_microduck_running_env_cfg,
)


def test_forward_progress_is_linear_monotonic_and_capped():
    velocity = torch.tensor([-1.0, 0.0, 0.35, 0.70, 1.40, 3.0, float("nan")])
    reward = mdp.running_forward_progress_from_velocity(velocity, speed_cap=1.4)
    torch.testing.assert_close(
        reward,
        torch.tensor([0.0, 0.0, 0.25, 0.50, 1.0, 1.0, 0.0]),
    )
    with pytest.raises(ValueError):
        mdp.running_forward_progress_from_velocity(velocity, speed_cap=0.0)


def test_planar_drift_cost_penalizes_yaw_and_lateral_error():
    cost = mdp.running_planar_drift_cost_from_values(
        lateral_velocity=torch.tensor([0.0, 0.2, 0.0]),
        yaw_rate=torch.tensor([0.0, 0.0, 2.0]),
        lateral_command=torch.zeros(3),
        yaw_command=torch.zeros(3),
        lateral_weight=4.0,
    )
    torch.testing.assert_close(cost, torch.tensor([0.0, 0.16, 4.0]))


def test_running_cfg_keeps_forward_command_and_idle_bucket():
    cfg = make_microduck_running_env_cfg()
    command = cfg.commands["twist"]
    first = RUNNING_SPEED_STAGES[0]
    assert command.ranges.lin_vel_x == (first["min_speed"], first["max_speed"])
    assert command.ranges.lin_vel_x[0] > 0.0
    assert command.ranges.lin_vel_y == (-0.02, 0.02)
    expected_yaw_range = (
        (0.0, 0.0) if RUNNING_ENABLE_HEADING_FEEDBACK else (-0.05, 0.05)
    )
    assert command.ranges.ang_vel_z == expected_yaw_range
    if RUNNING_ENABLE_HEADING_FEEDBACK:
        assert isinstance(command, mdp.SpawnHeadingVelocityCommandCfg)
    assert 0.0 < command.rel_standing_envs < 0.1
    assert command.rel_turn_in_place_envs == 0.0


def test_running_reward_stack_allows_dynamic_motion():
    cfg = make_microduck_running_env_cfg()
    assert cfg.rewards["forward_progress"].weight == RUNNING_FORWARD_PROGRESS_WEIGHT
    assert cfg.rewards["track_angular_velocity"].weight == 0.5
    assert cfg.rewards["planar_drift"].weight < 0.0
    assert cfg.rewards["heading_hold"].weight == 1.5
    assert cfg.rewards["heading_hold"].params["std"] == 0.4
    assert cfg.rewards["air_time"].weight == 0.0
    assert cfg.rewards["pose"].weight < 0.25
    assert cfg.rewards["upright"].weight < 1.0
    assert cfg.rewards["body_ang_vel"].weight < 0.0
    assert cfg.rewards["angular_momentum"].weight < 0.0
    assert cfg.rewards["action_rate_l2"].weight < 0.0
    assert cfg.rewards["head_pose_tracking"].weight == 0.0
    assert cfg.rewards["head_pose_bias"].weight == 0.0
    assert "push_robot" not in cfg.events
    assert "flight_event" not in cfg.rewards


def test_flight_ablation_changes_only_the_intended_objective():
    plain = make_microduck_running_env_cfg()
    flight = make_microduck_running_env_cfg(flight_reward_weight=1.5)
    assert flight.rewards["flight_event"].weight == 1.5
    assert flight.rewards["flight_event"].params["min_airborne_steps"] == 3
    for name, term in plain.rewards.items():
        assert flight.rewards[name].weight == term.weight


def test_speed_and_smoothness_curricula_are_delayed():
    assert RUNNING_CURRICULUM_START_ITERATION == 0
    cfg = make_microduck_running_env_cfg()
    speed_stages = cfg.curriculum["running_speed_range"].params["speed_stages"]
    assert all(a["step"] < b["step"] for a, b in pairwise(speed_stages))
    assert all(a["min_speed"] < b["min_speed"] for a, b in pairwise(speed_stages))
    assert all(a["max_speed"] < b["max_speed"] for a, b in pairwise(speed_stages))

    smooth = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert smooth[0] == {"step": 0, "weight": -0.02}
    assert smooth[1]["step"] >= 2000 * 24
    assert smooth[-1]["weight"] == RUNNING_FINAL_ACTION_RATE_WEIGHT
    assert RUNNING_SPEED_STAGES[-1]["max_speed"] == RUNNING_TARGET_MAX_SPEED


def test_continuation_speed_curriculum_ramps_in_three_stages():
    stages = _running_speed_stages(1.65)
    assert stages[-3]["max_speed"] == 1.35
    assert stages[-2]["max_speed"] == 1.50
    assert stages[-1]["max_speed"] == 1.65
    assert stages[-1]["step"] == 7750 * 24


def test_high_speed_continuation_starts_after_winning_checkpoint():
    stages = _running_speed_stages(2.2)
    continuation = stages[-4:]
    assert [stage["max_speed"] for stage in continuation] == [
        1.8,
        1.95,
        2.1,
        2.2,
    ]
    assert [stage["step"] // 24 for stage in continuation] == [
        8750,
        9250,
        9750,
        10250,
    ]
    assert all(a["min_speed"] < b["min_speed"] for a, b in pairwise(continuation))


def test_high_speed_continuation_interval_is_configurable():
    stages = _running_speed_stages(2.1, high_speed_stage_interval=750)
    continuation = stages[-3:]
    assert [stage["step"] // 24 for stage in continuation] == [8750, 9500, 10250]


def test_high_speed_continuation_rejects_invalid_interval():
    with pytest.raises(ValueError, match="stage interval"):
        _running_speed_stages(2.0, high_speed_stage_interval=0)


def test_running_progress_weight_is_positive():
    assert RUNNING_FORWARD_PROGRESS_WEIGHT > 0.0
    assert RUNNING_HIGH_SPEED_STAGE_INTERVAL > 0


def test_play_cfg_is_exact_forward_speed_without_training_curricula():
    cfg = make_microduck_running_env_cfg(play=True)
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x[0] == command.ranges.lin_vel_x[1] > 0.0
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.rel_standing_envs == 0.0
    assert "running_speed_range" not in cfg.curriculum
    assert "action_rate_weight" not in cfg.curriculum
    assert cfg.viewer.distance <= 0.6
    assert cfg.viewer.max_extra_envs == 0


def test_runner_families_have_distinct_artifact_directories():
    assert MicroduckRunningRlCfg.experiment_name == "running"
    assert MicroduckRunningFlightRlCfg.experiment_name == "running_flight"
    assert (
        MicroduckRunningRlCfg.experiment_name
        != MicroduckRunningFlightRlCfg.experiment_name
    )

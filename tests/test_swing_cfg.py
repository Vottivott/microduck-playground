"""Structural tests for the tension-only MicroDuck swing task."""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot.microduck_constants import (
    SWING_BOTTOM_TRUNK_Z,
    SWING_HANG_LENGTH,
    SWING_STRING_LIMIT,
    SWING_STRING_LENGTH,
    SWING_STRING_STIFFNESS,
    get_swing_spec,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_swing_env_cfg import (
    MicroduckSwingRlCfg,
    make_microduck_swing_env_cfg,
)
from mjlab_microduck.tasks.swing_planar_ppo import (
    PLANAR_ACTION_INDICES,
    _configured_action_indices,
    swing_tail_environment_weights,
)


def test_swing_model_uses_two_length_limited_spatial_tendons() -> None:
    model = get_swing_spec().compile()
    assert model.ntendon == 2
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        for i in range(model.ntendon)
    ] == ["swing_string_left", "swing_string_right"]
    np.testing.assert_allclose(
        model.tendon_range,
        [[0.0, SWING_STRING_LIMIT], [0.0, SWING_STRING_LIMIT]],
    )
    assert np.all(model.tendon_limited)
    np.testing.assert_allclose(model.tendon_stiffness, SWING_STRING_STIFFNESS)
    np.testing.assert_allclose(
        model.tendon_lengthspring,
        [[0.0, SWING_STRING_LENGTH], [0.0, SWING_STRING_LENGTH]],
    )

    # Rendering is provided by the tendons themselves. A capsule here would
    # turn the string back into a static rod and invalidate the task physics.
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    }
    assert not any(name and "string" in name for name in geom_names)


def test_hanging_spawn_is_exactly_taut_and_still() -> None:
    model = get_swing_spec().compile()
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    qpos_adr = model.jnt_qposadr[root_id]
    np.testing.assert_allclose(
        model.qpos0[qpos_adr : qpos_adr + 3],
        (0.0, 0.0, SWING_BOTTOM_TRUNK_Z),
        atol=1e-9,
    )
    data.qpos[qpos_adr : qpos_adr + 7] = (
        0.0,
        0.0,
        SWING_BOTTOM_TRUNK_Z,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.ten_length, SWING_HANG_LENGTH, atol=1e-9)
    np.testing.assert_allclose(data.qvel, 0.0)


def test_seat_mass_is_in_the_dynamic_model() -> None:
    model = get_swing_spec().compile()
    payload_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "swing_seat_payload"
    )
    assert math.isclose(model.body_mass[payload_id], 0.158, abs_tol=1e-6)


def test_swing_plane_heading_ignores_pitch_but_observes_yaw_and_roll() -> None:
    half = math.pi / 8.0
    quaternions = torch.tensor(
        [
            [math.cos(half), 0.0, math.sin(half), 0.0],  # pitch
            [math.cos(half), 0.0, 0.0, math.sin(half)],  # yaw
            [math.cos(half), math.sin(half), 0.0, 0.0],  # roll
        ],
        dtype=torch.float32,
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(root_link_quat_w=quaternions),
    )
    env = SimpleNamespace(scene={"robot": robot})
    heading = microduck_mdp.swing_plane_heading_observation(env)
    torch.testing.assert_close(heading[0], torch.zeros(3), atol=1e-6, rtol=0.0)
    assert abs(float(heading[1, 1])) > 0.5
    assert abs(float(heading[2, 2])) > 0.5


def test_swing_positive_reward_validity_gate_is_strict_and_smooth() -> None:
    factor = microduck_mdp.swing_physical_validity_factor(
        lateral_m=torch.tensor([0.0, 0.012, 0.020, 0.012]),
        alignment_penalty=torch.tensor([0.0, 0.04, 0.04, 0.05]),
        deepest_string_slack_m=torch.tensor([0.0, 0.0095, 0.0095, 0.0095]),
    )
    torch.testing.assert_close(factor[:2], torch.ones(2))
    assert 0.0 < float(factor[2]) < 0.02
    assert 0.0 < float(factor[3]) < 0.02

    slack_factor = microduck_mdp.swing_physical_validity_factor(
        lateral_m=torch.tensor([0.0, 0.0]),
        alignment_penalty=torch.tensor([0.0, 0.0]),
        deepest_string_slack_m=torch.tensor([0.0095, 0.0155]),
    )
    assert float(slack_factor[0]) == 1.0
    assert 0.0 < float(slack_factor[1]) < 0.02


def test_swing_invalid_geometry_latches_debt_without_termination(monkeypatch) -> None:
    zeros = torch.zeros(2)
    lateral = torch.tensor([0.025 / 0.38, 0.0])
    alignment = torch.tensor([0.0, 0.07])
    monkeypatch.setattr(
        microduck_mdp,
        "_swing_kinematics",
        lambda env, asset_cfg: (zeros, zeros, lateral, zeros, zeros),
    )
    monkeypatch.setattr(
        microduck_mdp,
        "swing_attachment_alignment_penalty",
        lambda env, asset_cfg: alignment,
    )
    monkeypatch.setattr(
        microduck_mdp,
        "_swing_string_lengths",
        lambda env, asset_cfg: torch.full((2, 2), 0.38),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        episode_length_buf=torch.tensor([2, 2]),
        common_step_counter=1,
    )
    assert not torch.any(microduck_mdp.swing_invalid_episode_debt(env))
    # Multiple reward terms in one control step must not advance the counter.
    assert not torch.any(microduck_mdp.swing_invalid_episode_debt(env))
    env.common_step_counter = 2
    debt = microduck_mdp.swing_invalid_episode_debt(env)
    assert torch.all(debt > 1.0)

    lateral.zero_()
    alignment.zero_()
    env.common_step_counter = 3
    torch.testing.assert_close(
        microduck_mdp.swing_invalid_episode_debt(env), torch.ones(2)
    )

    env.episode_length_buf[:] = 1
    env.common_step_counter = 4
    assert not torch.any(microduck_mdp.swing_invalid_episode_debt(env))


def test_swing_predictive_geometry_barrier_anticipates_exact_gates() -> None:
    penalty = microduck_mdp.swing_predictive_geometry_barrier_from_values(
        predicted_lateral_m=torch.tensor([0.0, 0.012, 0.020, 0.012]),
        predicted_alignment_penalty=torch.tensor([0.0, 0.04, 0.04, 0.05]),
    )
    torch.testing.assert_close(penalty[:2], torch.zeros(2))
    torch.testing.assert_close(penalty[2:], torch.ones(2))
    capped = microduck_mdp.swing_predictive_geometry_barrier_from_values(
        predicted_lateral_m=torch.tensor([1.0]),
        predicted_alignment_penalty=torch.tensor([2.0]),
    )
    torch.testing.assert_close(capped, torch.tensor([16.0]))

    extension = microduck_mdp.swing_string_extension_barrier_from_lengths(
        torch.tensor([[0.390, 0.392], [0.394, 0.391], [0.410, 0.380]])
    )
    torch.testing.assert_close(extension, torch.tensor([0.0, 1.0, 16.0]))


def test_swing_tail_weighting_selects_worst_exact_geometry(monkeypatch) -> None:
    validity = torch.zeros(2, 4, 6)
    validity[0, :, 0] = torch.tensor([0.1, 0.9, 0.2, 0.4])
    validity[1, 2, 4] = 1.2
    weights, severity, selected = swing_tail_environment_weights(
        validity, fraction=0.5, weight=4.0
    )
    assert set(selected.tolist()) == {1, 2}
    torch.testing.assert_close(severity, torch.tensor([0.1, 0.9, 1.2, 0.4]))
    torch.testing.assert_close(weights, torch.tensor([1.0, 4.0, 4.0, 1.0]))

    monkeypatch.setenv("MICRODUCK_SWING_PLANAR_ACTION_INDICES", "10,0,9,1")
    assert _configured_action_indices() == (0, 1, 9, 10)


def test_swing_task_preserves_actor_contract_and_exact_reset() -> None:
    cfg = make_microduck_swing_env_cfg()
    assert cfg.scene.env_spacing == 0.0
    assert cfg.viewer.origin_type == cfg.viewer.OriginType.WORLD
    actor_terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in actor_terms
    assert actor_terms["head_command"].params["dim"] == 4
    assert actor_terms["body_command"].params["dim"] == 6
    assert "swing_state" not in actor_terms
    assert "swing_state" in cfg.observations["critic"].terms
    assert (
        cfg.observations["critic"].terms["body_command"].func
        is microduck_mdp.swing_validity_observation
    )
    assert (
        actor_terms["body_command"].func
        is microduck_mdp.zero_command_padding
    )
    assert (
        actor_terms["command"].func
        is microduck_mdp.swing_plane_heading_observation
    )
    assert (
        cfg.observations["critic"].terms["command"].func
        is microduck_mdp.swing_frontier_observation
    )
    assert cfg.rewards["swing_lateral"].params["tolerance_m"] == 0.03
    assert cfg.rewards["swing_lateral"].weight == -3.0
    assert cfg.rewards["swing_lateral_barrier"].params == {
        "safe_m": 0.012,
        "scale_m": 0.008,
    }
    assert cfg.rewards["swing_lateral_barrier"].weight == -8.0
    assert cfg.rewards["swing_lateral_velocity"].params == {
        "tolerance_m_s": 0.1,
    }
    assert cfg.rewards["swing_lateral_velocity"].weight == -3.0
    assert cfg.rewards["swing_out_of_plane_angular_velocity"].params == {
        "tolerance_rad_s": 1.5,
    }
    assert cfg.rewards["swing_out_of_plane_angular_velocity"].weight == -1.0
    assert cfg.rewards["swing_alignment"].weight == -18.0
    assert cfg.rewards["swing_alignment_barrier"].params == {
        "safe_penalty": 0.04,
        "scale": 0.01,
    }
    assert cfg.rewards["swing_alignment_barrier"].weight == -4.0
    assert cfg.rewards["swing_predictive_geometry_barrier"].params == {
        "horizon_s": 0.08,
        "lateral_safe_m": 0.012,
        "lateral_scale_m": 0.008,
        "alignment_safe_penalty": 0.04,
        "alignment_scale": 0.01,
        "max_penalty": 16.0,
    }
    assert cfg.rewards["swing_predictive_geometry_barrier"].weight == -6.0
    assert cfg.rewards["string_slack"].weight == -8.0
    assert cfg.rewards["string_extension_barrier"].params == {
        "safe_length_m": 0.392,
        "scale_m": 0.002,
        "max_penalty": 16.0,
    }
    assert cfg.rewards["string_extension_barrier"].weight == -12.0
    assert cfg.rewards["invalid_episode_debt"].weight == -10.0
    assert cfg.rewards["swing_peak_progress"].params["frontier_power"] == 2.0
    assert cfg.rewards["swing_peak_progress"].weight == 224.0
    assert cfg.rewards["swing_energy"].weight == 0.5
    assert cfg.rewards["swing_late_height"].weight == 24.0
    assert cfg.rewards["swing_late_height"].params["time_power"] == 2.0
    assert MicroduckSwingRlCfg.algorithm.learning_rate == 1.0e-4
    assert MicroduckSwingRlCfg.algorithm.schedule == "fixed"
    assert MicroduckSwingRlCfg.algorithm.clip_param == 0.1
    assert MicroduckSwingRlCfg.algorithm.num_learning_epochs == 3
    assert MicroduckSwingRlCfg.algorithm.entropy_coef == 0.002
    assert MicroduckSwingRlCfg.algorithm.class_name.endswith(
        ":SwingPlanarCorrectionPPO"
    )
    assert PLANAR_ACTION_INDICES == (0, 1, 7, 8, 9, 10)

    reset = cfg.events["reset_base"]
    assert reset.params["pose_range"] == {}
    assert reset.params["velocity_range"] == {}
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x == (0.0, 0.0)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert "invalid_geometry" not in cfg.terminations
    assert make_microduck_swing_env_cfg(play=True).seed == 72


def test_nominal_actuator_limit_mode_fixes_conservative_midpoints(monkeypatch) -> None:
    monkeypatch.setenv("MICRODUCK_SWING_NOMINAL_ACTUATOR", "1")
    nominal = make_microduck_swing_env_cfg()
    actuator = nominal.scene.entities["robot"].articulation.actuators[0]
    assert actuator.vin_range == (7.35, 7.35)
    assert actuator.vin_drop_gain_range == (0.10, 0.10)
    assert actuator.delay_min_lag == 5
    assert actuator.delay_max_lag == 5

    monkeypatch.delenv("MICRODUCK_SWING_NOMINAL_ACTUATOR")
    randomized = make_microduck_swing_env_cfg()
    randomized_actuator = randomized.scene.entities["robot"].articulation.actuators[0]
    assert randomized_actuator.vin_range == (6.5, 8.2)
    assert randomized_actuator.vin_drop_gain_range == (0.0, 0.2)
    assert randomized_actuator.delay_min_lag == 3
    assert randomized_actuator.delay_max_lag == 6

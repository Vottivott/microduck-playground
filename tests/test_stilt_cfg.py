"""Regression tests for the staged stilt locomotion task."""

from collections import Counter

import mujoco
import pytest
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.robot.stilt_constants import (
    default_stilt_mass_kg,
    get_stilt_walk_spec,
    stilt_mesh_data,
    stilt_profile_dimensions,
    validate_stilt_morphology,
)
from mjlab_microduck.tasks.microduck_stilt_env_cfg import (
    MicroduckStiltRlCfg,
    make_microduck_stilt_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def _compiled_stilt_model(height_cm: float = 2.0, blend: float = 0.0):
    cfg = make_microduck_stilt_env_cfg(height_cm=height_cm, blend=blend)
    robot_cfg = cfg.scene.entities["robot"]
    spec = robot_cfg.spec_fn()
    for collision_cfg in robot_cfg.collisions:
        collision_cfg.edit_spec(spec)
    return spec.compile()


def test_profile_interpolates_platform_to_round_peg():
    platform = stilt_profile_dimensions(0.0)
    middle = stilt_profile_dimensions(0.5)
    peg = stilt_profile_dimensions(1.0)
    assert (platform["tip_width_mm"], platform["tip_length_mm"]) == (22.0, 32.0)
    assert (middle["tip_width_mm"], middle["tip_length_mm"]) == (17.0, 22.0)
    assert (peg["tip_width_mm"], peg["tip_length_mm"]) == (12.0, 12.0)
    assert peg["tip_radius_mm"] == 6.0


def test_generated_loft_is_closed_and_reaches_exact_height():
    vertices_flat, faces_flat = stilt_mesh_data(25.0, 1.0)
    vertices = list(zip(*(iter(vertices_flat),) * 3, strict=True))
    faces = list(zip(*(iter(faces_flat),) * 3, strict=True))
    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        for start, end in zip(face, face[1:] + face[:1], strict=True):
            edge_counts[tuple(sorted((start, end)))] += 1
    assert set(edge_counts.values()) == {2}
    z_values = [vertex[2] for vertex in vertices]
    assert min(z_values) == pytest.approx(-0.250)
    assert max(z_values) == pytest.approx(0.0)


def test_morphology_validation_and_mass_schedule():
    validate_stilt_morphology(0.8, 0.0)
    validate_stilt_morphology(300.0, 1.0)
    with pytest.raises(ValueError):
        validate_stilt_morphology(300.1, 1.0)
    with pytest.raises(ValueError):
        validate_stilt_morphology(2.0, 1.01)
    assert default_stilt_mass_kg(2.0) == pytest.approx(0.014)
    assert default_stilt_mass_kg(25.0) == pytest.approx(0.037)
    assert default_stilt_mass_kg(50.0) == pytest.approx(0.062)
    assert default_stilt_mass_kg(100.0) == pytest.approx(0.112)
    assert default_stilt_mass_kg(200.0) == pytest.approx(0.212)
    assert default_stilt_mass_kg(210.0) == pytest.approx(0.222)
    assert default_stilt_mass_kg(300.0) == pytest.approx(0.312)


def test_extended_height_mesh_reaches_50_cm_tip():
    vertices_flat, _ = stilt_mesh_data(50.0, 0.5)
    z_values = vertices_flat[2::3]
    assert min(z_values) == pytest.approx(-0.500)


def test_extended_height_mesh_reaches_100_cm_tip():
    vertices_flat, _ = stilt_mesh_data(100.0, 0.5)
    z_values = vertices_flat[2::3]
    assert min(z_values) == pytest.approx(-1.000)


def test_extended_height_mesh_reaches_200_cm_tip():
    vertices_flat, _ = stilt_mesh_data(200.0, 0.5)
    z_values = vertices_flat[2::3]
    assert min(z_values) == pytest.approx(-2.000)


def test_extended_height_mesh_reaches_210_cm_tip():
    vertices_flat, _ = stilt_mesh_data(210.0, 0.5)
    z_values = vertices_flat[2::3]
    assert min(z_values) == pytest.approx(-2.100)


def test_extended_height_mesh_reaches_300_cm_tip():
    vertices_flat, _ = stilt_mesh_data(300.0, 0.5)
    z_values = vertices_flat[2::3]
    assert min(z_values) == pytest.approx(-3.000)


def test_compiled_robot_uses_stilts_not_original_soles_for_contact():
    model = _compiled_stilt_model()
    for side in ("left", "right"):
        collision_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision"
        )
        old_sole_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_original_sole_disabled"
        )
        assert model.geom_type[collision_id] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_contype[collision_id] == 1
        assert model.geom_conaffinity[collision_id] == 1
        assert model.geom_contype[old_sole_id] == 0
        assert model.geom_conaffinity[old_sole_id] == 0
    assert model.nu == 14


def test_compiled_robot_accepts_measured_10cm_print_mass():
    spec = get_stilt_walk_spec(height_cm=10.0, blend=0.5, mass_kg=0.029)
    model = spec.compile()
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"stilt_{side}")
        assert model.body_mass[body_id] == pytest.approx(0.029)


def test_tip_sites_move_to_ground_contact_plane():
    spec = get_stilt_walk_spec(height_cm=25.0, blend=1.0)
    model = spec.compile()
    for side in ("left", "right"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        assert model.site_pos[site_id][2] == pytest.approx(-0.250)


def test_env_raises_reset_height_and_keeps_obs_layout():
    stilt = make_microduck_stilt_env_cfg(height_cm=2.0, blend=0.0)
    velocity = make_microduck_velocity_env_cfg()
    assert stilt.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (0.14, 0.15)
    )
    for group in ("actor", "critic"):
        assert list(stilt.observations[group].terms) == list(
            velocity.observations[group].terms
        )


def test_stage_zero_prioritizes_balance_and_gait_discovery():
    cfg = make_microduck_stilt_env_cfg(height_cm=2.0, blend=0.0)
    command = cfg.commands["twist"]
    assert command.rel_standing_envs == 0.25
    assert command.ranges.lin_vel_x == (-0.12, 0.25)
    assert cfg.rewards["upright"].weight == 3.0
    assert cfg.rewards["action_rate_l2"].weight == -0.02
    assert cfg.rewards["foot_slip"].weight < 0.0
    assert "push_robot" not in cfg.events
    assert "head_pose_range" not in cfg.curriculum
    smoothness = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert smoothness[0] == {"step": 0, "weight": -0.02}
    assert smoothness[-1]["step"] == 4000 * 24


def test_play_cfg_is_deterministic_and_has_no_training_curriculum():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=25.0, blend=1.0)
    command = cfg.commands["twist"]
    assert command.ranges.lin_vel_x[0] == command.ranges.lin_vel_x[1] > 0.0
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert "action_rate_weight" not in cfg.curriculum
    assert cfg.viewer.distance >= 0.9


def test_extended_play_cfg_tracks_50_cm_height():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=50.0, blend=0.5)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (0.62, 0.63)
    )
    assert cfg.viewer.distance >= 1.4


def test_extended_play_cfg_tracks_100_cm_height():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=100.0, blend=0.5)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (1.12, 1.13)
    )
    assert cfg.viewer.distance >= 2.4


def test_extended_play_cfg_tracks_200_cm_height():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=200.0, blend=0.5)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (2.12, 2.13)
    )
    assert cfg.viewer.distance >= 4.4


def test_extended_play_cfg_tracks_210_cm_height():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=210.0, blend=0.5)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (2.22, 2.23)
    )
    assert cfg.viewer.distance >= 4.6


def test_extended_play_cfg_tracks_300_cm_height():
    cfg = make_microduck_stilt_env_cfg(play=True, height_cm=300.0, blend=0.5)
    assert cfg.events["reset_base"].params["pose_range"]["z"] == pytest.approx(
        (3.12, 3.13)
    )
    assert cfg.viewer.distance >= 6.4


def test_task_registration_and_artifact_family():
    assert "Mjlab-Stilt-Flat-MicroDuck" in list_tasks()
    assert MicroduckStiltRlCfg.experiment_name == "stilt_locomotion"
    assert MicroduckStiltRlCfg.save_interval == 100

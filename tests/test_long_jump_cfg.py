"""CPU invariants for the standing long jump task (no GPU needed)."""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import mjlab_microduck.tasks  # noqa: F401  (registry)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.onnx_policy_contract import bake_action_bounds, bake_action_clip
from mjlab_microduck.robot.long_jump_robot import (
    add_full_collision_geoms,
    add_leg_fold_pairs,
    get_long_jump_spec,
)
from mjlab_microduck.robot.microduck_constants import FULL_COLLISION, HOME_FRAME, MICRODUCK_ALLCOLLISIONS_XML
from mjlab_microduck.tasks.bounded_actions import BoundedJointPositionActionCfg

TASK = "Mjlab-LongJump-MicroDuck"


def _compiled_long_jump_model(with_floor: bool = True) -> mujoco.MjModel:
    spec = get_long_jump_spec()
    FULL_COLLISION.edit_spec(spec)
    if with_floor:
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    return spec.compile()


def _home_qpos(model: mujoco.MjModel, overrides: dict | None = None) -> np.ndarray:
    q = model.qpos0.copy()
    for j in range(model.njnt):
        name = model.joint(j).name
        adr = model.jnt_qposadr[j]
        for pat, val in HOME_FRAME.joint_pos.items():
            if re.match(pat, name):
                q[adr] = val
        if overrides and name in overrides:
            q[adr] = overrides[name]
    q[2] = 0.12
    return q


def _robot_self_contacts(model: mujoco.MjModel, qpos: np.ndarray) -> list[tuple[float, str, str]]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    rows = set()
    for c in data.contact[: data.ncon]:
        g1, g2 = model.geom(c.geom1).name, model.geom(c.geom2).name
        if "floor" in (g1, g2):
            continue
        rows.add((round(float(c.dist) * 1000, 2), g1, g2))
    return sorted(rows)


# ── robot model ───────────────────────────────────────────────────────────────
def test_every_visible_part_has_collision_coverage():
    """Every visual mesh (except bearings/head internals) has a same-body collision hull."""
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    added = add_full_collision_geoms(spec)
    assert len(added) >= 30
    for body in spec.bodies:
        if body.name in ("", "world", "jaw_soft"):
            continue
        visual = [(g.meshname, tuple(round(float(p), 5) for p in g.pos)) for g in body.geoms
                  if g.classname is not None and g.classname.name == "visual" and not g.meshname.startswith("seeed_bearing")]
        collision = [(g.meshname, tuple(round(float(p), 5) for p in g.pos)) for g in body.geoms
                     if g.name.endswith("_collision")]
        for key in visual:
            assert key in collision, f"{body.name}: visual mesh {key[0]} has no collision hull"


def test_collision_geoms_survive_collision_cfg_and_are_named():
    model = _compiled_long_jump_model(with_floor=False)
    active = [model.geom(g).name for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    assert len(active) >= 40
    assert all(name.endswith("_collision") for name in active), active
    assert "left_foot_collision" in active and "right_foot_collision" in active
    # trunk shells, thighs and neck are the parts that were missing before
    for part in ("trunk_base_left_shell", "trunk_base_right_shell", "upper_leg_left_upper_leg_left",
                 "upper_leg_right_upper_leg_right", "neck_neck"):
        assert any(part in name for name in active), part


def test_no_spurious_self_contact_at_home_and_fold_limits_are_physical():
    model = _compiled_long_jump_model()
    assert _robot_self_contacts(model, _home_qpos(model)) == []
    # a full fold must now collide (thigh plate / hip bracket vs shin parts)
    fold = {"left_hip_pitch": -1.57, "right_hip_pitch": 1.57, "left_knee": 1.57, "right_knee": -1.57,
            "left_ankle": 1.57, "right_ankle": -1.57}
    contacts = _robot_self_contacts(model, _home_qpos(model, fold))
    assert contacts, "a fully folded leg must produce thigh/shin contacts"
    assert model.npair > 0


def test_explicit_pairs_cover_thigh_shin():
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    add_full_collision_geoms(spec)
    n = add_leg_fold_pairs(spec)
    assert n > 0
    pairs = {(p.geomname1, p.geomname2) for p in spec.pairs}
    assert any(a.startswith("upper_leg_left") and b.startswith("leg_") for a, b in pairs)


# ── env cfg ───────────────────────────────────────────────────────────────────
def test_cfg_builds_with_bounded_actions_and_61d_layout():
    cfg = load_env_cfg(TASK, play=False)
    act = cfg.actions["joint_pos"]
    assert isinstance(act, BoundedJointPositionActionCfg)
    assert act.scale == 1.0 and act.use_default_offset
    actor = cfg.observations["actor"].terms
    order = list(actor.keys())
    assert order.index("head_command") < order.index("body_command")
    assert actor["head_command"].params["dim"] == 4
    assert actor["body_command"].params["dim"] == 6
    assert cfg.episode_length_s == 4.0
    assert cfg.scene.entities["robot"].spec_fn is get_long_jump_spec


def test_reward_signs_follow_penalty_convention():
    cfg = load_env_cfg(TASK, play=False)
    r = cfg.rewards
    # self-negating terms need POSITIVE weights
    for name in ("long_jump_stand_tax", "long_jump_body_contact", "long_jump_lateral", "command_saturation", "gentle_landing"):
        assert r[name].weight >= 0.0, name
    # mjlab-base costs need NEGATIVE weights
    for name in ("action_rate_l2", "self_collisions", "body_ang_vel", "angular_momentum"):
        assert r[name].weight <= 0.0, name
    assert r["long_jump_progress"].weight > 0 and r["long_jump_landing"].weight > 0 and r["long_jump_stand"].weight > 0
    assert 0.0 < r["long_jump_distance"].weight <= 1.0 and r["long_jump_stepped"].weight <= 0.0
    for name in ("track_linear_velocity", "pose", "air_time"):
        assert name not in r
    assert r["upright"].weight > 0 and r["fall_cost"].weight > 0


def test_play_cfg_spawns_standing_only():
    cfg = load_env_cfg(TASK, play=True)
    p = cfg.events["set_long_jump_state"].params
    assert p["standing_prob"] == 1.0 and p["flight_prob"] == 0.0 and p["crouch_prob"] == 0.0
    assert "fell" in cfg.terminations and "nan_state" in cfg.terminations
    assert "push_robot" not in cfg.events


def test_sensors_cover_feet_body_and_head():
    cfg = load_env_cfg(TASK, play=False)
    names = {s.name for s in cfg.scene.sensors}
    assert {"feet_ground_contact", "body_ground_contact", "head_ground_contact", "self_collision"} <= names
    body = next(s for s in cfg.scene.sensors if s.name == "body_ground_contact")
    pat = re.compile(body.primary.pattern)
    assert pat.match("upper_leg_left_upper_leg_left_0_collision")
    assert not pat.match("left_foot_collision") and not pat.match("right_foot_collision")
    assert not pat.match("ankle_left_foot_left_0_collision") and not pat.match("ankle_right_ankle_right_0_collision")


# ── ONNX bounds ───────────────────────────────────────────────────────────────
def _write_linear_policy(path: Path, n: int = 3) -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "test-policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n])],
        [numpy_helper.from_array(np.eye(n, dtype=np.float32), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    onnx.save(model, path)


def test_action_bounds_are_baked_per_output(tmp_path: Path):
    path = tmp_path / "policy.onnx"
    _write_linear_policy(path)
    bake_action_clip(path, 10.0)  # the usual RSL-RL clamp first, then the joint bounds
    bake_action_bounds(path, lo=[-0.5, -1.0, -2.0], hi=[0.5, 1.0, 2.0])
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": np.array([[3.0, -3.0, 0.25]], dtype=np.float32)})[0]
    np.testing.assert_allclose(out, [[0.5, -1.0, 0.25]])
    meta = {m.key: m.value for m in onnx.load(path).metadata_props}
    assert meta["action_bounds_baked_into_onnx"] == "true"
    with pytest.raises(ValueError):
        bake_action_bounds(path, lo=[0.0] * 3, hi=[1.0] * 3)

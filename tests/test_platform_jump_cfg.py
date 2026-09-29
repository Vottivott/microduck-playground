"""CPU invariants for the platform jump task (no GPU needed)."""
from __future__ import annotations

import math

import mujoco
import torch

import mjlab_microduck.tasks  # noqa: F401  (registry)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot import platform_stage as ps
from mjlab_microduck.tasks import platform_jump_mdp as pj

TASK = "Mjlab-PlatformJump-MicroDuck"


def test_stage_geometry():
    for drop in (0.05, 0.3):
        a = ps.platform_a_center(drop)
        assert math.isclose(a[2] + 0.5 * ps.BLOCK_HEIGHT_M, ps.B_TOP_M + drop)      # A's top
        assert math.isclose(a[0] + 0.5 * ps.A_DEPTH_M, 0.0)                         # A's front edge at x = 0
    for gap in (0.0, 0.25):
        b = ps.platform_b_center(gap)
        assert math.isclose(b[0] - 0.5 * ps.B_DEPTH_M, gap)                         # B's near edge at x = gap
        assert math.isclose(b[2] + 0.5 * ps.BLOCK_HEIGHT_M, ps.B_TOP_M)
    for spec_fn in (ps.platform_a_spec, ps.platform_b_spec):
        model = spec_fn().compile()
        assert model.nmocap == 1 and model.ngeom == 1
        assert model.geom_contype[0] != 0 and model.geom_conaffinity[0] != 0


def test_cfg_terms_and_weights():
    cfg = load_env_cfg(TASK)
    names = {s.name for s in cfg.scene.sensors}
    assert {pj.FEET_A, pj.FEET_B, pj.FEET_FLOOR, pj.BODY_A, pj.BODY_B, pj.BODY_FLOOR, *pj.HEAD_ANY} <= names
    assert "feet_ground_contact" in names   # the velocity base's critic foot_air_time term reads it
    assert {ps.PLATFORM_A, ps.PLATFORM_B, "robot"} <= set(cfg.scene.entities)
    r = cfg.rewards
    assert not any(n.startswith("long_jump") for n in r)
    # self-negating penalties carry POSITIVE weights; cost functions negative
    for n in ("pj_fall_cost", "pj_stand_tax", "pj_body_contact", "pj_lateral", "pj_dawdle"):
        assert r[n].weight > 0
    for n in ("pj_progress", "pj_landing", "pj_stand", "pj_launch", "pj_takeoff"):
        assert r[n].weight > 0
    assert r["pj_stepped"].weight < 0 and r["pj_pit"].weight < 0
    assert r["pj_stand"].params["target_height"] > ps.B_TOP_M
    assert cfg.observations["actor"].terms["body_command"].func is pj.pj_command_obs
    assert cfg.terminations["fell"].func is pj.pj_fell
    ev = cfg.events["set_platform_jump_state"].params
    assert ev["gap_range"][0] >= 0.0 and ev["drop_range"][0] > 0.0
    stages = cfg.curriculum["platform_jump_stages"].params["param_stages"]
    assert stages[0]["step"] == 0 and stages[-1]["params"]["gap_range"][1] >= 0.3
    assert all(stages[i]["step"] <= stages[i + 1]["step"] for i in range(len(stages) - 1))
    play = load_env_cfg(TASK, play=True)
    assert play.events["set_platform_jump_state"].params["flight_prob"] == 0.0


def test_sync_factor_grading():
    g = torch.tensor([0, 1, 2, 3, 5])
    f = pj._sync_factor(g, pj.TAKEOFF_SYNC_FREE_STEPS, pj.TAKEOFF_SYNC_ZERO_STEPS)
    assert torch.allclose(f, torch.tensor([1.0, 1.0, 0.75, 0.5, 0.0]))


def test_command_obs_scaling():
    class _Env:
        num_envs = 3
        device = "cpu"

    env = _Env()
    st = pj._st(env)
    st["gap"][:] = torch.tensor([0.0, 0.15, 0.3])
    st["drop"][:] = torch.tensor([0.3, 0.0, 0.15])
    obs = pj.pj_command_obs(env, dim=6)
    assert obs.shape == (3, 6)
    assert torch.allclose(obs[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(obs[:, 1], torch.tensor([1.0, 0.0, 0.5]))
    assert torch.all(obs[:, 2:] == 0)


def test_gap_drop_override(monkeypatch):
    """MICRODUCK_PJ_GAP / _DROP pin the stage for renders and probes."""
    import importlib

    import mjlab_microduck.tasks.microduck_platform_jump_env_cfg as m

    monkeypatch.setenv("MICRODUCK_PJ_GAP", "0.22")
    monkeypatch.setenv("MICRODUCK_PJ_DROP", "0.10,0.14")
    m = importlib.reload(m)
    cfg = m.make_microduck_platform_jump_env_cfg(play=True)
    assert cfg.events["set_platform_jump_state"].params["gap_range"] == (0.22, 0.22)
    assert cfg.events["set_platform_jump_state"].params["drop_range"] == (0.10, 0.14)
    # the curriculum must not widen it back
    for stage in cfg.curriculum["platform_jump_stages"].params["param_stages"]:
        assert stage["params"]["gap_range"] == (0.22, 0.22)
        assert stage["params"]["drop_range"] == (0.10, 0.14)
    monkeypatch.delenv("MICRODUCK_PJ_GAP")
    monkeypatch.delenv("MICRODUCK_PJ_DROP")
    importlib.reload(m)


def test_progress_pay_is_capped_at_the_gap():
    """An uncapped distance reward makes the policy overshoot; the pay stops at gap + margin."""
    from types import SimpleNamespace

    class _Env:
        num_envs = 2
        device = "cpu"
        scene = {"robot": SimpleNamespace()}

    env = _Env()
    st = pj._st(env)
    st["gap"][:] = torch.tensor([0.10, 0.30])
    st["max_prog"][:] = torch.tensor([0.60, 0.60])
    st["phase"][:] = 1
    st["sync"][:] = 1.0
    orig = pj._update
    pj._update = lambda e, a: st
    try:
        first = pj.pj_progress(env, cap_margin=0.15, asset_cfg=SimpleNamespace(name="robot"))
        again = pj.pj_progress(env, cap_margin=0.15, asset_cfg=SimpleNamespace(name="robot"))
    finally:
        pj._update = orig
    # 0.10 + 0.15 and 0.30 + 0.15, not the 0.60 actually flown
    assert torch.allclose(first, torch.tensor([0.25, 0.45]))
    # potential-based: the same displacement pays only once
    assert torch.allclose(again, torch.zeros(2))

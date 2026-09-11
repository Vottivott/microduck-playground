"""Cfg invariants for the basketball circus-balance env (CPU only)."""
from __future__ import annotations

import mujoco
import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_basketball_env_cfg import (
    BALL_RADIUS,
    HOLD_LEVELS,
    _ball_spec,
    make_microduck_basketball_env_cfg,
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_basketball_env_cfg(blind=False)


def test_ball_spec_compiles():
    model = _ball_spec().compile()
    assert model.nq == 7 and model.nv == 6
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_sphere")
    assert abs(model.geom_size[gid][0] - BALL_RADIUS) < 1e-9
    assert model.geom_contype[gid] == 1 and model.geom_conaffinity[gid] == 1
    assert 0.55 < model.body_mass[1] < 0.7


def test_scene_and_obs_contract(cfg):
    assert set(cfg.scene.entities) == {"robot", "ball"}
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["body_command"].func is microduck_mdp.basketball_state
        assert terms["head_command"].func is microduck_mdp.basketball_head_pad
        assert "command" in terms
    assert "head_pose" not in cfg.commands and "body_pose" not in cfg.commands
    assert "twist" in cfg.commands


def test_reward_signs(cfg):
    r = cfg.rewards
    for name in ("track_linear_velocity", "track_angular_velocity", "upright", "centered", "feet_on_ball", "height"):
        assert r[name].weight > 0, name
    for name in ("body_ang_vel", "angular_momentum", "action_rate_l2", "ball_speed"):
        assert r[name].weight < 0, name
    for name in ("air_time", "foot_clearance", "foot_swing_height", "foot_slip", "head_pose_tracking", "body_pose_tracking"):
        assert name not in r, name
    assert r["feet_on_ball"].params["asset_cfg"].site_names == ["left_foot", "right_foot"]


def test_terminations_events_curriculum(cfg):
    assert "fell_over" not in cfg.terminations
    assert cfg.terminations["fell"].func is microduck_mdp.basketball_fell
    assert "nan_state" in cfg.terminations
    assert list(cfg.events)[0] == "reset_basketball"
    assert "reset_base" not in cfg.events
    assert cfg.actions["ball_hold"].levels == HOLD_LEVELS
    assert cfg.curriculum["basketball_hold"].params["levels"] == HOLD_LEVELS
    for name, term in cfg.curriculum.items():
        params = getattr(term, "params", {}) or {}
        if "command_name" in params:
            assert params["command_name"] in cfg.commands, name
        if "reward_name" in params:
            assert params["reward_name"] in cfg.rewards, name


def test_play_cfg_free_ball():
    play = make_microduck_basketball_env_cfg(play=True)
    assert play.actions["ball_hold"].levels == (0.0,)
    assert "basketball_hold" not in play.curriculum


def test_fixed_smoothness_overrides_inherited_curriculum(monkeypatch):
    from mjlab_microduck.tasks import microduck_basketball_env_cfg as basketball
    monkeypatch.setattr(basketball, "ACTION_RATE_WEIGHT", "-0.2")
    fixed = basketball.make_microduck_basketball_env_cfg(blind=True, history=1)
    assert fixed.rewards["action_rate_l2"].weight == -0.2
    assert "action_rate_weight" not in fixed.curriculum
    assert fixed.observations["actor"].terms["body_command"].func is microduck_mdp.basketball_body_pad
    monkeypatch.setattr(basketball, "ACTION_RATE_WEIGHT", "0.2")
    with pytest.raises(ValueError, match="non-positive"):
        basketball.make_microduck_basketball_env_cfg()


def test_released_default_actor_is_blind():
    cfg = make_microduck_basketball_env_cfg()
    assert cfg.observations["actor"].terms["body_command"].func is microduck_mdp.basketball_body_pad
    assert cfg.observations["critic"].terms["body_command"].func is microduck_mdp.basketball_state
    assert cfg.observations["actor"].terms["head_command"].func is microduck_mdp.basketball_head_pad


def test_basketball_reward_guard_matches_training(monkeypatch):
    from types import SimpleNamespace
    import torch

    manager = SimpleNamespace(_episode_sums={"feet_on_ball": torch.tensor([float("inf")])})
    monkeypatch.setattr(microduck_mdp, "_orig_reward_compute", lambda manager, dt:
                        torch.tensor([float("nan"), float("inf"), 20., -20., .5]))
    result = microduck_mdp._nan_safe_reward_compute(manager, .02)
    torch.testing.assert_close(result, torch.tensor([0., 0., 10., -10., .5]))
    torch.testing.assert_close(manager._episode_sums["feet_on_ball"], torch.tensor([0.]))


def test_default_basketball_actor_is_lstm():
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    assert MicroduckBasketballRlCfg.actor.class_name == "RNNModel"
    assert MicroduckBasketballRlCfg.actor.rnn_type == "lstm"
    assert MicroduckBasketballRlCfg.actor.rnn_hidden_dim == 256
    assert MicroduckBasketballRlCfg.actor.rnn_num_layers == 1

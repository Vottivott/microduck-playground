"""CPU regressions for the privileged-teacher / blind-student boundary."""
from copy import deepcopy

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microduck.basketball_distillation import (
    actor_model_cfg, load_actor, make_distillation_env_cfg, make_runner_cfg,
)
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg


def observations():
    torch.manual_seed(42)
    actor = torch.randn(4, 61)
    actor[:, -6:] = 0
    return TensorDict({"actor": actor, "teacher": torch.randn(4, 61)}, batch_size=[4])


def models(obs):
    groups = {"student": ["actor"], "teacher": ["teacher"]}
    kwargs = dict(hidden_dims=(16, 8), activation="elu", obs_normalization=True,
                  distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "scalar"})
    student = RNNModel(obs, groups, "student", 14, rnn_type="lstm", rnn_hidden_dim=8, **deepcopy(kwargs))
    teacher = MLPModel(obs, groups, "teacher", 14, **deepcopy(kwargs))
    return student, teacher


def test_teacher_layout_and_student_boundary():
    cfg = make_distillation_env_cfg()
    reference = make_microduck_basketball_env_cfg(blind=False, history=1)
    a, t = cfg.observations["actor"], cfg.observations["teacher"]
    assert list(a.terms) == list(t.terms) == list(reference.observations["actor"].terms)
    assert a.terms["body_command"].func is mdp.basketball_body_pad
    assert t.terms["body_command"].func is mdp.basketball_state
    for name, term in t.terms.items():
        ref = reference.observations["actor"].terms[name]
        assert term.func is ref.func
        assert term.history_length == ref.history_length
        assert term.delay_min_lag == ref.delay_min_lag
        assert term.delay_max_lag == ref.delay_max_lag
    assert cfg.actions["ball_hold"].levels == (0.0,)
    assert "basketball_hold" not in cfg.curriculum
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in {v.func.__name__ for v in cfg.events.values()}


def test_teacher_inputs_cannot_change_student_actions():
    obs = observations()
    student, teacher = models(obs)
    student.eval()
    teacher.eval()
    changed = obs.clone()
    changed["teacher"][:, -6:] += 100
    with torch.no_grad():
        expected = student(obs)
        student.reset()
        torch.testing.assert_close(student(changed), expected, rtol=0, atol=0)
        assert not torch.allclose(teacher(obs), teacher(changed))


def test_student_checkpoint_restores_normalizer_and_recurrence():
    obs = observations()
    student, _ = models(obs)
    student.update_normalization(obs)
    checkpoint = {"student_state_dict": student.state_dict()}
    restored = load_actor(checkpoint, obs)
    student.eval()
    with torch.no_grad():
        for _ in range(3):
            torch.testing.assert_close(restored(obs), student(obs))
        dones = torch.tensor([True, False, True, False])
        student.reset(dones)
        restored.reset(dones)
        torch.testing.assert_close(restored(obs), student(obs))
    assert actor_model_cfg(checkpoint)["rnn_hidden_dim"] == 8


def test_upstream_distillation_updates_student_and_freezes_teacher():
    obs = observations()
    student, teacher = models(obs)
    teacher.requires_grad_(False)
    storage = RolloutStorage("distillation", 4, 4, obs, [14])
    alg = Distillation(student, teacher, storage, gradient_length=4, learning_rate=1e-3)
    alg.train_mode()
    before_teacher = {k: v.clone() for k, v in teacher.state_dict().items()}
    before_student = student.mlp[0].weight.detach().clone()
    with torch.inference_mode():
        for i in range(4):
            actions = alg.act(obs)
            assert actions.shape == (4, 14)
            alg.process_env_step(obs, torch.zeros(4), torch.tensor([i == 1, False, False, False]), {})
    losses = alg.update()
    assert torch.isfinite(torch.tensor(losses["behavior"]))
    assert not torch.equal(student.mlp[0].weight, before_student)
    assert all(torch.equal(v, before_teacher[k]) for k, v in teacher.state_dict().items())


def test_runner_config_uses_separate_groups_and_complete_gradient_windows():
    obs = observations()
    student, teacher = models(obs)
    cfg = make_runner_cfg({"actor_state_dict": teacher.state_dict()},
                          {"actor_state_dict": student.state_dict()}, run_name="test")
    assert cfg["obs_groups"] == {"student": ["actor"], "teacher": ["teacher"]}
    assert cfg["num_steps_per_env"] % cfg["algorithm"]["gradient_length"] == 0
    assert cfg["student"]["class_name"] == "RNNModel"
    assert cfg["teacher"]["class_name"] == "MLPModel"

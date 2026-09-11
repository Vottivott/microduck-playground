"""Ball-state teacher / proprioceptive LSTM student wiring for basketball.

Uses rsl_rl's student-rollout Distillation, with mjlab checkpoint persistence
and normalized ONNX export. No teacher action is applied to the environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.runners import DistillationRunner


def make_distillation_env_cfg(*, play: bool = False, command_scale: float = 1.0):
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import make_microduck_basketball_env_cfg

    if command_scale <= 0:
        raise ValueError("command_scale must be positive")
    cfg = make_microduck_basketball_env_cfg(play=play, blind=True, history=1)
    # Match the teacher's ACTOR observations, not the larger privileged critic.
    teacher = deepcopy(cfg.observations["actor"])
    teacher.terms["body_command"].func = mdp.basketball_state
    teacher.terms["body_command"].params = {"pos_scale": 3.0, "vel_scale": 1.0}
    cfg.observations["teacher"] = teacher
    cfg.actions["ball_hold"].levels = (0.0,)
    cfg.curriculum.pop("basketball_hold", None)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.15 * command_scale, 0.15 * command_scale)
    twist.ranges.lin_vel_y = (-0.10 * command_scale, 0.10 * command_scale)
    twist.ranges.ang_vel_z = (-0.50 * command_scale, 0.50 * command_scale)
    return cfg


def actor_state(checkpoint: dict) -> dict:
    return checkpoint["student_state_dict"] if "student_state_dict" in checkpoint else checkpoint["actor_state_dict"]


def actor_model_cfg(checkpoint: dict) -> dict:
    """Reconstruct the supported basketball MLP/LSTM architecture from weights.

    The basketball family uses ELU, normalized observations and a scalar-space
    Gaussian. Strict state loading rejects incompatible checkpoint formats.
    """
    state = actor_state(checkpoint)
    layers = sorted(
        (int(k.split(".")[1]), v) for k, v in state.items()
        if k.startswith("mlp.") and k.endswith(".weight")
    )
    if not layers or layers[-1][1].shape[0] != 14:
        raise ValueError("Expected a 14-action basketball actor")
    cfg = dict(
        class_name="MLPModel", hidden_dims=[v.shape[0] for _, v in layers[:-1]],
        activation="elu", obs_normalization=True,
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=1.0, std_type="scalar"),
    )
    if "rnn.rnn.weight_ih_l0" in state:
        hidden = state["rnn.rnn.weight_hh_l0"].shape[1]
        if state["rnn.rnn.weight_hh_l0"].shape[0] != 4 * hidden:
            raise ValueError("Only LSTM basketball checkpoints are supported")
        num_layers = sum(k.startswith("rnn.rnn.weight_ih_l") for k in state)
        cfg.update(class_name="RNNModel", rnn_type="lstm", rnn_hidden_dim=hidden, rnn_num_layers=num_layers)
    return cfg


def load_actor(checkpoint: dict, observations, device: str = "cpu"):
    cfg = actor_model_cfg(checkpoint)
    cls = RNNModel if cfg.pop("class_name") == "RNNModel" else MLPModel
    model = cls(observations, {"actor": ["actor"]}, "actor", 14, **cfg).to(device)
    model.load_state_dict(actor_state(checkpoint), strict=True)
    model.eval()
    return model


def make_runner_cfg(teacher_checkpoint: dict, student_checkpoint: dict, *,
                    run_name: str, learning_rate: float = 1e-4, save_interval: int = 250) -> dict:
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg

    cfg = asdict(MicroduckBasketballRlCfg)
    cfg.pop("actor")
    cfg.pop("critic")
    cfg.update(
        class_name="BasketballDistillationRunner", experiment_name="basketball_distillation",
        run_name=run_name, obs_groups={"student": ["actor"], "teacher": ["teacher"]},
        student=actor_model_cfg(student_checkpoint), teacher=actor_model_cfg(teacher_checkpoint),
        upload_model=False, save_interval=save_interval,
        algorithm=dict(class_name="Distillation", num_learning_epochs=1,
                       gradient_length=24, learning_rate=learning_rate,
                       max_grad_norm=1.0, loss_type="mse", optimizer="adam"),
    )
    if cfg["student"]["class_name"] != "RNNModel":
        raise ValueError("Initialize the student from a recurrent basketball actor")
    return cfg


class BasketballDistillationRunner(DistillationRunner, MjlabOnPolicyRunner):
    """Upstream distillation plus mjlab's env-step persistence and ONNX export."""

    provenance: dict | None = None

    def initialize(self, teacher_checkpoint: dict, student_checkpoint: dict, action_std: float = 0.05):
        if action_std <= 0:
            raise ValueError("Student rollout action_std must be positive")
        self.alg.teacher.load_state_dict(actor_state(teacher_checkpoint), strict=True)
        self.alg.teacher.requires_grad_(False)
        self.alg.teacher.eval()
        self.alg.teacher_loaded = True
        self.alg.student.load_state_dict(actor_state(student_checkpoint), strict=True)
        if self.alg.student.obs_dim != 61 or self.alg.teacher.obs_dim != 61:
            raise ValueError("Teacher and student must each receive 61 observations")
        # Distillation's rollout uses stochastic_output=True. Reduce exploration
        # from PPO's inherited std while preserving the learned deterministic actor.
        with torch.no_grad():
            self.alg.student.distribution.std_param.fill_(action_std)
        self.alg.student.distribution.std_param.requires_grad_(False)
        # Preserve mature DR/standing-command schedules, but never re-enable hold.
        source_infos = student_checkpoint.get("infos") or {}
        self.env.unwrapped.common_step_counter = source_infos.get("env_state", {}).get(
            "common_step_counter", (student_checkpoint.get("iter", 0) + 1) * 24,
        )

    def save(self, path: str, infos=None):
        super().save(path, infos={**(self.provenance or {}), **(infos or {})})


def checkpoint_sha256(path: str | Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

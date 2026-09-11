"""PPO mode for correcting swing planarity without relearning the pump."""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from rsl_rl.algorithms.ppo import PPO


PLANAR_ACTION_INDICES = (0, 1, 7, 8, 9, 10)
"""Hip yaw/roll and head yaw/roll outputs in the 14-action contract."""


def _configured_action_indices() -> tuple[int, ...]:
    raw = os.environ.get("MICRODUCK_SWING_PLANAR_ACTION_INDICES")
    if raw is None:
        return PLANAR_ACTION_INDICES
    try:
        indices = tuple(sorted({int(part.strip()) for part in raw.split(",")}))
    except ValueError as exc:
        raise ValueError(
            "MICRODUCK_SWING_PLANAR_ACTION_INDICES must be comma-separated integers"
        ) from exc
    if not indices or any(index < 0 or index >= 14 for index in indices):
        raise ValueError(
            "MICRODUCK_SWING_PLANAR_ACTION_INDICES must select actions in [0, 13]"
        )
    return indices


def swing_tail_environment_weights(
    validity: torch.Tensor,
    fraction: float,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select and weight the worst exact-mechanism environments in a rollout."""
    if validity.ndim != 3 or validity.shape[-1] != 6:
        raise ValueError("validity must have shape [time, environments, 6]")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail fraction must be in (0, 1]")
    if weight < 1.0:
        raise ValueError("tail weight must be at least one")

    lateral = validity[..., 0].abs().amax(dim=0)
    lateral_velocity = validity[..., 1].abs().amax(dim=0)
    roll_yaw_kinetics = validity[..., 2:4].abs().amax(dim=(0, 2))
    alignment = validity[..., 4].clamp_min(0.0).amax(dim=0)
    cord_or_latch = validity[..., 5].clamp_min(0.0).amax(dim=0)
    severity = torch.maximum(
        torch.maximum(lateral, alignment),
        torch.maximum(
            0.5 * lateral_velocity,
            torch.maximum(roll_yaw_kinetics, cord_or_latch),
        ),
    )
    count = max(1, math.ceil(validity.shape[1] * fraction))
    tail_indices = torch.topk(severity, count, sorted=False).indices
    env_weights = torch.ones_like(severity)
    env_weights[tail_indices] = weight
    return env_weights, severity, tail_indices


class SwingPlanarCorrectionPPO(PPO):
    """Optionally train only the final actor rows that control yaw and roll.

    Set ``MICRODUCK_SWING_PLANAR_CORRECTION=1`` for this mode.  The critic
    remains fully trainable.  All shared actor features, observation
    normalization, sagittal output rows, and action standard deviations are
    frozen, preserving the learned pitch/knee/ankle pumping trajectory while
    allowing deployable IMU-heading feedback to adjust the six outputs that
    can actively reject lateral/yaw/roll drift.

    With the environment variable unset this class is behaviorally identical
    to upstream PPO, so ordinary evaluation and future full-policy training
    retain the standard path.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.planar_correction_enabled = (
            os.environ.get("MICRODUCK_SWING_PLANAR_CORRECTION") == "1"
        )
        self.train_last_hidden = (
            os.environ.get("MICRODUCK_SWING_TRAIN_LAST_HIDDEN") == "1"
        )
        self.planar_action_indices = _configured_action_indices()
        self.tail_fraction = float(
            os.environ.get("MICRODUCK_SWING_TAIL_FRACTION", "0")
        )
        self.tail_weight = float(
            os.environ.get("MICRODUCK_SWING_TAIL_WEIGHT", "1")
        )
        if not 0.0 <= self.tail_fraction <= 1.0:
            raise ValueError("MICRODUCK_SWING_TAIL_FRACTION must be in [0, 1]")
        if self.tail_weight < 1.0:
            raise ValueError("MICRODUCK_SWING_TAIL_WEIGHT must be at least one")
        self._last_tail_metrics: dict[str, float] = {}
        self._planar_gradient_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        if not self.planar_correction_enabled:
            return

        linear_layers = [module for module in self.actor.mlp if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("swing actor has no linear output layer")
        output_layer = linear_layers[-1]
        if output_layer.out_features != 14:
            raise RuntimeError(
                f"expected 14 swing actions, got {output_layer.out_features}"
            )

        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        last_hidden_parameters = 0
        if self.train_last_hidden:
            if len(linear_layers) < 2:
                raise RuntimeError("swing actor has no final hidden linear layer")
            last_hidden_layer = linear_layers[-2]
            last_hidden_layer.weight.requires_grad_(True)
            last_hidden_layer.bias.requires_grad_(True)
            last_hidden_parameters = sum(
                parameter.numel() for parameter in last_hidden_layer.parameters()
            )
        output_layer.weight.requires_grad_(True)
        output_layer.bias.requires_grad_(True)

        row_mask = torch.zeros(
            output_layer.out_features,
            dtype=output_layer.weight.dtype,
            device=output_layer.weight.device,
        )
        row_mask[list(self.planar_action_indices)] = 1.0
        self._planar_gradient_hook_handles.extend(
            (
                output_layer.weight.register_hook(
                    lambda gradient: gradient * row_mask[:, None]
                ),
                output_layer.bias.register_hook(lambda gradient: gradient * row_mask),
            )
        )
        effective_trainable_actor_parameters = (
            len(self.planar_action_indices) * (output_layer.in_features + 1)
            + last_hidden_parameters
        )
        print(
            "SWING_PLANAR_CORRECTION_ACTIVE "
            f"action_indices={self.planar_action_indices} "
            f"effective_trainable_actor_parameters="
            f"{effective_trainable_actor_parameters} "
            "actor_normalizer=frozen actor_std=frozen "
            f"train_last_hidden={self.train_last_hidden} "
            f"tail_fraction={self.tail_fraction} tail_weight={self.tail_weight}"
        )

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if not self.planar_correction_enabled:
            super().process_env_step(obs, rewards, dones, extras)
            return
        # MLPModel.update_normalization checks this flag, while forward() still
        # applies the already-learned normalizer module unconditionally.
        actor_normalization_enabled = self.actor.obs_normalization
        self.actor.obs_normalization = False
        try:
            super().process_env_step(obs, rewards, dones, extras)
        finally:
            self.actor.obs_normalization = actor_normalization_enabled

    def compute_returns(self, obs) -> None:
        super().compute_returns(obs)
        if (
            not self.planar_correction_enabled
            or self.tail_fraction <= 0.0
            or self.tail_weight == 1.0
        ):
            return

        # The final ten privileged critic slots are body_command(6), replaced
        # by swing_validity_observation, followed by swing_state(4).  Rank
        # complete environments by the worst exact normalized mechanism state
        # seen in this rollout fragment.  This changes only optimization
        # weights; none of these values enter the deployment actor.
        critic_obs = self.storage.observations["critic"]
        validity = critic_obs[..., -10:-4]
        env_weights, severity, tail_indices = swing_tail_environment_weights(
            validity,
            self.tail_fraction,
            self.tail_weight,
        )
        advantages = self.storage.advantages
        advantages.mul_(env_weights[None, :, None])
        advantages.sub_(advantages.mean()).div_(advantages.std() + 1e-8)
        self._last_tail_metrics = {
            "tail/severity_mean": float(severity.mean()),
            "tail/severity_cutoff": float(severity[tail_indices].min()),
            "tail/selected_fraction": tail_indices.numel() / self.storage.num_envs,
            "tail/weight": self.tail_weight,
        }

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses.update(self._last_tail_metrics)
        return losses

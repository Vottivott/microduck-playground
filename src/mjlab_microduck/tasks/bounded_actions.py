"""Joint-position action term with the servo command bounded to the joint travel.

Lesson from the running policy (hardware test, 2026-09): the policy commanded
the hip-yaw servos beyond their mechanical travel, so the servos pushed
against the stops instead of reaching the target (wasted current, heating,
poor tracking).  The MJCF ``ctrlrange`` is deliberately wide (low-kp servos
need overshoot headroom), so nothing in the stock pipeline bounds the
command.

``BoundedJointPositionAction`` clamps the commanded target
``raw · scale + offset`` to the joint's hard position range (optionally
inset by ``bound_margin``) and writes the bounded command back into the raw
action and the action manager's history buffers, so

* the ``actions`` observation (last action) and the action-rate penalties
  see the bounded command, exactly as the runtime's action history will;
* :func:`raw_action_bounds` gives the per-output bounds in policy-output
  units, which ``scripts/export.py`` bakes into the ONNX graph
  (``bake_action_bounds``) so deployment applies the identical clamp;
* ``last_saturation`` (per env, radians summed over joints) exposes how much
  the policy asked beyond the bounds, for a small training penalty and the
  pre-hardware command audit (``scripts/audit_joint_commands.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class BoundedJointPositionAction(JointPositionAction):
    """Position targets clamped to the joint travel, bounded values fed back."""

    cfg: "BoundedJointPositionActionCfg"

    def __init__(self, cfg: "BoundedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]  # (B, n, 2)
        margin = float(cfg.bound_margin)
        self._cmd_lo = limits[..., 0] + margin
        self._cmd_hi = limits[..., 1] - margin
        self._history_start: int | None = None
        self.last_saturation = torch.zeros(self.num_envs, device=self.device)

    # -- bounds -------------------------------------------------------------
    @property
    def command_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env (lo, hi) bounds of the commanded joint target, radians."""
        return self._cmd_lo, self._cmd_hi

    def raw_action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(lo, hi) of the policy output (env 0) that map onto the command bounds."""
        scale = self._scale if torch.is_tensor(self._scale) else torch.full_like(self._cmd_lo, float(self._scale))
        offset = self._offset if torch.is_tensor(self._offset) else torch.full_like(self._cmd_lo, float(self._offset))
        lo = (self._cmd_lo - offset) / scale
        hi = (self._cmd_hi - offset) / scale
        return lo[0].clone(), hi[0].clone()

    # -- processing ---------------------------------------------------------
    def _slice_in_manager(self) -> slice:
        if self._history_start is None:
            start = 0
            for term in self._env.action_manager._terms.values():
                if term is self:
                    break
                start += term.action_dim
            self._history_start = start
        return slice(self._history_start, self._history_start + self.action_dim)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        unbounded = self._processed_actions
        bounded = torch.clamp(unbounded, self._cmd_lo, self._cmd_hi)
        self.last_saturation = (unbounded - bounded).abs().sum(dim=-1)
        self._processed_actions = bounded
        raw = (bounded - self._offset) / self._scale
        self._raw_actions[:] = raw
        # The manager copied the unbounded policy output into its history
        # buffer before dispatching; overwrite our slice so observations and
        # action-rate terms see the command that was actually sent.
        self._env.action_manager._action[:, self._slice_in_manager()] = raw


@dataclass(kw_only=True)
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    """Config for :class:`BoundedJointPositionAction`."""

    bound_margin: float = 0.0
    """Inset (rad) from the hard joint range applied to the command bounds."""

    def build(self, env) -> BoundedJointPositionAction:  # type: ignore[override]
        return BoundedJointPositionAction(self, env)


def bounded_cfg_from(cfg: JointPositionActionCfg, bound_margin: float = 0.0) -> BoundedJointPositionActionCfg:
    """Copy a stock joint-position action cfg into the bounded variant."""
    fields = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    fields.pop("transmission_type", None)
    return BoundedJointPositionActionCfg(bound_margin=bound_margin, **fields)


def command_saturation_penalty(env, action_name: str = "joint_pos") -> torch.Tensor:
    """SELF-NEGATING: -(radians the policy asked beyond the command bounds)."""
    term = env.action_manager.get_term(action_name)
    sat = getattr(term, "last_saturation", None)
    if sat is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -sat

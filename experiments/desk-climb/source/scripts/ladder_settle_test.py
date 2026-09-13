"""Settle test for the stair-ladder spawn states (AGENTS.md rule: verify before training).

For each curriculum level, spawn many environments (floor and on-ladder
spawns), hold the spawn joint targets, and report trunk tilt/drop at 0.5 s
and 1 s, which feet are supported by treads, and whether any other body part
touches the ladder.  Under the BAM actuator even the walker's HOME stance on
the floor sags forward within about a second (the policy normally corrects
it), so the verdict is relative: an on-ladder spawn passes when its 0.5 s
tilt and drop are no worse than the floor spawn's plus a small margin.

Requires a GPU (MuJoCo Warp).  Example::

    MUJOCO_GL=egl uv run python scripts/ladder_settle_test.py --num-envs 256 --output settle.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401 - populate registry
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ladder_env_cfg import LADDER_LEVELS


@dataclass(frozen=True)
class Config:
    task_id: str = "Mjlab-LadderClimb-MicroDuck"
    num_envs: int = 256
    hold_s: float = 1.0
    check_s: float = 0.5
    levels: tuple[int, ...] = tuple(range(len(LADDER_LEVELS)))
    floor_spawn_prob: float = 0.5
    fall_tilt_deg: float = 30.0
    seed: int = 7
    output: str | None = None
    nominal_physics: bool = False


def _run_level(cfg: Config, level: int, device: str) -> dict:
    env_cfg = load_env_cfg(cfg.task_id, play=False)
    env_cfg.seed = cfg.seed + level
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.hold_s + 2.0
    env_cfg.terminations.clear()
    env_cfg.curriculum.clear()
    env_cfg.events.pop("push_robot", None)
    if cfg.nominal_physics:
        for name in (
            "foot_friction", "encoder_bias", "base_com", "randomize_com",
            "randomize_head_com", "randomize_mass_inertia",
            "randomize_joint_friction", "randomize_armature",
        ):
            env_cfg.events.pop(name, None)
    spawn = env_cfg.events["reset_stair_ladder"].params
    spawn["fixed_level"] = level
    spawn["floor_spawn_prob"] = cfg.floor_spawn_prob
    spawn["swing_spawn_prob"] = 0.0  # gate/evaluate static stance spawns only
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    robot = env.scene["robot"]
    state = env._stair
    on_floor = state.spawn_on_floor.clone()
    servo_ids = microduck_mdp._servo_joint_ids(env, robot)
    hold_action = (robot.data.joint_pos[:, servo_ids] - robot.data.default_joint_pos[:, servo_ids]).clone()
    spawn_z = robot.data.root_link_pos_w[:, 2].clone()

    steps = round(cfg.hold_s / env.step_dt)
    check_step = max(1, round(cfg.check_s / env.step_dt))
    max_tilt = torch.zeros(cfg.num_envs, device=device)
    body_touch_steps = torch.zeros(cfg.num_envs, device=device)
    support_sum = torch.zeros(cfg.num_envs, device=device)
    check_tilt = torch.zeros(cfg.num_envs, device=device)
    check_drop = torch.zeros(cfg.num_envs, device=device)
    check_support = torch.zeros(cfg.num_envs, device=device)
    for step in range(steps):
        with torch.inference_mode():
            env.step(hold_action)
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        max_tilt = torch.maximum(max_tilt, tilt)
        contacts = microduck_mdp._stair_contacts(env)
        body_touch_steps += contacts["body_touch"].float()
        support_sum += contacts["foot_support"].float().mean(dim=1)
        if step + 1 == check_step:
            check_tilt = tilt.clone()
            check_drop = spawn_z - robot.data.root_link_pos_w[:, 2]
            check_support = contacts["foot_support"].float().mean(dim=1)
    final_tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
    drop = spawn_z - robot.data.root_link_pos_w[:, 2]
    fell = (max_tilt > math.radians(cfg.fall_tilt_deg)) | (drop > 0.04)
    contacts = microduck_mdp._stair_contacts(env)

    def summarize(mask: torch.Tensor) -> dict:
        count = int(mask.sum())
        if count == 0:
            return {"count": 0}
        return {
            "count": count,
            "stood_fraction": float((~fell[mask]).float().mean()),
            "check_tilt_deg_p50": float(torch.rad2deg(check_tilt[mask]).median()),
            "check_tilt_deg_p90": float(torch.rad2deg(check_tilt[mask]).quantile(0.9)),
            "check_drop_m_p90": float(check_drop[mask].quantile(0.9)),
            "check_feet_supported_mean": float(check_support[mask].mean()),
            "max_tilt_deg_p50": float(torch.rad2deg(max_tilt[mask]).median()),
            "max_tilt_deg_p90": float(torch.rad2deg(max_tilt[mask]).quantile(0.9)),
            "final_tilt_deg_p50": float(torch.rad2deg(final_tilt[mask]).median()),
            "drop_m_p50": float(drop[mask].median()),
            "drop_m_p90": float(drop[mask].quantile(0.9)),
            "feet_supported_fraction_mean": float(support_sum[mask].mean() / steps),
            "both_feet_supported_final": float(contacts["foot_support"][mask].all(dim=1).float().mean()),
            "body_touch_step_fraction": float(body_touch_steps[mask].mean() / steps),
        }

    result = {
        "level": level,
        "riser_m": (float(state.riser.min()), float(state.riser.max())),
        "angle_deg": (float(torch.rad2deg(state.angle).min()), float(torch.rad2deg(state.angle).max())),
        "ladder_spawn": summarize(~on_floor),
        "floor_spawn": summarize(on_floor),
    }
    env.close()
    return result


def main(cfg: Config) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = [_run_level(cfg, level, device) for level in cfg.levels]
    for r in results:
        print(json.dumps(r))
        ladder = r["ladder_spawn"]
        floor = r["floor_spawn"]
        ok = (
            ladder.get("count", 0) > 0
            and ladder["check_tilt_deg_p90"] <= floor.get("check_tilt_deg_p90", 10.0) + 5.0
            and ladder["check_drop_m_p90"] <= 0.02
            and ladder["check_feet_supported_mean"] >= 0.9
        )
        verdict = "PASS" if ok else "FAIL"
        print(
            f"LADDER_SETTLE level={r['level']} riser={r['riser_m'][0]*1000:.0f}-{r['riser_m'][1]*1000:.0f}mm "
            f"angle={r['angle_deg'][0]:.0f}-{r['angle_deg'][1]:.0f}deg "
            f"ladder@{cfg.check_s}s tilt_p90={ladder.get('check_tilt_deg_p90', float('nan')):.1f} "
            f"drop_p90={ladder.get('check_drop_m_p90', float('nan'))*1000:.0f}mm "
            f"support={ladder.get('check_feet_supported_mean', float('nan')):.2f} "
            f"| floor tilt_p90={floor.get('check_tilt_deg_p90', float('nan')):.1f} "
            f"| stood@{cfg.hold_s}s ladder={ladder.get('stood_fraction', float('nan')):.2f} "
            f"floor={floor.get('stood_fraction', float('nan')):.2f} {verdict}"
        )
    if cfg.output:
        with open(cfg.output, "w") as f:
            json.dump({"config": cfg.__dict__, "results": results}, f, indent=1)


if __name__ == "__main__":
    main(tyro.cli(Config))

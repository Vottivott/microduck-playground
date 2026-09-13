"""Play the phase-based reference gait open loop and report what happens.

Spawns on the ladder (static stance), then commands the reference leg
offsets from ``_stair_reference_offsets`` at the given climb speed with all
other joints at HOME.  If the open-loop reference itself climbs a riser or
two before falling, the reference is dynamically feasible and PPO can be
expected to track it; if it falls at once, the reference is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp as microduck_mdp


@dataclass(frozen=True)
class Config:
    task_id: str = "Mjlab-LadderClimb-MicroDuck"
    num_envs: int = 64
    level: int = 0
    climb_speed: float = 0.02
    duration_s: float = 6.0
    nominal_physics: bool = True
    trace_env: int = 0
    roll_amp: float = 0.0
    roll_sign: float = 1.0


def main(cfg: Config) -> None:
    configure_torch_backends()
    device = "cuda:0"
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.duration_s + 1.0
    env_cfg.terminations.clear()
    env_cfg.curriculum.clear()
    env_cfg.events.pop("push_robot", None)
    if cfg.nominal_physics:
        for name in ("foot_friction", "encoder_bias", "base_com", "randomize_com", "randomize_head_com",
                     "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"):
            env_cfg.events.pop(name, None)
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (cfg.climb_speed, cfg.climb_speed)
    twist.rel_standing_envs = 0.0
    spawn = env_cfg.events["reset_stair_ladder"].params
    spawn["fixed_level"] = cfg.level
    spawn["floor_spawn_prob"] = 0.0
    spawn["swing_spawn_prob"] = 0.0
    spawn["position_noise"] = 0.0
    spawn["yaw_noise_deg"] = 0.0
    spawn["tilt_noise_deg"] = 0.0
    spawn["joint_noise"] = 0.0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    robot = env.scene["robot"]
    state = env._stair
    servo_ids = microduck_mdp._servo_joint_ids(env, robot)
    left_ids = microduck_mdp._stair_leg_joint_ids(env, robot, "left")
    right_ids = microduck_mdp._stair_leg_joint_ids(env, robot, "right")
    # action index of each servo joint within the 14-D action (model order)
    pos_in_action = {jid: k for k, jid in enumerate(servo_ids)}
    left_a = [pos_in_action[j] for j in left_ids]
    right_a = [pos_in_action[j] for j in right_ids]
    roll_ids = [robot.find_joints("^left_hip_roll$")[0][0], robot.find_joints("^right_hip_roll$")[0][0]]
    roll_a = [pos_in_action[j] for j in roll_ids]
    microduck_mdp.STAIR_REF_ROLL_AMP = cfg.roll_amp
    microduck_mdp.STAIR_REF_ROLL_SIGN = cfg.roll_sign
    spawn_z = robot.data.root_link_pos_w[:, 2].clone()
    if not hasattr(state, "phase"):
        state.phase = torch.zeros(cfg.num_envs, device=device)
    steps = round(cfg.duration_s / env.step_dt)
    fell = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    for step in range(steps):
        rate = cfg.climb_speed / (2.0 * state.riser)
        state.phase = torch.remainder(state.phase + rate * env.step_dt, 1.0)
        left_ref, right_ref = microduck_mdp._stair_reference_offsets(env, state, state.phase)
        action = torch.zeros(cfg.num_envs, 14, device=device)
        action[:, left_a] = left_ref
        action[:, right_a] = right_ref
        roll = microduck_mdp._stair_reference_roll(state.phase)
        action[:, roll_a[0]] = roll
        action[:, roll_a[1]] = roll
        with torch.inference_mode():
            env.step(action)
        tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
        fell |= (tilt > math.radians(60.0)) | microduck_mdp.ladder_fallen(env)
        if step % 10 == 0:
            c = microduck_mdp._stair_contacts(env)
            e = cfg.trace_env
            g = robot.data.projected_gravity_b[e]
            sites = microduck_mdp._stair_foot_sites(env, robot)
            fz = robot.data.site_pos_w[e, sites, 2]
            root = robot.data.root_link_pos_w[e] - env.scene.env_origins[e]
            print(
                f"REF t={step * env.step_dt:.2f} phase={state.phase[e]:.2f} root=({root[0]:+.3f},{root[2]:.3f}) "
                f"grav=({g[0]:+.2f},{g[1]:+.2f}) feet_z=({fz[0]:.3f},{fz[1]:.3f}) "
                f"tread={c['foot_tread'][e].tolist()} sup={c['foot_support'][e].int().tolist()} fell={int(fell[e])}"
            )
    rise = robot.data.root_link_pos_w[:, 2] - spawn_z
    print(
        f"REF_SUMMARY level={cfg.level} speed={cfg.climb_speed} fell={fell.float().mean():.2f} "
        f"rise_p50={rise.median() * 1000:.0f}mm risers_p50={(rise / state.riser).median():.1f}"
    )
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))

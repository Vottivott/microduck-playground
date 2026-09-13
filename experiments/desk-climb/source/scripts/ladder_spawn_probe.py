"""Probe a few stair-ladder environments on the GPU: contacts, poses, tilt over time."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
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
    num_envs: int = 4
    level: int = 0
    floor_spawn_prob: float = 1.0
    hold_s: float = 1.5
    zero_action: bool = False
    nominal_physics: bool = False
    swing_spawn_prob: float = 0.0


def main(cfg: Config) -> None:
    configure_torch_backends()
    device = "cuda:0"
    env_cfg = load_env_cfg(cfg.task_id, play=False)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.terminations.clear()
    env_cfg.curriculum.clear()
    env_cfg.events.pop("push_robot", None)
    if cfg.nominal_physics:
        for name in ("foot_friction", "encoder_bias", "base_com", "randomize_com", "randomize_head_com",
                     "randomize_mass_inertia", "randomize_joint_friction", "randomize_armature"):
            env_cfg.events.pop(name, None)
    spawn = env_cfg.events["reset_stair_ladder"].params
    spawn["fixed_level"] = cfg.level
    spawn["floor_spawn_prob"] = cfg.floor_spawn_prob
    spawn["position_noise"] = 0.0
    spawn["yaw_noise_deg"] = 0.0
    spawn["tilt_noise_deg"] = 0.0
    spawn["joint_noise"] = 0.0
    spawn["swing_spawn_prob"] = cfg.swing_spawn_prob
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    robot = env.scene["robot"]
    state = env._stair
    model = env.sim.mj_model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}" for g in range(model.ngeom)]
    servo_ids = microduck_mdp._servo_joint_ids(env, robot)
    hold = (robot.data.joint_pos[:, servo_ids] - robot.data.default_joint_pos[:, servo_ids]).clone()
    if cfg.zero_action:
        hold.zero_()
    print("action term joint names:", env.action_manager.get_term("joint_pos")._joint_names
          if hasattr(env.action_manager.get_term("joint_pos"), "_joint_names") else "n/a")
    print("servo joint names:", [robot.joint_names[i] for i in servo_ids])
    print("hold action env0:", [round(v, 3) for v in hold[0].tolist()])
    print("origins:", env.scene.env_origins[: cfg.num_envs].tolist())
    print("root pos:", robot.data.root_link_pos_w[: cfg.num_envs].tolist())
    print("riser:", state.riser.tolist(), "angle deg:", torch.rad2deg(state.angle).tolist(), "x0:", state.x0.tolist())
    print("tread0 centre:", state.tread_centre[0, 0].tolist(), "tread1:", state.tread_centre[0, 1].tolist())
    tread0 = env.scene["tread_00"]
    print("tread_00 mocap pos (sim):", env.sim.data.mocap_pos[0].tolist()[:3])
    sites = microduck_mdp._stair_foot_sites(env, robot)
    print("foot sites:", robot.data.site_pos_w[0, sites].tolist())

    def dump_contacts(tag):
        count = int(env.sim.data.nacon[0].item())
        pairs = env.sim.data.contact.geom[:count].cpu()
        world = env.sim.data.contact.worldid[:count].cpu()
        dist = env.sim.data.contact.dist[:count].cpu()
        rows = []
        for i in range(count):
            if int(world[i]) != 0 or float(dist[i]) > 0:
                continue
            rows.append((names[int(pairs[i, 0])], names[int(pairs[i, 1])], round(float(dist[i]) * 1000, 2)))
        print(f"[{tag}] env0 contacts ({len(rows)}):", rows[:16])

    dump_contacts("t=0")
    steps = round(cfg.hold_s / env.step_dt)
    for step in range(steps):
        with torch.inference_mode():
            env.step(hold)
        if step in (0, 2, 5, 10, 25, 50, steps - 1):
            g = robot.data.projected_gravity_b[0]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-g[2])))))
            print(f"step {step:3d} t={step * env.step_dt:.2f}s root={[round(v, 3) for v in robot.data.root_link_pos_w[0].tolist()]} "
                  f"grav_b={[round(v, 3) for v in g.tolist()]} tilt={tilt:.1f} "
                  f"joints={[round(v, 2) for v in robot.data.joint_pos[0, servo_ids].tolist()]}")
            dump_contacts(f"step {step}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))

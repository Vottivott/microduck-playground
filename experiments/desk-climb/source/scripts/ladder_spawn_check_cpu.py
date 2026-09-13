"""CPU check of the stair-ladder spawn geometry with plain MuJoCo.

Rebuilds the scene (robot + treads + rails + floor) with the same placement
math as ``reset_stair_ladder`` and reports, for floor and on-ladder spawns,
which robot geoms touch which ladder geoms at t=0 and after a short hold with
stiff position servos.  No GPU needed; use it whenever the settle test on the
cluster reports unexpected contacts.

    uv run python scripts/ladder_spawn_check_cpu.py --riser-mm 16 --angle-deg 45 --start-tread 2
"""

from __future__ import annotations

import argparse
import math
import re

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot import ladder
from mjlab_microduck.robot.microduck_constants import HOME_FRAME, MICRODUCK_ALLCOLLISIONS_XML


def build(geometry: ladder.StairLadderGeometry, riser: float, angle_deg: float, x0: float, kp: float):
    spec = mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))
    for act in spec.actuators:
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.forcerange = [-0.96, 0.96]
    w = spec.worldbody
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], condim=3)
    angle = torch.tensor([math.radians(angle_deg)])
    centres, tops = ladder.tread_poses(geometry, torch.tensor([riser]), angle, torch.tensor([x0]), torch.zeros(1))
    for i in range(geometry.num_treads):
        w.add_geom(
            name=f"tread_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=centres[0, i].tolist(),
            size=[0.5 * geometry.tread_depth_m, geometry.tread_half_width_m, 0.5 * geometry.tread_thickness_m],
            condim=3,
        )
    rail_pos, rail_quat = ladder.rail_poses(geometry, angle, torch.tensor([x0]), torch.zeros(1))
    for j, name in enumerate(ladder.RAIL_ENTITY_NAMES):
        w.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=rail_pos[0, j].tolist(),
            quat=rail_quat[0, j].tolist(),
            size=[0.5 * geometry.rail_depth_m, 0.5 * geometry.rail_width_m, 0.5 * geometry.rail_length_m],
            condim=3,
        )
    return spec.compile(), tops[0]


def set_home(model, data):
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None or model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[j]] = value
                break


def contacts(model, data):
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > 0:
            continue
        a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
        out.append((a, b, round(float(c.dist) * 1000, 2)))
    return out


def run(args, on_floor: bool):
    g = ladder.LADDER_GEOMETRY
    riser, angle_deg = args.riser_mm * 1e-3, args.angle_deg
    angle = math.radians(angle_deg)
    run_ = riser / math.tan(angle)
    k = args.start_tread
    top_k = riser * (k + 1)
    if on_floor:
        x0 = ladder.SOLE_TOE_AHEAD_OF_SITE_M + 0.04 + g.tread_depth_m - run_
        root_z = 0.12 + 0.002
    else:
        x0 = 0.5 * g.tread_depth_m - top_k / math.tan(angle)
        root_z = top_k + ladder.SOLE_BOTTOM_ABOVE_SITE_M + 0.002 + ladder.HOME_TRUNK_ABOVE_SITE_M
    model, tops = build(g, riser, angle_deg, x0, args.kp)
    data = mujoco.MjData(model)
    set_home(model, data)
    data.qpos[0:3] = [0.0, 0.0, root_z]
    data.qpos[3:7] = [1, 0, 0, 0]
    if not on_floor:
        table = ladder.load_leg_table()
        hip, knee, ankle = ladder.leg_offsets_for(table, riser, run_)
        higher = "left" if (k + 1) % 2 == 0 else "right"
        sign = 1.0 if higher == "left" else -1.0
        for jname, d in ((f"{higher}_hip_pitch", hip), (f"{higher}_knee", knee), (f"{higher}_ankle", ankle)):
            data.qpos[model.jnt_qposadr[model.joint(jname).id]] += sign * d
    jn = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i][0]): i for i in range(model.nu)}
    for name, i in jn.items():
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.joint(name).id]]
    mujoco.mj_forward(model, data)
    label = "FLOOR" if on_floor else f"LADDER(k={k})"
    print(f"== {label} riser={args.riser_mm}mm angle={angle_deg} x0={x0:.3f} root_z={root_z:.3f}")
    for name in ("left_foot", "right_foot"):
        p = data.site_xpos[model.site(name).id]
        print(f"   {name} site {np.round(p, 3)}")
    print("   t=0 contacts:", contacts(model, data)[:12])
    for step in range(int(args.hold_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    R = data.xmat[model.body("trunk_base").id].reshape(3, 3)
    tilt = math.degrees(math.acos(min(1.0, max(-1.0, R[2, 2]))))
    print(f"   after {args.hold_s}s: trunk z={data.qpos[2]:.3f} (drop {root_z - data.qpos[2]:.3f}) tilt={tilt:.1f}deg")
    print("   contacts:", contacts(model, data)[:12])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--riser-mm", type=float, default=16.0)
    parser.add_argument("--angle-deg", type=float, default=45.0)
    parser.add_argument("--start-tread", type=int, default=2)
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    args = parser.parse_args()
    run(args, on_floor=True)
    run(args, on_floor=False)


if __name__ == "__main__":
    main()

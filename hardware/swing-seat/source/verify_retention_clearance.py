#!/usr/bin/env python3
"""Check that the complete retention kit adds no collision to valid robot poses.

The comparison deliberately ignores poses that already collide with the stock
seat or with the robot itself.  All remaining poses must clear the removable
strap, compliant battery bumpers, buckle, and the previously verified locating
pads, with an optional manufacturing margin on the soft parts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np
import trimesh

from generate_seat import PP, locator_pad_meshes
from retention_system import bumper_meshes, buckle_meshes, strap_mesh
from verify_locator_clearance import (
    CHAINS,
    ROBOT_XML,
    chain_geom_ids,
    joint_info,
    mesh_for_geom,
    poses_sobol,
    poses_structured,
)


HERE = Path(__file__).resolve().parent
MARGIN_MM = float(os.environ.get("RETENTION_SAFETY_MARGIN_MM", "1.0"))


class Comparison:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.self_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.self_data = mujoco.MjData(self.self_model)

        self.original = trimesh.collision.CollisionManager()
        original = trimesh.load(HERE / "output" / "param_seat.stl", force="mesh")
        self.original.add_object("original_seat", original)

        # The stock XML only enables simplified policy collision geoms.  Add
        # exact trunk visual geometry so an extreme pose that already passes
        # through the fixed battery/trunk is not mislabelled as useful range
        # lost to the retention kit.
        self.fixed_robot = trimesh.collision.CollisionManager()
        fixed_data = mujoco.MjData(model)
        fixed_root = model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        ]
        fixed_data.qpos[fixed_root:fixed_root + 7] = (0, 0, 0, 1, 0, 0, 0)
        mujoco.mj_forward(model, fixed_data)
        trunk_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != trunk_id:
                continue
            mesh = mesh_for_geom(model, geom_id)
            if mesh is not None:
                transform = np.eye(4)
                transform[:3, :3] = fixed_data.geom_xmat[geom_id].reshape(3, 3)
                transform[:3, 3] = fixed_data.geom_xpos[geom_id]
                self.fixed_robot.add_object(
                    f"fixed_g{geom_id}", mesh, transform=transform
                )

        self.retention = trimesh.collision.CollisionManager()
        params = {**PP, "locator_pads": True}
        parts = []
        parts.extend((f"locator_{i}", m) for i, m in enumerate(
            locator_pad_meshes(params, margin=MARGIN_MM)
        ))
        parts.append(("strap", strap_mesh(margin=MARGIN_MM)))
        parts.extend((f"bumper_{i}", m) for i, m in enumerate(
            bumper_meshes(margin=MARGIN_MM)
        ))
        # The metal frame is not expanded: it sits outside the articulated
        # corridors and should retain its exact manufactured envelope.
        parts.extend((f"buckle_{i}", m) for i, m in enumerate(buckle_meshes()))
        for name, mesh_mm in parts:
            mesh = mesh_mm.copy()
            mesh.apply_scale(0.001)
            self.retention.add_object(name, mesh)

        self.moving = {}
        for chain, spec in CHAINS.items():
            manager = trimesh.collision.CollisionManager()
            geom_ids = []
            for geom_id in chain_geom_ids(model, spec["bodies"]):
                mesh = mesh_for_geom(model, geom_id)
                if mesh is None:
                    continue
                manager.add_object(f"g{geom_id}", mesh)
                geom_ids.append(geom_id)
            self.moving[chain] = manager, geom_ids

    def classify(self, data: mujoco.MjData, chain: str):
        manager, geom_ids = self.moving[chain]
        for geom_id in geom_ids:
            transform = np.eye(4)
            transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
            transform[:3, 3] = data.geom_xpos[geom_id]
            manager.set_transform(f"g{geom_id}", transform)
        hit_retention, names = manager.in_collision_other(
            self.retention, return_names=True
        )
        hit_original = bool(manager.in_collision_other(self.original))
        # The leg meshes intentionally overlap their trunk-side bearing faces
        # at the kinematic joint, so trunk-visual filtering is meaningful only
        # for the head chain.  Leg self validity remains governed by the XML's
        # purpose-built collision geoms, as in the original locator verifier.
        hit_fixed_robot = (
            bool(manager.in_collision_other(self.fixed_robot))
            if chain == "head" else False
        )
        self.self_data.qpos[:] = data.qpos[: self.self_model.nq]
        self.self_data.qvel[:] = 0
        mujoco.mj_forward(self.self_model, self.self_data)
        return (
            bool(hit_retention), hit_original, hit_fixed_robot,
            self.self_data.ncon > 0, sorted(names),
        )


def run_family(model, data, exact, name, poses):
    spec = CHAINS[name]
    addrs, ranges = joint_info(model, spec["joints"])
    tested = valid = blocked_original = blocked_fixed_robot = self_colliding = 0
    contacts_by_part = {}
    for pose in poses:
        for adr, value in zip(addrs, pose):
            data.qpos[adr] = value
        mujoco.mj_forward(model, data)
        hit, hit_original, hit_fixed_robot, hit_self, names = exact.classify(data, name)
        tested += 1
        if hit_self:
            self_colliding += 1
            continue
        if hit_original:
            blocked_original += 1
            continue
        if hit_fixed_robot:
            blocked_fixed_robot += 1
            continue
        valid += 1
        if hit:
            for _, fixed in names:
                contacts_by_part[fixed] = contacts_by_part.get(fixed, 0) + 1
            raise RuntimeError(
                f"{name} gains a NEW retention collision at "
                + json.dumps(dict(zip(spec["joints"], map(float, pose))))
                + f"; parts={names}"
            )
    return {
        "tested": tested,
        "baseline_valid": valid,
        "already_blocked_by_original": blocked_original,
        "already_blocked_by_fixed_robot_visuals": blocked_fixed_robot,
        "self_colliding": self_colliding,
        "newly_blocked": 0,
        "contacts_by_part": contacts_by_part,
        "ranges": dict(zip(spec["joints"], ranges.tolist())),
    }


def main():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    root = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    ]
    data.qpos[root:root + 7] = (0, 0, 0, 1, 0, 0, 0)
    exact = Comparison(model)
    report = {
        "soft_part_safety_margin_mm": MARGIN_MM,
        "method": "exact triangle FCL, excluding robot-self and original-seat collisions",
        "families": {},
    }
    seeds = {"left_leg": 731, "right_leg": 947, "head": 1217}
    for name, spec in CHAINS.items():
        _, ranges = joint_info(model, spec["joints"])
        family = {
            "structured": run_family(model, data, exact, name, poses_structured(ranges, 5)),
            "sobol": run_family(
                model, data, exact, name,
                poses_sobol(ranges, 15 if "leg" in name else 14, seeds[name]),
            ),
        }
        if "leg" in name:
            dense = ((0.0, 0.0) + tuple(p) for p in poses_structured(ranges[2:], 17))
            family["dense_sagittal"] = run_family(model, data, exact, name, dense)
        report["families"][name] = family

    out = HERE / "output_retained" / "retention_clearance_report_margin_1mm.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    total = sum(v["tested"] for f in report["families"].values() for v in f.values())
    valid = sum(v["baseline_valid"] for f in report["families"].values() for v in f.values())
    print(
        f"PASS: {total:,} sampled poses ({valid:,} baseline-valid), no new "
        f"collision with the complete retention system at {MARGIN_MM:g} mm margin"
    )
    print("report", out)


if __name__ == "__main__":
    main()

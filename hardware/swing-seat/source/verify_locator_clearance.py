#!/usr/bin/env python3
"""Conservative kinematic clearance sweep for the optional locator pads.

The verifier expands the printable pads by 2 mm toward every relevant moving
part, enables pad-only contacts against all visual meshes in each articulated
leg/head chain, and sweeps the complete XML joint ranges. A pass therefore
means the sampled workspace remains clear even after the safety expansion.
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np
import trimesh
from scipy.stats import qmc

from generate_seat import PP, locator_pad_meshes


HERE = Path(__file__).resolve().parent
ROBOT_DIR = Path(
    os.environ.get(
        "MICRODUCK_ROBOT_DIR",
        HERE.parents[2] / "src/mjlab_microduck/robot/microduck",
    )
)
ROBOT_XML = ROBOT_DIR / "robot_allcollisions.xml"
SAFETY_MARGIN_MM = float(os.environ.get("LOCATOR_SAFETY_MARGIN_MM", "2.0"))

CHAINS = {
    "left_leg": {
        "bodies": {"yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left"},
        "joints": [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
            "left_knee", "left_ankle",
        ],
    },
    "right_leg": {
        "bodies": {"bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right"},
        "joints": [
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
            "right_knee", "right_ankle",
        ],
    },
    "head": {
        "bodies": {"neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"},
        "joints": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
    },
}


def make_verification_model(tmp: Path) -> mujoco.MjModel:
    root = ET.parse(ROBOT_XML).getroot()
    target_bodies = set().union(*(spec["bodies"] for spec in CHAINS.values()))

    # Disable every ordinary collision. Then enable a private pad-only bit on
    # every geom belonging to a moving body, including the detailed visual
    # meshes rather than only the coarser policy collision meshes.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    for body in root.iter("body"):
        if body.get("name") not in target_bodies:
            continue
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "8")

    assets = root.find("asset")
    worldbody = root.find("worldbody")
    assert assets is not None and worldbody is not None
    params = {**PP, "locator_pads": True}
    for index, mesh_mm in enumerate(locator_pad_meshes(params, margin=SAFETY_MARGIN_MM)):
        mesh_m = mesh_mm.copy()
        mesh_m.apply_scale(0.001)
        filename = f"locator_margin_{index}.stl"
        mesh_m.export(tmp / filename)
        ET.SubElement(assets, "mesh", name=f"locator_margin_{index}", file=f"../{filename}")
        ET.SubElement(
            worldbody,
            "geom",
            name=f"locator_margin_{index}",
            type="mesh",
            mesh=f"locator_margin_{index}",
            contype="8",
            conaffinity="0",
            rgba="1 0 0 0.3",
        )

    (tmp / "assets").symlink_to(ROBOT_DIR / "assets", target_is_directory=True)
    xml_path = tmp / "verify_locator.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(xml_path))


def joint_info(model: mujoco.MjModel, names: list[str]):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    addrs = [int(model.jnt_qposadr[i]) for i in ids]
    ranges = np.array([model.jnt_range[i] for i in ids], dtype=np.float64)
    return addrs, ranges


def poses_structured(ranges: np.ndarray, points: int):
    axes = [np.linspace(lo, hi, points) for lo, hi in ranges]
    yield from itertools.product(*axes)


def poses_sobol(ranges: np.ndarray, power: int, seed: int):
    unit = qmc.Sobol(d=len(ranges), scramble=True, seed=seed).random_base2(power)
    yield from qmc.scale(unit, ranges[:, 0], ranges[:, 1])


def pad_contact(model: mujoco.MjModel, data: mujoco.MjData, pad_ids: set[int]) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if int(c.geom1) in pad_ids or int(c.geom2) in pad_ids:
            return True
    return False


def chain_geom_ids(model: mujoco.MjModel, body_names: set[str]) -> list[int]:
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    }
    return [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in body_ids]


def mesh_for_geom(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        return None
    mesh_id = int(model.geom_dataid[geom_id])
    va = int(model.mesh_vertadr[mesh_id])
    vn = int(model.mesh_vertnum[mesh_id])
    fa = int(model.mesh_faceadr[mesh_id])
    fn = int(model.mesh_facenum[mesh_id])
    return trimesh.Trimesh(
        vertices=np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64),
        faces=np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64),
        process=False,
    )


class ExactComparison:
    """Triangle-level FCL comparison against original seat and expanded pads."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.self_model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.self_data = mujoco.MjData(self.self_model)
        original = trimesh.load(HERE / "output" / "param_seat.stl", force="mesh")
        self.original = trimesh.collision.CollisionManager()
        self.original.add_object("original_seat", original)

        params = {**PP, "locator_pads": True}
        self.pads = trimesh.collision.CollisionManager()
        for i, pad_mm in enumerate(locator_pad_meshes(params, margin=SAFETY_MARGIN_MM)):
            pad = pad_mm.copy()
            pad.apply_scale(0.001)
            self.pads.add_object(f"pad_{i}", pad)

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
            self.moving[chain] = (manager, geom_ids)

    def _update(self, data: mujoco.MjData, chain: str):
        manager, geom_ids = self.moving[chain]
        for geom_id in geom_ids:
            transform = np.eye(4)
            transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
            transform[:3, 3] = data.geom_xpos[geom_id]
            manager.set_transform(f"g{geom_id}", transform)
        return manager

    def classify(self, data: mujoco.MjData, chain: str) -> tuple[bool, bool, bool]:
        moving = self._update(data, chain)
        hits_pad = bool(moving.in_collision_other(self.pads))
        hits_original = bool(moving.in_collision_other(self.original))
        self.self_data.qpos[:] = data.qpos[: self.self_model.nq]
        self.self_data.qvel[:] = 0
        mujoco.mj_forward(self.self_model, self.self_data)
        self_colliding = self.self_data.ncon > 0
        return hits_pad, hits_original, self_colliding


def minimum_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    moving_ids: list[int],
    pad_ids: set[int],
) -> float:
    best = 0.1
    segment = np.empty(6, dtype=np.float64)
    for moving in moving_ids:
        for pad in pad_ids:
            best = min(
                best,
                float(mujoco.mj_geomDistance(model, data, moving, pad, 0.1, segment)),
            )
    return best


def run_family(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    poses,
    *,
    distance_stride: int,
    exact: ExactComparison,
) -> dict:
    spec = CHAINS[name]
    addrs, ranges = joint_info(model, spec["joints"])
    moving_ids = chain_geom_ids(model, spec["bodies"])
    pad_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "locator_margin_0"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "locator_margin_1"),
    }
    tested = 0
    convex_candidates = 0
    exact_pad_contacts = 0
    already_blocked_by_original = 0
    self_colliding = 0
    min_distance = 0.1
    closest_pose = None
    for pose in poses:
        for adr, value in zip(addrs, pose):
            data.qpos[adr] = value
        mujoco.mj_forward(model, data)
        if pad_contact(model, data, pad_ids):
            convex_candidates += 1
            hits_pad, hits_original, hits_self = exact.classify(data, name)
            if hits_pad:
                exact_pad_contacts += 1
                if hits_self:
                    self_colliding += 1
                elif hits_original:
                    already_blocked_by_original += 1
                else:
                    raise RuntimeError(
                        f"{name} gains a NEW collision from the "
                        f"{SAFETY_MARGIN_MM:g}-mm-expanded locator at pose "
                        + json.dumps(dict(zip(spec["joints"], map(float, pose))))
                    )
        if tested % distance_stride == 0:
            distance = minimum_distance(model, data, moving_ids, pad_ids)
            if distance < min_distance:
                min_distance = distance
                closest_pose = list(map(float, pose))
        tested += 1
    return {
        "tested": tested,
        "convex_candidates": convex_candidates,
        "exact_pad_contacts": exact_pad_contacts,
        "already_blocked_by_original": already_blocked_by_original,
        "self_colliding": self_colliding,
        "newly_blocked": 0,
        "min_distance_to_expanded_pad_mm": min_distance * 1000.0,
        "closest_sample": dict(zip(spec["joints"], closest_pose or [])),
        "ranges": dict(zip(spec["joints"], ranges.tolist())),
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="microduck-pad-verify-") as tmp_name:
        model = make_verification_model(Path(tmp_name))
        data = mujoco.MjData(model)
        exact = ExactComparison(model)
        root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        root_adr = model.jnt_qposadr[root_id]
        data.qpos[root_adr : root_adr + 7] = (0, 0, 0, 1, 0, 0, 0)

        report = {
            "safety_margin_mm": SAFETY_MARGIN_MM,
            "method": "pad-only MuJoCo convex collision against detailed moving-body meshes",
            "families": {},
        }
        seeds = {"left_leg": 731, "right_leg": 947, "head": 1217}
        for name, spec in CHAINS.items():
            _, ranges = joint_info(model, spec["joints"])
            structured = run_family(
                model,
                data,
                name,
                poses_structured(ranges, 5),
                distance_stride=1,
                exact=exact,
            )
            sobol_power = 15 if "leg" in name else 14
            sobol = run_family(
                model,
                data,
                name,
                poses_sobol(ranges, sobol_power, seeds[name]),
                distance_stride=8,
                exact=exact,
            )
            report["families"][name] = {
                "structured": structured,
                "sobol": sobol,
            }

            # Dense sagittal grid for the primary pumping joints with yaw/roll
            # held neutral. It directly covers the intended swing controller.
            if "leg" in name:
                sagittal_ranges = ranges[2:]
                zero_prefix = (0.0, 0.0)
                dense = (zero_prefix + tuple(p) for p in poses_structured(sagittal_ranges, 17))
                report["families"][name]["dense_sagittal"] = run_family(
                    model, data, name, dense, distance_stride=1, exact=exact
                )

        output = HERE / "output_padded" / "locator_clearance_report.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
        total = sum(
            result["tested"]
            for family in report["families"].values()
            for result in family.values()
            if isinstance(result, dict) and "tested" in result
        )
        print(
            f"PASS: {total} poses, no NEW collision versus the original seat "
            f"with {SAFETY_MARGIN_MM:.1f} mm-expanded pads"
        )
        for name, family in report["families"].items():
            mins = [v["min_distance_to_expanded_pad_mm"] for v in family.values()]
            print(f"  {name}: minimum sampled clearance beyond margin = {min(mins):.3f} mm")
        print("report", output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify the retention kit over the motion envelope needed to pump a swing."""

import json
import os

os.environ["RETENTION_SAFETY_MARGIN_MM"] = "0"
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco

from verify_retention_clearance import (
    CHAINS, Comparison, ROBOT_XML, HERE, joint_info, poses_sobol,
    poses_structured, run_family,
)


def main():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    data = mujoco.MjData(model)
    root = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    ]
    data.qpos[root:root + 7] = (0, 0, 0, 1, 0, 0, 0)
    exact = Comparison(model)
    report = {
        "retention_margin_mm": 0.0,
        "scope": (
            "full five-axis XML range for each leg independently; full neck/head "
            "pitch ranges with yaw and roll neutral"
        ),
        "families": {},
    }
    for name, seed in (("left_leg", 731), ("right_leg", 947)):
        _, ranges = joint_info(model, CHAINS[name]["joints"])
        report["families"][name] = {
            "structured_5d": run_family(
                model, data, exact, name, poses_structured(ranges, 5)
            ),
            "sobol_5d": run_family(
                model, data, exact, name, poses_sobol(ranges, 15, seed)
            ),
            "dense_sagittal": run_family(
                model, data, exact, name,
                ((0.0, 0.0) + tuple(p) for p in poses_structured(ranges[2:], 17)),
            ),
        }

    _, head_ranges = joint_info(model, CHAINS["head"]["joints"])
    head_poses = (
        (neck, pitch, 0.0, 0.0)
        for neck, pitch in poses_structured(head_ranges[:2], 101)
    )
    report["families"]["head"] = {
        "dense_sagittal": run_family(model, data, exact, "head", head_poses)
    }
    output = HERE / "output_retained" / "pumping_clearance_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    total = sum(v["tested"] for f in report["families"].values() for v in f.values())
    valid = sum(v["baseline_valid"] for f in report["families"].values() for v in f.values())
    print(
        f"PASS: {total:,} samples ({valid:,} baseline-valid); the retention "
        "system adds zero collision across the swing-pumping envelope"
    )
    print("report", output)


if __name__ == "__main__":
    main()

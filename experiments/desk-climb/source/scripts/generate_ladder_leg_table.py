"""Generate the staggered-stance leg table used by the stair-ladder reset.

On an alternating-tread ladder the two feet never rest at the same height:
one foot stands one riser above the other and one run further forward.  The
reset event therefore needs a joint pose for the *higher* leg that shortens it
vertically by ``delta`` and moves the sole forward by ``forward`` while keeping
the sole level.  This script solves that with a coarse-to-fine grid search on
the plain MuJoCo model (CPU, a few seconds) and writes the table that
``mjlab_microduck.robot.ladder`` ships.

Run once after any change to the leg kinematics::

    uv run python scripts/generate_ladder_leg_table.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import (
    HOME_FRAME,
    MICRODUCK_ALLCOLLISIONS_XML,
)

# Vertical shortening and forward shift of the higher foot, in millimetres.
DELTA_MM = tuple(range(0, 85, 5))
FORWARD_MM = tuple(range(-35, 80, 5))
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mjlab_microduck"
    / "robot"
    / "ladder_leg_table.json"
)


def _home_joint_pos(model: mujoco.MjModel) -> dict[str, float]:
    import re

    values = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name is None or model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        for pattern, value in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                values[name] = float(value)
                break
    return values


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    data = mujoco.MjData(model)
    home = _home_joint_pos(model)

    def qadr(name: str) -> int:
        return int(model.jnt_qposadr[model.joint(name).id])

    for name, value in home.items():
        data.qpos[qadr(name)] = value
    mujoco.mj_kinematics(model, data)
    site_id = model.site("left_foot").id
    home_pos = data.site_xpos[site_id].copy()
    home_mat = data.site_xmat[site_id].reshape(3, 3).copy()

    hip = qadr("left_hip_pitch")
    knee = qadr("left_knee")
    ankle = qadr("left_ankle")
    base = (home["left_hip_pitch"], home["left_knee"], home["left_ankle"])

    def evaluate(dh: float, dk: float, da: float) -> tuple[np.ndarray, float]:
        data.qpos[hip] = base[0] + dh
        data.qpos[knee] = base[1] + dk
        data.qpos[ankle] = base[2] + da
        mujoco.mj_kinematics(model, data)
        pos = data.site_xpos[site_id] - home_pos
        mat = data.site_xmat[site_id].reshape(3, 3)
        # Sole orientation error: angle between HOME and candidate site frames.
        cos_err = 0.5 * (np.trace(home_mat.T @ mat) - 1.0)
        return pos.copy(), math.degrees(math.acos(min(1.0, max(-1.0, cos_err))))

    def solve(delta: float, forward: float) -> tuple[tuple[float, float, float], float]:
        best = None
        # Coarse grid over hip and knee; the ankle keeps the sole level to first
        # order (planar chain: hip + knee + ankle = const), then refine.
        for dh in np.linspace(-1.5, 1.5, 61):
            for dk in np.linspace(-1.5, 1.5, 61):
                da = -(dh + dk)
                pos, ang = evaluate(dh, dk, da)
                err = (
                    (pos[2] - delta) ** 2
                    + (pos[0] - forward) ** 2
                    + (pos[1]) ** 2
                    + (0.002 * ang) ** 2
                )
                if best is None or err < best[0]:
                    best = (err, dh, dk, da)
        assert best is not None
        _, dh, dk, da = best
        step = 0.05
        for _ in range(60):
            improved = False
            for ddh, ddk, dda in (
                (step, 0, 0), (-step, 0, 0), (0, step, 0), (0, -step, 0),
                (0, 0, step), (0, 0, -step),
            ):
                pos, ang = evaluate(dh + ddh, dk + ddk, da + dda)
                err = (
                    (pos[2] - delta) ** 2
                    + (pos[0] - forward) ** 2
                    + (pos[1]) ** 2
                    + (0.002 * ang) ** 2
                )
                if err < best[0]:
                    best = (err, dh + ddh, dk + ddk, da + dda)
                    dh, dk, da = best[1:]
                    improved = True
            if not improved:
                step *= 0.5
        pos, ang = evaluate(dh, dk, da)
        residual = float(np.hypot(pos[2] - delta, pos[0] - forward))
        return (float(dh), float(dk), float(da)), max(residual, 0.0), ang

    table = {}
    worst = 0.0
    for delta_mm in DELTA_MM:
        for forward_mm in FORWARD_MM:
            offsets, residual, ang = solve(delta_mm * 1e-3, forward_mm * 1e-3)
            worst = max(worst, residual)
            table[f"{delta_mm}:{forward_mm}"] = {
                "hip_pitch": offsets[0],
                "knee": offsets[1],
                "ankle": offsets[2],
                "residual_m": residual,
                "sole_tilt_deg": ang,
            }
            print(
                f"delta={delta_mm:2d}mm forward={forward_mm:3d}mm "
                f"hip={offsets[0]:+.3f} knee={offsets[1]:+.3f} ankle={offsets[2]:+.3f} "
                f"residual={residual * 1000:.2f}mm tilt={ang:.1f}deg"
            )
    payload = {
        "description": (
            "Left-leg joint offsets from HOME (hip_pitch, knee, ankle) that lift "
            "the left sole by delta and move it forward by forward while keeping "
            "it level. Right leg = negated offsets. Keys are 'delta_mm:forward_mm'."
        ),
        "delta_mm": list(DELTA_MM),
        "forward_mm": list(FORWARD_MM),
        "table": table,
        "worst_residual_m": worst,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUTPUT} (worst residual {worst * 1000:.2f} mm)")


if __name__ == "__main__":
    main()

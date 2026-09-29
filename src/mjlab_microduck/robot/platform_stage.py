"""Stage for the platform jump (user sketch 2026-09-18): the robot stands on
a raised block A and long-jumps DOWN onto a lower, separate block B; the
floor in between is the pit.

Env frame: A's front edge is the line x = 0 (A occupies x < 0); B's near
edge is at x = GAP (B occupies GAP < x < GAP + B_DEPTH).  B's top is at
B_TOP above the floor; A's top is B_TOP + DROP.  Both blocks are mocap
bodies placed per environment at reset (the gap and drop are per-env
curriculum variables).  The boxes are tall and poke below the floor plane
so one fixed geom size serves every height.
"""
from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg

PLATFORM_A = "platform_a"
PLATFORM_B = "platform_b"

A_DEPTH_M = 0.60          # along x
B_DEPTH_M = 0.60
WIDTH_M = 0.60            # along y, both blocks
BLOCK_HEIGHT_M = 0.80     # box height; the part below the floor is hidden
B_TOP_M = 0.10            # B's top above the floor (fixed)
PARKED_POS = (0.0, 0.0, -5.0)


def _block_spec(name: str, depth: float, rgba: str) -> mujoco.MjSpec:
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}_collision" type="box" size="{0.5 * depth:.4f} {0.5 * WIDTH_M:.4f} {0.5 * BLOCK_HEIGHT_M:.4f}"
            rgba="{rgba}" friction="1.0 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def platform_a_spec() -> mujoco.MjSpec:
    return _block_spec(PLATFORM_A, A_DEPTH_M, "0.62 0.52 0.40 1")


def platform_b_spec() -> mujoco.MjSpec:
    return _block_spec(PLATFORM_B, B_DEPTH_M, "0.45 0.55 0.62 1")


def make_platform_stage_entity_cfgs() -> dict[str, EntityCfg]:
    parked = EntityCfg.InitialStateCfg(pos=PARKED_POS)
    return {
        PLATFORM_A: EntityCfg(spec_fn=platform_a_spec, init_state=parked),
        PLATFORM_B: EntityCfg(spec_fn=platform_b_spec, init_state=parked),
    }


def a_top(drop: float) -> float:
    return B_TOP_M + drop


def platform_a_center(drop: float) -> tuple[float, float, float]:
    """Mocap position of block A for a drop (its top is B_TOP + drop)."""
    return (-0.5 * A_DEPTH_M, 0.0, a_top(drop) - 0.5 * BLOCK_HEIGHT_M)


def platform_b_center(gap: float) -> tuple[float, float, float]:
    return (gap + 0.5 * B_DEPTH_M, 0.0, B_TOP_M - 0.5 * BLOCK_HEIGHT_M)

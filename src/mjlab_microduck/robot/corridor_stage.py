"""Two vertical walls forming a narrow corridor, for braced climbing (user
request 2026-09-18: "learning to wall jump from side to side to get up a
narrow vertical corridor").

Jumping up such a corridor is ruled out by the robot's vertical launch speed
(0.04-0.26 m/s: every bounce would have to beat gravity on its own).  What
the 2026-09-18 bracing measurement DID find is that the duck wedges between
walls 12-14 cm apart with its back shell on one and both feet on the other:

  * 12 cm holds at wall friction as low as 0.3 - a geometric wedge, not a
    friction grip - and crept UPWARD 0.6 cm in 2 s;
  * 14 cm holds only for friction >= 0.8;
  * 16 cm and wider: nothing held.

so this stage keeps the corridor inside that band.  Each wall is its own
mocap entity placed at reset, so the width is a per-environment curriculum
variable (the stair-ladder pattern).
"""
from __future__ import annotations

import os as _os

import mujoco

from mjlab.entity import EntityCfg

WALL_LEFT = "corridor_left"
WALL_RIGHT = "corridor_right"
WALL_THICKNESS_M = 0.06
# along y (the corridor runs along y; x is the gap).  Env-overridable because
# the EXIT stage extends the walls out over the landing platform, so they can
# still be braced against while the robot travels horizontally.
WALL_DEPTH_M = float(_os.getenv("MICRODUCK_CH_WALL_DEPTH", "0.30"))
WALL_HEIGHT_M = float(_os.getenv("MICRODUCK_CH_WALL_H", "2.40"))   # taller for long-climb runs
FLOOR_TO_WALL_BASE_M = 0.0   # the walls stand on the floor
# Parked below the floor until the reset event places them.  The depth has to
# scale with the wall: a fixed -10 m hides a 4 m wall (top at -8) but a 25 m
# wall is half-height 12.5, so parking it at -10 puts its TOP 2.5 m above the
# floor, intersecting every robot at spawn.  That surfaced as a
# "nconmax overflow" at env construction, not as anything about the corridor.
PARKED_POS = (0.0, 0.0, -(0.5 * WALL_HEIGHT_M + 5.0))

# the measured working band (see the module docstring)
WIDTH_MIN_M = 0.115
WIDTH_MAX_M = 0.145


# Height marks on the inner face, purely visual.  OFF by default: the user
# asked for them removed once clips carried a live height readout instead
# (2026-09-18).  MICRODUCK_CH_WALL_MARKS=1 brings them back.
WALL_MARKS = _os.getenv("MICRODUCK_CH_WALL_MARKS", "0") != "0"
MARKER_SPACING_M = 1.0
MARKER_THICKNESS_M = 0.012
MARKER_RGBA = "0.95 0.75 0.20 1"


def _markers(name: str) -> str:
    """Decorative height marks every metre up the inner face.

    A blank corridor gives a video no scale: the 60 s render at 11 m framing
    showed the robot as a 20-pixel speck between two featureless lines, and a
    follow camera on blank walls shows motion but no height.  These are
    contype=0 conaffinity=0, so they are invisible to physics and to the wall
    friction randomiser (which targets ``<name>_collision`` by name).
    """
    if not WALL_MARKS:
        return ""
    out = []
    n = int(WALL_HEIGHT_M / MARKER_SPACING_M)
    for i in range(1, n + 1):
        z = i * MARKER_SPACING_M - 0.5 * WALL_HEIGHT_M
        out.append(
            f'<geom name="{name}_mark{i:02d}" type="box" contype="0" conaffinity="0" '
            f'size="{0.5 * WALL_THICKNESS_M + 0.002:.4f} {0.5 * WALL_DEPTH_M:.4f} {0.5 * MARKER_THICKNESS_M:.4f}" '
            f'pos="0 0 {z:.4f}" rgba="{MARKER_RGBA}"/>'
        )
    return "\n      ".join(out)


def _wall_spec(name: str, rgba: str) -> mujoco.MjSpec:
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}_collision" type="box"
            size="{0.5 * WALL_THICKNESS_M:.4f} {0.5 * WALL_DEPTH_M:.4f} {0.5 * WALL_HEIGHT_M:.4f}"
            rgba="{rgba}" friction="0.9 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
      {_markers(name)}
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_corridor_entity_cfgs() -> dict[str, EntityCfg]:
    parked = EntityCfg.InitialStateCfg(pos=PARKED_POS)
    return {
        WALL_LEFT: EntityCfg(spec_fn=lambda: _wall_spec(WALL_LEFT, "0.60 0.55 0.50 1"), init_state=parked),
        WALL_RIGHT: EntityCfg(spec_fn=lambda: _wall_spec(WALL_RIGHT, "0.54 0.50 0.46 1"), init_state=parked),
    }


def wall_x(width: float, side: int) -> float:
    """Mocap x of a wall centre: side -1 = left (behind the robot's back), +1 = right."""
    return side * (0.5 * width + 0.5 * WALL_THICKNESS_M)


def wall_z() -> float:
    return 0.5 * WALL_HEIGHT_M + FLOOR_TO_WALL_BASE_M

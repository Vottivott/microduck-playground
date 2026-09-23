"""The top of the vertical corridor: a landing platform on the near side, with
the walls overhanging it (user sketch 2026-09-20).

Geometry, in the corridor's own frame (the slot runs along y, the gap is along
x, and the robot climbs in +z):

                        |   |          <- walls continue above
        platform top    |   |
      ==================|   |          <- PLATFORM_TOP_M, extends toward +y
                        |   |
                        |   |          <- the climbing corridor
                        |   |

  * the two climbing walls are EXTENDED along +y by ``OVERHANG_M`` so they
    continue out over the platform.  That is what lets the robot keep bracing
    while it travels horizontally instead of vertically - the user's "safety
    walls on the platform";
  * the platform is a slab whose TOP sits at ``PLATFORM_TOP_M`` and which
    starts at the corridor's original near face (y = +WALL_DEPTH/2) and runs
    out to +y.  It is not inside the corridor: the climb passes it on the way
    up and the exit drops onto it.

So the route is: climb the slot to just above the platform, switch policy,
traverse out along +y still braced between the overhanging walls, then descend
onto the platform and stand.
"""
from __future__ import annotations

import os as _os

import mujoco

from mjlab.entity import EntityCfg

from mjlab_microduck.robot import corridor_stage as cs

PLATFORM = "exit_platform"

# how far the walls reach out over the platform
OVERHANG_M = float(_os.getenv("MICRODUCK_EX_OVERHANG", "0.45"))
# ...and how much further the PLATFORM runs once the walls have ended, leaving
# open platform with nothing to brace against (user 2026-09-20: "make the
# above-platform walls end after a while so there is also a region with only
# the platform").  The exit has to side-walk out of the walled part onto this.
OPEN_M = float(_os.getenv("MICRODUCK_EX_OPEN", "0.40"))
# the platform's top surface, above the floor
PLATFORM_TOP_M = float(_os.getenv("MICRODUCK_EX_PLATFORM_TOP", "1.20"))
PLATFORM_WIDTH_M = float(_os.getenv("MICRODUCK_EX_PLATFORM_W", "0.50"))   # along x
PLATFORM_THICK_M = 0.12
# Clear air between the climbing slot and the platform's near edge.  With the
# platform starting flush at the slot mouth the duck clipped its underside on
# the way past and snagged briefly (user 2026-09-21).
PLATFORM_CLEAR_M = float(_os.getenv("MICRODUCK_EX_CLEAR", "0.07"))
# How far the walls carry on above the platform before the corridor ends:
# about one and a half duck heights (a duck is ~25 cm).
WALL_ABOVE_M = float(_os.getenv("MICRODUCK_EX_WALL_ABOVE", "0.375"))
DUCK_H_M = 0.25
PARKED_POS = (0.0, 0.0, -30.0)

# the platform is part of the structure, so it wears the walls' colour
_RGBA = "0.58 0.53 0.48 1"


def platform_near_y() -> float:
    """Where the platform starts: clear of the slot, not flush with it."""
    return 0.5 * cs.WALL_DEPTH_M + PLATFORM_CLEAR_M


def wall_end_y() -> float:
    """Where the overhanging walls stop and the open platform begins."""
    return platform_near_y() + OVERHANG_M


def platform_far_y() -> float:
    """The platform runs past the wall end, into open air."""
    return wall_end_y() + OPEN_M


def platform_centre_y() -> float:
    return 0.5 * (platform_near_y() + platform_far_y())


def clear_line_y() -> float:
    """Where "out on the open platform" starts counting: the middle of the open
    region, not the wall end.  A line at the wall end is crossed by a
    centimetre and no further, because crossing it ends the episode.
    """
    return wall_end_y() + 0.5 * OPEN_M


def goal_y() -> float:
    """Where the robot should end up standing: out in the OPEN part of the
    platform, past the end of the walls, so it has genuinely walked clear
    rather than stopped while still braced between them."""
    return wall_end_y() + 0.5 * OPEN_M


def _platform_spec() -> mujoco.MjSpec:
    xml = f"""
<mujoco model="{PLATFORM}">
  <worldbody>
    <body name="{PLATFORM}" mocap="true">
      <geom name="{PLATFORM}_collision" type="box"
            size="{0.5 * PLATFORM_WIDTH_M:.4f} {0.5 * (OVERHANG_M + OPEN_M):.4f} {0.5 * PLATFORM_THICK_M:.4f}"
            rgba="{_RGBA}" friction="1.0 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_exit_stage_entity_cfgs() -> dict[str, EntityCfg]:
    parked = EntityCfg.InitialStateCfg(pos=PARKED_POS)
    return {PLATFORM: EntityCfg(spec_fn=_platform_spec, init_state=parked)}


def platform_mocap_pos() -> tuple[float, float, float]:
    """Centre of the platform box, so its TOP lands on PLATFORM_TOP_M."""
    return (0.0, platform_centre_y(), PLATFORM_TOP_M - 0.5 * PLATFORM_THICK_M)


def wall_depth_with_overhang() -> float:
    """Depth of the UPPER wall segment: the slot, plus the reach out over the
    platform.  Only the part above the platform overhangs - below it the walls
    are the plain climbing slot, so nothing juts into the climb."""
    return (wall_end_y() - (-0.5 * cs.WALL_DEPTH_M))


def wall_centre_y() -> float:
    """Centre of the UPPER wall segment."""
    return 0.5 * (wall_end_y() + (-0.5 * cs.WALL_DEPTH_M))


def wall_top_z(platform_top: float | None = None) -> float:
    """The corridor ends about a duck and a half above the platform."""
    top = PLATFORM_TOP_M if platform_top is None else platform_top
    return top + WALL_ABOVE_M


def _stepped_wall_spec(name: str, rgba: str) -> mujoco.MjSpec:
    """A wall in two parts: the plain climbing slot below the platform, and an
    overhanging section above it that reaches out over the platform and stops a
    duck and a half higher, which is where the corridor ends.

    The body sits on the FLOOR so the two segments can be positioned by height.
    """
    t = cs.WALL_THICKNESS_M
    low_d = cs.WALL_DEPTH_M
    up_d = wall_depth_with_overhang()
    up_h = wall_top_z() - PLATFORM_TOP_M
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}_collision" type="box"
            size="{0.5 * t:.4f} {0.5 * low_d:.4f} {0.5 * PLATFORM_TOP_M:.4f}"
            pos="0 0 {0.5 * PLATFORM_TOP_M:.4f}"
            rgba="{rgba}" friction="0.9 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
      <geom name="{name}_upper" type="box"
            size="{0.5 * t:.4f} {0.5 * up_d:.4f} {0.5 * up_h:.4f}"
            pos="0 {wall_centre_y():.4f} {PLATFORM_TOP_M + 0.5 * up_h:.4f}"
            rgba="{rgba}" friction="0.9 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def _overhang_wall_spec(name: str, rgba: str) -> mujoco.MjSpec:
    """A climbing wall extended out over the platform.

    ``corridor_stage`` reads its depth from the environment at IMPORT time, so
    setting a variable inside the exit cfg would not change the walls it builds
    - they would stay 30 cm deep and would not overhang the platform at all,
    while every helper here said they did.  Build them explicitly instead.
    """
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}_collision" type="box"
            size="{0.5 * cs.WALL_THICKNESS_M:.4f} {0.5 * wall_depth_with_overhang():.4f} {0.5 * cs.WALL_HEIGHT_M:.4f}"
            rgba="{rgba}" friction="0.9 0.005 0.0001" condim="3" priority="1"
            solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def make_overhang_corridor_entity_cfgs() -> dict[str, EntityCfg]:
    parked = EntityCfg.InitialStateCfg(pos=cs.PARKED_POS)
    return {
        cs.WALL_LEFT: EntityCfg(spec_fn=lambda: _stepped_wall_spec(cs.WALL_LEFT, "0.60 0.55 0.50 1"),
                                init_state=parked),
        cs.WALL_RIGHT: EntityCfg(spec_fn=lambda: _stepped_wall_spec(cs.WALL_RIGHT, "0.54 0.50 0.46 1"),
                                 init_state=parked),
    }


def stepped_wall_mocap_z() -> float:
    """The stepped wall's body sits on the floor, not at its own mid-height."""
    return 0.0

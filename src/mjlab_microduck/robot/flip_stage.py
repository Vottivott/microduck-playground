"""Stage for the platform flip: a raised platform (mocap, per-env height) and
the padded crash mat from the run-into-mat video lying flat in front of its
edge (user request 2026-09-17: same 3D asset, softness modelled).

Env frame: the platform edge is the line x = 0; the platform occupies
x < 0, the mat x > MAT_GAP.  "Drop height" H is the platform top above the
MAT TOP (the mat is 0.443 m thick).  The robot spawns on the platform near
the edge and lands on the mat.  Heights are set at reset through the mocap
pose.  The mat collides as a hidden soft box (solref 0.03 s: a few cm of
give under the landing load); its visual mesh (Meshy "Blue Padded Barrier",
decimated to 150k faces with vertex colours) is only added when
MICRODUCK_FLIP_MAT_VISUAL=1 (renders), not in the 4096-env training scene.
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

_ASSETS = Path(__file__).with_name("assets") / "crash_mat"

PLATFORM_ENTITY = "platform"
MAT_ENTITY = "mat"

PLATFORM_DEPTH_M = 0.60      # along x (edge at x = 0, slab extends to x = -0.6)
PLATFORM_WIDTH_M = 0.80
PLATFORM_THICKNESS_M = 0.06
MAT_GAP_M = 0.02             # mat starts this far past the edge
MAT_LENGTH_M = 1.90154       # measured mesh extents (x long, y width, z thick)
MAT_WIDTH_M = 0.60500
MAT_THICKNESS_M = 0.44342
MAT_SOLREF_S = 0.03          # soft: a few cm of give under the landing load
PILLAR_HEIGHT_M = 1.6        # visual support under the slab (pokes below the floor when low)
PARKED_POS = (0.0, 0.0, -3.0)


def _box_spec(name: str, half: tuple[float, float, float], rgba: str, friction: float,
              solref: tuple[float, float], solimp: tuple[float, ...]) -> mujoco.MjSpec:
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}_collision" type="box" size="{half[0]:.5f} {half[1]:.5f} {half[2]:.5f}"
            rgba="{rgba}" friction="{friction:.3f} 0.005 0.0001" condim="3" priority="1"
            solref="{solref[0]:.4f} {solref[1]:.2f}" solimp="{' '.join(f'{v:g}' for v in solimp)}"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def platform_spec() -> mujoco.MjSpec:
    spec = _box_spec(PLATFORM_ENTITY, (0.5 * PLATFORM_DEPTH_M, 0.5 * PLATFORM_WIDTH_M, 0.5 * PLATFORM_THICKNESS_M),
                     "0.55 0.45 0.35 1", 1.0, (0.01, 1.0), (0.95, 0.99, 0.001, 0.5, 2.0))
    body = spec.bodies[1]
    pillar = body.add_geom(name=f"{PLATFORM_ENTITY}_pillar", type=mujoco.mjtGeom.mjGEOM_BOX)
    pillar.size = [0.5 * PLATFORM_DEPTH_M - 0.05, 0.5 * PLATFORM_WIDTH_M - 0.1, 0.5 * PILLAR_HEIGHT_M]
    pillar.pos = [0.0, 0.0, -0.5 * PLATFORM_THICKNESS_M - 0.5 * PILLAR_HEIGHT_M]
    pillar.rgba = [0.45, 0.37, 0.30, 1.0]
    pillar.contype = 0
    pillar.conaffinity = 0
    pillar.group = 2
    return spec


def mat_spec() -> mujoco.MjSpec:
    spec = _box_spec(MAT_ENTITY, (0.5 * MAT_LENGTH_M, 0.5 * MAT_WIDTH_M, 0.5 * MAT_THICKNESS_M),
                     "0.06 0.27 0.63 1", 0.9, (MAT_SOLREF_S, 1.0), (0.9, 0.95, 0.01, 0.5, 2.0))
    # Prefer the full-resolution mesh (3.1M faces, gitignored, copied to render
    # hosts): the 150k-face decimation tore holes in the padded surface
    # (user 2026-09-17).  Falls back to the decimated file.
    full = _ASSETS / "crash_mat_flat_full.obj"
    low = _ASSETS / "crash_mat_flat.obj"
    mesh_file = full if full.exists() else low
    if os.getenv("MICRODUCK_FLIP_MAT_VISUAL", "0") == "1" and mesh_file.exists():
        mesh = spec.add_mesh()
        mesh.name = "crash_mat_mesh"
        mesh.file = str(mesh_file)
        body = spec.bodies[1]
        for g in body.geoms:
            g.rgba = [0.0, 0.0, 0.0, 0.0]          # hide the collision box behind the mesh
        v = body.add_geom(name=f"{MAT_ENTITY}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="crash_mat_mesh")
        v.contype = 0
        v.conaffinity = 0
        v.group = 2
        v.rgba = [0.08, 0.20, 0.46, 1.0]        # MuJoCo ignores OBJ vertex colours; the run video's blue
    return spec


def make_flip_stage_entity_cfgs() -> dict[str, EntityCfg]:
    parked = EntityCfg.InitialStateCfg(pos=PARKED_POS)
    return {
        PLATFORM_ENTITY: EntityCfg(spec_fn=platform_spec, init_state=parked),
        MAT_ENTITY: EntityCfg(spec_fn=mat_spec, init_state=parked),
    }


def platform_center(drop_height: float) -> tuple[float, float, float]:
    """Mocap position of the platform box for a drop height above the MAT TOP."""
    return (-0.5 * PLATFORM_DEPTH_M, 0.0, MAT_THICKNESS_M + drop_height - 0.5 * PLATFORM_THICKNESS_M)


def mat_center() -> tuple[float, float, float]:
    return (MAT_GAP_M + 0.5 * MAT_LENGTH_M, 0.0, 0.5 * MAT_THICKNESS_M)

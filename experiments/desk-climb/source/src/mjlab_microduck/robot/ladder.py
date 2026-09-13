"""Parametric alternating-tread stair-ladder for Microduck.

Design rationale (2026-09-03 rewrite).  The previous ladder used 6 mm round
rungs 45 mm apart: a flat sole on a round rung is a line contact with no pitch
authority, and the 45 mm spacing was forced by the 38 mm tall ankle shell
having to fit under the next rung.  Neither a hold nor a step had a stable
basin, so RL was asked to fix physics and never did.

This ladder is built for the morphology instead:

* **Flat treads** (default 36 mm deep, 4 mm thick) give the ~48 mm sole an
  area contact, so every tread is a small piece of floor.
* **Alternating half-width treads**: even treads are on the left, odd on the
  right.  Same-side treads are two risers apart, so the riser can be 15–30 mm
  (a stair-like step for a 105 mm leg) while the foot and ankle still fit
  under the next same-side tread.  The body rises one riser per step.
* **Riser and angle are per-environment** (each tread is its own mocap
  entity, positioned at reset), so difficulty can be curriculum-ramped and
  randomized for sim2real without recompiling.

Only the tread box dimensions are shared by all environments.  The geometry
must satisfy two clearances for a foot standing on tread ``i``:

* the toe (22 mm tall) slides under tread ``i+2`` whose underside is
  ``2·riser − thickness`` above the sole → ``riser ≥ 14.5 mm``;
* the ankle shell (38 mm tall, 22 mm forward of the ankle axis) either fits
  under tread ``i+2`` (``2·riser − thickness ≥ 41 mm``) or stays behind its
  rear edge → ``depth ≤ 2·riser / tan(angle) + 14 mm``.

The spawn puts the ankle axis at the tread centre, so the centre of mass has
half the tread depth (18 mm) of static margin both ways.

``validate_tread_clearance`` encodes both; the reset event clamps sampled
geometry to them.

Environment overrides (process start):

* ``MICRODUCK_LADDER_TREAD_DEPTH_MM`` (default 36)
* ``MICRODUCK_LADDER_TREAD_THICKNESS_MM`` (default 4)
* ``MICRODUCK_LADDER_SIDE_WIDTH_MM`` (default 110, per-side tread width)
* ``MICRODUCK_LADDER_NUM_TREADS`` (default 16)
* ``MICRODUCK_LADDER_ALTERNATING`` (default 1)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import mujoco
import torch

from mjlab.entity import EntityCfg

# Microduck sole/ankle dimensions measured in HOME on the all-collisions
# model (see tests/test_ladder_cfg.py, which re-measures them).
SOLE_TOE_AHEAD_OF_SITE_M = 0.031
SOLE_HEEL_BEHIND_SITE_M = 0.017
SOLE_BOTTOM_ABOVE_SITE_M = 0.0028
TOE_HEIGHT_M = 0.022
ANKLE_SHELL_HEIGHT_M = 0.038
ANKLE_SHELL_AHEAD_OF_SITE_M = 0.022
FOOT_LATERAL_OFFSET_M = 0.042
HOME_TRUNK_ABOVE_SITE_M = 0.117

TREAD_ENTITY_PREFIX = "tread_"
# Contact stiffness of treads and rails.  MuJoCo's default 20 ms time constant
# let the 0.8 kg robot sink ~2 mm into a 4 mm tread and pass up to 2 mm into
# neighbouring treads (measured 2026-09-03 with scripts/probe_foot_penetration.py).
# 10 ms is the stiffest stable value at the 5 ms physics step (>= 2 dt);
# priority=1 makes these parameters win over the foot's defaults.
TREAD_SOLREF_TIMECONST_S = float(os.getenv("MICRODUCK_TREAD_SOLREF_S", "0.010"))
RAIL_ENTITY_NAMES = ("rail_left", "rail_right")


def rail_entity_names(geometry: "StairLadderGeometry") -> tuple[str, ...]:
    """Two rails per flight: rail_left, rail_right, rail_left_1, rail_right_1, ..."""
    names = list(RAIL_ENTITY_NAMES)
    for f in range(1, geometry.num_flights):
        names += [f"rail_left_{f}", f"rail_right_{f}"]
    return tuple(names)
# Treads and rails are parked here until the reset event positions them.
PARKED_POS = (0.0, 0.0, -2.0)

_LEG_TABLE_PATH = Path(__file__).with_name("ladder_leg_table.json")


@dataclass(frozen=True)
class StairLadderGeometry:
    """Fixed (compile-time) dimensions of the stair-ladder."""

    num_treads: int = 16
    alternating: bool = True
    tread_depth_m: float = 0.036
    tread_thickness_m: float = 0.004
    side_width_m: float = 0.110
    center_gap_m: float = 0.010
    rail_width_m: float = 0.014
    rail_depth_m: float = 0.018
    rail_length_m: float = 0.80
    tread_friction: float = 1.0
    # Staircase mode: every ``landing_every``-th tread (1-based count) is a
    # full-width landing box standing in for a human tread.  Its top sits one
    # riser above the previous mini tread (like tread 8 of a segment), its
    # box extends ``landing_depth_m`` *forward* of that nose, and the next
    # flight of mini treads starts ``flat`` metres past the nose (per env).
    # 0 = plain ladder.
    landing_every: int = 0
    landing_depth_m: float = 0.40
    landing_width_m: float = 0.60
    landing_thickness_m: float = 0.02
    # Foot target on a landing: at its nose (sole 31 mm on, 17 mm over, like a
    # mini tread).  A 45 mm target made the last step a 9 cm reach.
    landing_target_ahead_m: float = 0.0
    # The mini segment is set back from the riser face: a foot on the seventh
    # mini tread has its toe 31 mm ahead of the ankle, i.e. exactly at the
    # face when the segment ends one run short of it, and the swing onto that
    # tread is blocked by the riser.  The landing nose (riser face) is this
    # much further than a regular tread would be.  12 mm cleared the toe but
    # the ankle shell and thigh of a foot on the top mini tread still leaned
    # on the riser face (measured 2026-09-03); 30 mm clears them.
    landing_setback_m: float = 0.030
    # Foot target height above a landing's top.  The policy reaches for a
    # target on a straight line; with the landing 74 mm ahead and 46 mm up
    # that line meets the riser face 35 mm below the top (s4-s8 stall).  A
    # target 5 cm above the surface makes the line clear the nose.
    landing_target_up_m: float = 0.05
    # Aim each foot at its own side of the landing (a regular tread's lateral
    # position) rather than the centreline: with the centreline target both
    # feet were asked for a converging step and hesitated two treads below
    # (landing study 2026-09-04).
    landing_target_per_side: bool = True
    # The landing step (a foot targeting a landing tread) is only offered
    # once the other foot stands on the tread right below the landing:
    # from two treads below the reach is too long and the toe catches the
    # landing's nose (s18 dive trace, 2026-09-04).
    landing_step_gate: bool = False
    # Fixed staircase (demo): explicit flight frames relative to flight 0
    # (offsets (F, 2) in flight-0 coordinates, yaws (F,) relative to flight 0)
    # and per-landing convex polygons ((u, v) in the landing's flight frame,
    # relative to the riser-face centre at the landing nose) extruded
    # ``landing_thickness_m`` downwards from the landing top.
    frame_offsets: tuple[tuple[float, float], ...] | None = None
    frame_yaws: tuple[float, ...] | None = None
    landing_polygons: tuple[tuple[tuple[float, float], ...], ...] | None = None

    def is_landing(self, index: int) -> bool:
        return self.landing_every > 0 and (index + 1) % self.landing_every == 0

    def flight_of(self, index: int) -> int:
        return index // self.landing_every if self.landing_every > 0 else 0

    @property
    def num_flights(self) -> int:
        if self.landing_every <= 0:
            return 1
        return (self.num_treads + self.landing_every - 1) // self.landing_every

    def validate(self) -> None:
        if self.num_treads < 6:
            raise ValueError("stair ladder needs at least six treads")
        if not 0.015 <= self.tread_depth_m <= 0.060:
            raise ValueError("tread depth must be within [15, 60] mm")
        if not 0.002 <= self.tread_thickness_m <= 0.010:
            raise ValueError("tread thickness must be within [2, 10] mm")
        if self.side_width_m < 2.0 * FOOT_LATERAL_OFFSET_M:
            raise ValueError("side width must cover the foot lateral offset")
        if self.landing_every < 0 or self.landing_every == 1:
            raise ValueError("landing_every must be 0 (none) or >= 2")
        if self.landing_every and self.landing_width_m < self.clear_width_m:
            raise ValueError("landing must be at least as wide as the tread pair")

    @property
    def tread_half_width_m(self) -> float:
        if self.alternating:
            return 0.5 * self.side_width_m
        return self.side_width_m + 0.5 * self.center_gap_m

    @property
    def clear_width_m(self) -> float:
        return 2.0 * self.side_width_m + self.center_gap_m

    def tread_side(self, index: int) -> int:
        """+1 = left (positive y), -1 = right; 0 for full-width treads/landings."""
        if not self.alternating or self.is_landing(index):
            return 0
        return 1 if index % 2 == 0 else -1

    def tread_center_y(self, index: int) -> float:
        side = self.tread_side(index)
        return side * (0.5 * self.center_gap_m + 0.5 * self.side_width_m)

    def same_side_spacing(self) -> int:
        return 2 if self.alternating else 1

    def min_riser_m(self, margin_m: float = 0.003) -> float:
        """Riser below which the toe cannot fit under the next same-side tread."""
        return (TOE_HEIGHT_M + margin_m + self.tread_thickness_m) / self.same_side_spacing()

    def ankle_clears_vertically(self, riser_m: float, margin_m: float = 0.003) -> bool:
        """True when the next same-side tread is above the ankle shell entirely."""
        gap = self.same_side_spacing() * riser_m - self.tread_thickness_m
        return gap >= ANKLE_SHELL_HEIGHT_M + margin_m

    def max_depth_m(self, riser_m: float, angle_deg: float, toe_margin_m: float = 0.005) -> float:
        """Deepest tread for which the ankle shell clears the tread above.

        If the next same-side tread is higher than the ankle shell there is
        no constraint.  Otherwise, with the toe ``toe_margin_m`` short of the
        nose, the ankle axis sits ``SOLE_TOE_AHEAD_OF_SITE_M + toe_margin_m``
        behind the nose and the shell reaches ``ANKLE_SHELL_AHEAD_OF_SITE_M``
        further forward, so it must stay behind that tread's rear edge.
        """
        if self.ankle_clears_vertically(riser_m):
            return math.inf
        run = self.same_side_spacing() * riser_m / math.tan(math.radians(angle_deg))
        ankle_allowance = SOLE_TOE_AHEAD_OF_SITE_M + toe_margin_m - ANKLE_SHELL_AHEAD_OF_SITE_M
        return run + ankle_allowance


def validate_tread_clearance(
    geometry: StairLadderGeometry, riser_m: float, angle_deg: float
) -> None:
    if riser_m < geometry.min_riser_m():
        raise ValueError(
            f"riser {riser_m * 1000:.1f} mm is below the toe clearance minimum "
            f"{geometry.min_riser_m() * 1000:.1f} mm"
        )
    max_depth = geometry.max_depth_m(riser_m, angle_deg)
    if geometry.tread_depth_m > max_depth:
        raise ValueError(
            f"tread depth {geometry.tread_depth_m * 1000:.1f} mm exceeds the ankle "
            f"clearance limit {max_depth * 1000:.1f} mm at riser "
            f"{riser_m * 1000:.1f} mm / {angle_deg:.0f} deg"
        )


def stair_ladder_geometry_from_env() -> StairLadderGeometry:
    def mm(name: str, default: float) -> float:
        return float(os.getenv(name, str(default))) / 1000.0

    g = StairLadderGeometry(
        num_treads=int(os.getenv("MICRODUCK_LADDER_NUM_TREADS", "16")),
        alternating=os.getenv("MICRODUCK_LADDER_ALTERNATING", "1") == "1",
        tread_depth_m=mm("MICRODUCK_LADDER_TREAD_DEPTH_MM", 36.0),
        tread_thickness_m=mm("MICRODUCK_LADDER_TREAD_THICKNESS_MM", 4.0),
        side_width_m=mm("MICRODUCK_LADDER_SIDE_WIDTH_MM", 110.0),
    )
    g.validate()
    return g


def _box_spec(name: str, half_size: tuple[float, float, float], rgba: str, friction: float) -> mujoco.MjSpec:
    xml = f"""
<mujoco model="{name}">
  <worldbody>
    <body name="{name}" mocap="true">
      <geom name="{name}" type="box" size="{half_size[0]:.6f} {half_size[1]:.6f} {half_size[2]:.6f}"
            rgba="{rgba}" friction="{friction:.3f} 0.005 0.0001" condim="3" priority="1"
            solref="{TREAD_SOLREF_TIMECONST_S:.4f} 1" solimp="0.95 0.99 0.001 0.5 2"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjSpec.from_string(xml)


def _prism_spec(name: str, polygon_uv, thickness: float, rgba: str, friction: float) -> mujoco.MjSpec:
    """Mocap body with a convex prism (polygon extruded from z=-thickness to 0)."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    verts = []
    for u, v in polygon_uv:
        verts += [float(u), float(v), -float(thickness)]
        verts += [float(u), float(v), 0.0]
    mesh = spec.add_mesh(name=f"{name}_mesh")
    mesh.uservert = verts
    body = spec.worldbody.add_body(name=name, mocap=True)
    geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh")
    geom.rgba = [float(c) for c in rgba.split()]
    geom.friction = [friction, 0.005, 0.0001]
    geom.condim = 3
    geom.priority = 1
    geom.solref = [TREAD_SOLREF_TIMECONST_S, 1.0]
    geom.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    return spec


def make_tread_spec(geometry: StairLadderGeometry, index: int) -> mujoco.MjSpec:
    if geometry.is_landing(index) and geometry.landing_polygons is not None:
        flight = geometry.flight_of(index)
        return _prism_spec(
            "tread",
            geometry.landing_polygons[flight],
            geometry.landing_thickness_m,
            "0.62 0.48 0.30 1",
            geometry.tread_friction,
        )
    if geometry.is_landing(index):
        return _box_spec(
            "tread",
            (0.5 * geometry.landing_depth_m, 0.5 * geometry.landing_width_m, 0.5 * geometry.landing_thickness_m),
            "0.62 0.48 0.30 1",
            geometry.tread_friction,
        )
    rgba = "0.80 0.62 0.32 1" if geometry.tread_side(index) >= 0 else "0.70 0.52 0.26 1"
    return _box_spec(
        "tread",
        (0.5 * geometry.tread_depth_m, geometry.tread_half_width_m, 0.5 * geometry.tread_thickness_m),
        rgba,
        geometry.tread_friction,
    )


def make_rail_spec(geometry: StairLadderGeometry) -> mujoco.MjSpec:
    return _box_spec(
        "rail",
        (0.5 * geometry.rail_depth_m, 0.5 * geometry.rail_width_m, 0.5 * geometry.rail_length_m),
        "0.52 0.32 0.14 1",
        0.9,
    )


def make_stair_ladder_entity_cfgs(geometry: StairLadderGeometry) -> dict[str, EntityCfg]:
    """One fixed mocap entity per tread and rail, parked until reset."""
    geometry.validate()
    parked = EntityCfg.InitialStateCfg(pos=PARKED_POS)
    cfgs: dict[str, EntityCfg] = {}
    for index in range(geometry.num_treads):
        # Bind the index in a default argument so each spec_fn is independent.
        def tread_spec_fn(index: int = index) -> mujoco.MjSpec:
            return make_tread_spec(geometry, index)

        cfgs[f"{TREAD_ENTITY_PREFIX}{index:02d}"] = EntityCfg(
            spec_fn=tread_spec_fn, init_state=parked
        )
    for name in rail_entity_names(geometry):
        cfgs[name] = EntityCfg(
            spec_fn=lambda: make_rail_spec(geometry), init_state=parked
        )
    if geometry.landing_polygons is not None:
        # Corner staircase: the upper floor beyond the last landing (c2 probe
        # 2026-09-06: 72 % of climbs reached the last tread and toppled off the
        # summit because nothing continued the surface).  Placed at the last
        # tread's pose at reset, in that tread's frame.
        cfgs[TOP_FLOOR_ENTITY] = EntityCfg(
            spec_fn=lambda: _prism_spec(
                TOP_FLOOR_ENTITY, top_floor_polygon(geometry), geometry.landing_thickness_m, "0.60 0.46 0.29 1", geometry.tread_friction
            ),
            init_state=parked,
        )
    return cfgs


TOP_FLOOR_ENTITY = "top_floor"
TOP_FLOOR_DEPTH_M = 1.5
TOP_FLOOR_HALF_WIDTH_M = 0.9


def top_floor_polygon(geometry: StairLadderGeometry) -> tuple[tuple[float, float], ...]:
    """Rectangle continuing the last landing's surface, in the last tread's frame."""
    poly = geometry.landing_polygons[-1]
    u_far = max(u for u, _ in poly)
    v_mid = 0.5 * (min(v for _, v in poly) + max(v for _, v in poly))
    return (
        (u_far, v_mid - TOP_FLOOR_HALF_WIDTH_M),
        (u_far + TOP_FLOOR_DEPTH_M, v_mid - TOP_FLOOR_HALF_WIDTH_M),
        (u_far + TOP_FLOOR_DEPTH_M, v_mid + TOP_FLOOR_HALF_WIDTH_M),
        (u_far, v_mid + TOP_FLOOR_HALF_WIDTH_M),
    )


def tread_top_heights(riser: torch.Tensor, num_treads: int) -> torch.Tensor:
    """(N, T) top-surface heights above the floor; tread 0 is one riser up."""
    index = torch.arange(1, num_treads + 1, device=riser.device, dtype=riser.dtype)
    return riser[:, None] * index[None, :]


def nose_x(x0: torch.Tensor, z: torch.Tensor, angle_rad: torch.Tensor) -> torch.Tensor:
    """x of the ladder nose line at height ``z`` (broadcasts over trailing dims)."""
    while angle_rad.dim() < z.dim():
        angle_rad = angle_rad.unsqueeze(-1)
        x0 = x0.unsqueeze(-1)
    return x0 + z / torch.tan(angle_rad)


def flight_frames(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    flat: torch.Tensor | None = None,
    lateral: torch.Tensor | None = None,
    dyaw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per flight f: base point (N, F, 2), yaw (N, F) and base height (N, F).

    Flight 0 starts at (x0, y0) with yaw 0 (its nose line runs along +x).
    Flight f starts on the previous landing: from that landing's nose point
    move ``flat`` along the previous flight's axis and ``lateral`` to its
    left, and turn by ``dyaw`` (staircase winders turn the next segment).
    ``flat``/``lateral``/``dyaw`` are (N,) applied to every flight change or
    (N, F-1) per change.
    """
    n, dev, dt = riser.shape[0], riser.device, riser.dtype
    F = geometry.num_flights
    base = torch.zeros(n, F, 2, device=dev, dtype=dt)
    yaw = torch.zeros(n, F, device=dev, dtype=dt)
    z_base = torch.zeros(n, F, device=dev, dtype=dt)
    base[:, 0, 0] = x0
    base[:, 0, 1] = y0
    if geometry.landing_every <= 0 or F == 1:
        return base, yaw, z_base
    if geometry.frame_offsets is not None:
        off = torch.tensor(geometry.frame_offsets, device=dev, dtype=dt)[None, :, :]
        base = torch.stack((x0, y0), dim=-1)[:, None, :] + off
        yaw = torch.tensor(geometry.frame_yaws, device=dev, dtype=dt)[None, :].expand(n, F).clone()
        L = float(geometry.landing_every)
        z_base = torch.arange(F, device=dev, dtype=dt)[None, :] * (L * riser)[:, None]
        return base, yaw, z_base

    def per_change(v: torch.Tensor | None) -> torch.Tensor:
        if v is None:
            return torch.zeros(n, F - 1, device=dev, dtype=dt)
        if v.dim() == 1:
            return v[:, None].expand(n, F - 1)
        return v

    flat_c, lat_c, dyaw_c = per_change(flat), per_change(lateral), per_change(dyaw)
    L = float(geometry.landing_every)
    # nose advance over one flight, to the riser face (landing nose)
    incline_run = L * riser / torch.tan(angle_rad) + geometry.landing_setback_m
    for f in range(1, F):
        c, sn = torch.cos(yaw[:, f - 1]), torch.sin(yaw[:, f - 1])
        # previous landing nose point = previous base advanced along its axis
        nose = base[:, f - 1] + torch.stack((incline_run * c, incline_run * sn), dim=-1)
        u, v = flat_c[:, f - 1], lat_c[:, f - 1]
        base[:, f] = nose + torch.stack((u * c - v * sn, u * sn + v * c), dim=-1)
        yaw[:, f] = yaw[:, f - 1] + dyaw_c[:, f - 1]
        z_base[:, f] = z_base[:, f - 1] + L * riser
    return base, yaw, z_base


def flight_offsets(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    flat: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-flight convenience: (N, F) x offset of each flight base from
    ``x0`` and (N, F) base height (no lateral shift, no turn)."""
    base, _yaw, z_base = flight_frames(
        geometry, riser, angle_rad, torch.zeros_like(riser), torch.zeros_like(riser), flat
    )
    return base[:, :, 0], z_base


def tread_poses(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    flat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-environment tread box centres (N, T, 3) and top heights (N, T).

    Landings (staircase mode) are centred ``landing_depth/2`` *ahead* of their
    nose; every other tread ``tread_depth/2`` behind it.
    """
    centres, top, _nose, _target, _yaw = tread_layout(geometry, riser, angle_rad, x0, y0, flat)
    return centres, top


def tread_layout(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    flat: torch.Tensor | None = None,
    lateral: torch.Tensor | None = None,
    dyaw: torch.Tensor | None = None,
    nose_jitter: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(centres (N,T,3), tops (N,T), nose along-axis u (N,T), foot-target
    xy (N,T,2), tread yaw (N,T)).  Each tread lies in its flight's frame:
    ``u`` along the flight axis from its base, ``v`` to the left.
    ``nose_jitter`` (N,T) shifts each tread's nose along the axis (irregular
    spacing, so the policy learns to read the per-foot target distance)."""
    T = geometry.num_treads
    top = tread_top_heights(riser, T)
    base, yaw, base_z = flight_frames(geometry, riser, angle_rad, x0, y0, flat, lateral, dyaw)
    flight = torch.tensor([geometry.flight_of(i) for i in range(T)], device=riser.device)
    z_base = base_z[:, flight]
    u_nose = (top - z_base) / torch.tan(angle_rad)[:, None]
    if nose_jitter is not None:
        u_nose = u_nose + nose_jitter
    landing = torch.tensor([geometry.is_landing(i) for i in range(T)], device=riser.device)
    u_nose = torch.where(landing, u_nose + geometry.landing_setback_m, u_nose)
    mesh_landing = geometry.landing_polygons is not None
    depth_back = torch.where(
        landing,
        torch.full_like(u_nose, 0.0 if mesh_landing else -0.5 * geometry.landing_depth_m),
        torch.full_like(u_nose, 0.5 * geometry.tread_depth_m),
    )
    u_centre = u_nose - depth_back
    v_centre = torch.tensor(
        [geometry.tread_center_y(i) for i in range(T)], device=riser.device, dtype=riser.dtype
    )[None, :].expand(riser.shape[0], T)
    thickness = torch.where(
        landing,
        torch.full_like(u_nose, 0.0 if mesh_landing else geometry.landing_thickness_m),
        torch.full_like(u_nose, geometry.tread_thickness_m),
    )
    centre_z = top - 0.5 * thickness  # mesh landings: body origin at the top surface
    u_target = torch.where(landing, u_nose + geometry.landing_target_ahead_m, u_centre)
    tread_yaw = yaw[:, flight]
    b = base[:, flight]  # (N, T, 2)
    c, sn = torch.cos(tread_yaw), torch.sin(tread_yaw)
    centre_x = b[:, :, 0] + u_centre * c - v_centre * sn
    centre_y = b[:, :, 1] + u_centre * sn + v_centre * c
    target_x = b[:, :, 0] + u_target * c - v_centre * sn
    target_y = b[:, :, 1] + u_target * sn + v_centre * c
    centres = torch.stack((centre_x, centre_y, centre_z), dim=-1)
    target_xy = torch.stack((target_x, target_y), dim=-1)
    return centres, top, u_nose, target_xy, tread_yaw


WALK_PATH_POINTS = 16


def walk_paths(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    base: torch.Tensor,
    yaw: torch.Tensor,
    num_points: int = WALK_PATH_POINTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walking path across each landing: a cubic Hermite curve from the
    landing nose of flight f-1 (tangent along that flight) to the base of
    flight f (tangent along the next flight), sampled at ``num_points``.

    Returns points (N, F-1, K, 2), tangent yaw (N, F-1, K) and the remaining
    path length from each point to the end (N, F-1, K).  ``base``/``yaw`` are
    the flight frames from :func:`flight_frames` (any origin).
    """
    n, F = yaw.shape
    dev, dt = base.device, base.dtype
    P = max(F - 1, 1)
    K = num_points
    if F <= 1:
        z = torch.zeros(n, P, K, device=dev, dtype=dt)
        return torch.zeros(n, P, K, 2, device=dev, dtype=dt), z, z
    L = float(geometry.landing_every)
    incline_run = L * riser / torch.tan(angle_rad) + geometry.landing_setback_m  # (N,)
    d_prev = torch.stack((torch.cos(yaw[:, :-1]), torch.sin(yaw[:, :-1])), dim=-1)  # (N, P, 2)
    d_next = torch.stack((torch.cos(yaw[:, 1:]), torch.sin(yaw[:, 1:])), dim=-1)
    p0 = base[:, :-1] + incline_run[:, None, None] * d_prev
    p1 = base[:, 1:]
    chord = (p1 - p0).norm(dim=-1, keepdim=True).clamp_min(1e-3)  # (N, P, 1)
    m0, m1 = d_prev * chord, d_next * chord
    t = torch.linspace(0.0, 1.0, K, device=dev, dtype=dt)[None, None, :, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    pts = h00 * p0[:, :, None] + h10 * m0[:, :, None] + h01 * p1[:, :, None] + h11 * m1[:, :, None]
    dh00 = 6 * t**2 - 6 * t
    dh10 = 3 * t**2 - 4 * t + 1
    dh01 = -6 * t**2 + 6 * t
    dh11 = 3 * t**2 - 2 * t
    tang = dh00 * p0[:, :, None] + dh10 * m0[:, :, None] + dh01 * p1[:, :, None] + dh11 * m1[:, :, None]
    tyaw = torch.atan2(tang[..., 1], tang[..., 0])
    seg = (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1)  # (N, P, K-1)
    rem = torch.cat((seg.flip(-1).cumsum(-1).flip(-1), torch.zeros(n, P, 1, device=dev, dtype=dt)), dim=-1)
    return pts, tyaw, rem


def flight_at_height(geometry: StairLadderGeometry, riser: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Index (N,) of the flight whose height band contains ``z``."""
    if geometry.landing_every <= 0 or geometry.num_flights == 1:
        return torch.zeros_like(z, dtype=torch.long)
    L = float(geometry.landing_every)
    # A NaN state (guarded elsewhere by the nan_state termination) must not
    # turn into an out-of-range gather index: clamp does not remove NaN.
    band = torch.nan_to_num(z / (L * riser), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.floor(band).clamp(0, geometry.num_flights - 1).long()


def nose_line_x(
    geometry: StairLadderGeometry,
    riser: torch.Tensor,
    angle_rad: torch.Tensor,
    x0: torch.Tensor,
    flat: torch.Tensor | None,
    z: torch.Tensor,
) -> torch.Tensor:
    """x of the (piecewise, straight-flight) nose line at height ``z`` (N,).
    Equals :func:`nose_x` for a plain ladder."""
    if geometry.landing_every <= 0 or geometry.num_flights == 1:
        return nose_x(x0, z, angle_rad)
    off_x, base_z = flight_offsets(geometry, riser, angle_rad, flat)
    f = flight_at_height(geometry, riser, z)
    xb = off_x.gather(1, f[:, None]).squeeze(1)
    zb = base_z.gather(1, f[:, None]).squeeze(1)
    return x0 + xb + (z - zb) / torch.tan(angle_rad)


def to_flight_frame(
    point_xy: torch.Tensor, base_xy: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy (N, 2) -> (u along the flight axis, v to its left), (N,) each."""
    d = point_xy - base_xy
    c, sn = torch.cos(yaw), torch.sin(yaw)
    return d[:, 0] * c + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * c


def rail_poses(
    geometry: StairLadderGeometry,
    angle_rad: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    riser: torch.Tensor | None = None,
    flat: torch.Tensor | None = None,
    lateral: torch.Tensor | None = None,
    dyaw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rail centres (N, 2F, 3) and quaternions (N, 2F, 4), long axis along the
    incline; order matches :func:`rail_entity_names` (left, right per flight)."""
    half = 0.5 * geometry.rail_length_m
    rail_v = 0.5 * geometry.clear_width_m + 0.5 * geometry.rail_width_m
    if riser is None:
        riser = torch.zeros_like(x0)
    base, yaw, base_z = flight_frames(geometry, riser, angle_rad, x0, y0, flat, lateral, dyaw)
    slots, quats = [], []
    # Box local z onto the incline: yaw about z (flight heading) after a
    # pitch about y by (90° − angle).  Scalar-first quaternions.
    for f in range(geometry.num_flights):
        c, sn = torch.cos(yaw[:, f]), torch.sin(yaw[:, f])
        u = half * torch.cos(angle_rad)
        cz = base_z[:, f] + half * torch.sin(angle_rad)
        for v in (rail_v, -rail_v):
            slots.append(torch.stack((base[:, f, 0] + u * c - v * sn, base[:, f, 1] + u * sn + v * c, cz), dim=-1))
            quats.append(_quat_yaw_pitch(yaw[:, f], 0.5 * math.pi - angle_rad))
    return torch.stack(slots, dim=1), torch.stack(quats, dim=1)


def _quat_yaw_pitch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternion of R = Rz(yaw) · Ry(pitch)."""
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    return torch.stack((cy * cp, -sy * sp, cy * sp, sy * cp), dim=-1)


def quat_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Scalar-first quaternion of a rotation about z."""
    return torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=-1)


def load_leg_table() -> dict:
    """Staggered-stance leg offsets (see scripts/generate_ladder_leg_table.py)."""
    return json.loads(_LEG_TABLE_PATH.read_text())


def leg_offsets_for(table: dict, delta_m: float, forward_m: float) -> tuple[float, float, float]:
    """Nearest-entry lookup of (hip_pitch, knee, ankle) offsets for the left leg."""
    deltas = table["delta_mm"]
    forwards = table["forward_mm"]
    d = min(deltas, key=lambda v: abs(v - delta_m * 1000.0))
    f = min(forwards, key=lambda v: abs(v - forward_m * 1000.0))
    entry = table["table"][f"{d}:{f}"]
    return entry["hip_pitch"], entry["knee"], entry["ankle"]


LADDER_GEOMETRY = stair_ladder_geometry_from_env()


# --- fixed staircase from the presentation export ------------------------------

_STAIRCASE_JSON = Path(__file__).with_name("staircase_corner.json")


def staircase_geometry_from_json(
    side: str = "right", path: Path = _STAIRCASE_JSON, angle_deg: float | None = None
) -> tuple[StairLadderGeometry, dict]:
    """Build the fixed corner-staircase geometry: one flight of seven mini
    treads per human step (the eighth coincides with the human tread, which
    is the landing mesh), frames from the exported placement.  Returns the
    geometry and a dict with ``riser`` (m) and ``angle_deg``.

    Flight 0 is rotated to yaw 0 (its axis is +x); all offsets are relative to
    flight 0's base.  Landing polygons are the human tread top faces in each
    flight's frame relative to the riser-face centre of that step.
    """
    data = json.loads(Path(path).read_text())
    steps = data["steps"]
    L = int(data["mini_risers_per_step"])
    rise = float(data["rise"])
    angle = float(angle_deg if angle_deg is not None else data["mini_angle_deg"])
    yaw0 = math.radians(steps[0]["yaw_deg"])
    base0 = steps[0]["mini"][side]["base"]
    c0, s0 = math.cos(-yaw0), math.sin(-yaw0)

    def to0(xy):
        dx, dy = xy[0] - base0[0], xy[1] - base0[1]
        return (dx * c0 - dy * s0, dx * s0 + dy * c0)

    offsets, yaws, polys = [], [], []
    for st in steps:
        m = st["mini"][side]
        offsets.append(to0(m["base"]))
        yaw = math.radians(st["yaw_deg"])
        yaws.append(yaw - yaw0)
        cx, cy = m["riser_face_centre"]
        c, sn = math.cos(-yaw), math.sin(-yaw)
        poly = []
        for x, y in st["tread_polygon_xy"]:
            dx, dy = x - cx, y - cy
            poly.append((dx * c - dy * sn, dx * sn + dy * c))
        polys.append(tuple(poly))
    geometry = StairLadderGeometry(
        num_treads=L * len(steps),
        landing_every=L,
        rail_length_m=0.20,
        # full step block: tread extruded down one human riser (closed riser)
        landing_thickness_m=rise,
        frame_offsets=tuple(offsets),
        frame_yaws=tuple(yaws),
        landing_polygons=tuple(polys),
    )
    geometry.validate()
    return geometry, {"riser": rise / L, "angle_deg": angle, "num_steps": len(steps)}

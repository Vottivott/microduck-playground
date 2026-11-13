"""The MicroDock v2 swing (experiment, on the way to MicroDock v3; the egg dock builds on its parameters and scene): the hood hangs on four cords from a hanger on the board instead of sliding on pegs
in a window. Nothing slides while it floats, gravity centres it again, and the parts' flex no longer matters.

- The hanger: a plate sticking out of the board's face above the hood, short enough that the dozing duck's head passes
  over it (the head comes no lower than ~137 mm within 50 mm of the board, and never within 20 mm of it).
- Four cords from the hanger to tabs low on the hood's side walls, parallel and of equal length, so the hood keeps its
  attitude as it swings (a parallelogram). The hood's centre of mass lies between the front and rear cords.
- Stops: the board's face behind the hood (a small gap at rest), and a slot in the hanger that a tooth on the hood's
  rear wall rises into once the duck has lifted the hood onto its back: its front edge stops the hood a few mm forward,
  so the magnet lets go when the duck scoots out, as it does against the keeper today. Hanging empty, the tooth stays
  below the hanger and the hood swings freely. The rest of the rear wall ends at the top plate, clear of the hanger.
- No window, pegs, keeper or pins.

`build_scene` is dock.build_scene for this hood; `SliderFriction` is today's hood in the simulator with the friction and
cable spring the print has (for comparison). Frames and units as dock.py (mm; world x away from the board's face).
"""
import os, sys
from dataclasses import dataclass, field
# MicroDock v3 builds on the v2 (seated docking) and v1 (dock, CAD, plates) modules beside it
_HW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in ("microdock-v2/source", "microdock-v2/sim", "microdock/source", "microdock/sim"):
    if os.path.join(_HW, _p) not in sys.path: sys.path.append(os.path.join(_HW, _p))
import numpy as np
import seated
import dock as dk
_DOCK_BUILD = dk.build_scene           # today's scene builder (callers may patch dk.build_scene to dispatch here)


@dataclass
class Params(seated.Params):
    hood_mass: float = 0.045        # the print with its adapter and cable head
    hood_com: tuple = (-58.5, 1.9, 28.6)       # hood frame (cad.hood volume + adapter + head)
    rest_gap: float = 2.0           # the rear wall in front of the board's face when hanging
    rest_drop: float = 10.0         # the hood hangs this much below docked (the duck's back lifts it)
    # the hanger (world mm)
    bar_z: tuple = (122.0, 129.0)   # bottom, top
    bar_x1: float = 49.0            # front edge
    bar_half_w: float = 45.0
    slot_x1: float = 9.5            # the forward stop: the hanger's slot ends here (the tooth's front face rests at rest_gap + 3)
    slot_hw: float = 29.0           # the slot's half width (the tooth's, plus the sideways float)
    chamfer: float = 3.0            # under the slot's front edge: guides a tooth that rises early back into the slot
    tooth_hw: float = 8.0           # the tooth on the rear wall: half width, top (hood frame z)
    tooth_top: float = 67.6
    # the cords: tabs on the side walls' outer faces (hood frame x, z), at |y| = side_in + wall_t + cord_out
    cord_x: tuple = (-79.0, -37.5)
    cord_z: float = 24.0
    cord_out: float = 3.0
    # cable: a spring at its exit (left wall) toward where it rests, plus a preload (N/m, N)
    cable_k: float = 3.0
    cable_pre: tuple = (0.0, 0.0, 0.0)

    @property
    def cord_y(self): return self.side_in + self.wall_t + self.cord_out
    def hood_home(self):            # the hood body's position when hanging at rest (world mm)
        return np.array([self.dock_x + self.rest_gap, 0.0, self.z0 - self.rest_drop])
    def cord_sites(self):
        """(name, hood-frame point, world point on the hanger's underside) per cord."""
        h = self.hood_home(); out = []
        for i, xa in enumerate(self.cord_x):
            for sy in (1, -1):
                ph = np.array([xa, sy*self.cord_y, self.cord_z]); pw = h + ph; pw[2] = self.bar_z[0]
                out.append((f"cord_{'rf'[i]}{'lr'[sy < 0]}", ph, pw))
        return out
    def cord_len(self): return self.bar_z[0] - (self.hood_home()[2] + self.cord_z)


@dataclass
class Bars(Params):
    """The swing on four printed bars instead of cords: each bar hangs on a filament pin (along x) in a clevis under the
    hanger and carries the hood on a second pin that rides in slots in a clevis on the hood. The bars swing sideways
    only, so the hood is held front to back (no tooth, no back stop) and can't turn; the slots let the duck lift it.
    Pivots (world x at rest): rear 6.3 (a clevis on the rear wall, clear of the cable's slot), front 33 (behind the
    duck's back, which the battery stop keeps at 38.5: the seated trunk is wider than the hood low down, where the
    clevises hang); |y| 55, so a bar tilted with the hood lifted and to the side clears the top plate's edge. The slots
    (22 mm) are longer than the duck ever lifts the hood (~16 mm): at their ends the bars would push it sideways."""
    rest_gap: float = 4.0
    pivot_xw: tuple = (6.3, 33.0)   # world x of the rear and front pivots at rest
    bar_y: float = 55.0
    pin_z: float = 24.0             # the hood's pins (hood frame z)
    top_pin_z: float = 117.0        # the hanger's pins (world z)
    slot: float = 22.0              # the hood's slots: how far the duck can lift it
    bar_hw: float = 4.0             # a bar's half width (y); 4 mm thick (x)
    bar_half_w: float = 61.0        # the hanger plate's half width (it carries the clevises at |y| 55)
    fric_y: float = 0.01            # N: the pivots (0.1 N on a 1.75 mm pin, 2 pivots per bar)
    fric_z: float = 0.15            # N: the pins sliding in the slots while the duck lifts the hood (conservative)

    def cord_sites(self):           # the bars' pins: (name, hood-frame point, world point of the top pin)
        h = self.hood_home(); out = []
        for i, xw in enumerate(self.pivot_xw):
            for sy in (1, -1):
                ph = np.array([xw - h[0], sy*self.bar_y, self.pin_z]); pw = h + ph; pw[2] = self.top_pin_z
                out.append((f"bar_{'rf'[i]}{'lr'[sy < 0]}", ph, pw))
        return out
    def cord_len(self): return self.top_pin_z - (self.hood_home()[2] + self.pin_z)


BALL_D, ROD_D, THROAT_D = 8.0, 3.0, 6.8   # the rods' top balls and beads, the rods, the seats' throats: a rod clears a
                                          # throat's edge up to acos(ROD_D/BALL_D) - asin(2 seat_dz/BALL_D) = 36 deg of tilt
TAB_HX = 6.0                              # a tab's half length (x): its seat's countersink is 10.2 mm across underneath


@dataclass
class Rods(Params):
    """v3: the swing on four printed rods instead of cords. Each rod hangs from a ball (BALL_D) resting in a cone seat in
    the hanger, passes down through a hole in a tab on the hood, and carries the hood on a bead (a second ball, pinned on
    with filament) resting in a cone under the tab. Like a cord, a rod only pulls: when the duck pushes the hood up, the
    tab slides up the rod and the bead drops away; the rod itself never rises (the dozing head passes ~10 mm over the
    top balls). The rods' lengths set the hood's height. Rods at |y| 54: 5 mm clear of the hood's top edges as they tilt
    and 12 mm clear of the duck (at 50 the edges came within 1.4 mm, at 44 they hit). World x 9 / 46 at rest: the rear
    tabs start in front of the rear wall (at 4.5 they rubbed on the board), and the front pair is far enough forward that
    a stiff cable pushing up at its exit (x 12.8, behind the centre of mass at 25) can't tip the hood nose-down onto the
    front rods and wedge it (with the front pair at 33 it did: 4 of 30 stiff-cable dockings failed). The front seats sit
    on two ears past the hanger's front edge. The forward stop is the cord swing's tooth and slot. A slack rod can lean
    past what its top seat allows (36 deg) for a moment as the duck twists the hood on release: its ball then tips on the
    throat's edge and drops back."""
    rod_xw: tuple = (9.0, 46.0)
    rod_y: float = 54.0
    bar_half_w: float = 63.0        # the hanger plate carries the seats at |y| 54 (and their flares)
    tab_z: float = 30.0             # the tabs' middle (hood frame z); 3 mm thick. 49 mm rods (55 with the tabs at 24 let a
                                    # pushing cable hold the hood further off-centre and passed 2.4 mm from the duck's legs)
    seat_dz: float = round(float(np.sqrt((BALL_D/2)**2 - (THROAT_D/2)**2)), 3)   # a ball's centre above its seat's throat (2.1)
    throat_depth: float = 1.7       # the cone seats: throat this far into the plate / tab

    def cord_sites(self):           # (name, hood-frame bead centre, world top-ball centre)
        h = self.hood_home(); out = []
        for i, xw in enumerate(self.rod_xw):
            for sy in (1, -1):
                ph = np.array([xw - h[0], sy*self.rod_y, self.tab_z - 1.5 + self.throat_depth - self.seat_dz])
                pw = h + ph; pw[2] = self.bar_z[1] - self.throat_depth + self.seat_dz
                out.append((f"rod_{'rf'[i]}{'lr'[sy < 0]}", ph, pw))
        return out
    def cord_len(self): return self.cord_sites()[0][2][2] - (self.hood_home()[2] + self.cord_sites()[0][1][2])


EAR_PAST, EAR_IN = 6.5, 10.0       # the ears reach this far past a front seat, and this far inside the rods


def rods_tab_boxes(p):
    """The rods' tabs on the hood's side walls (hood frame convex pieces)."""
    out = {}; yo = p.side_in + p.wall_t
    for name, ph, _ in p.cord_sites():
        sy = 1.0 if ph[1] > 0 else -1.0
        out["tab_" + name[4:]] = dk.box(ph[0] - TAB_HX, ph[0] + TAB_HX, *sorted((sy*(yo - 0.5), sy*(p.rod_y + TAB_HX + 0.5))), p.tab_z - 1.5, p.tab_z + 1.5)
    return out


def bars_hood_extra(p):
    """The hood's clevises for the bars (hood frame convex pieces): two plates out from each side wall to the pin."""
    out = {}; yo = p.side_in + p.wall_t
    for name, ph, _ in p.cord_sites():
        sy = 1.0 if ph[1] > 0 else -1.0
        for k, (x0, x1) in enumerate(clevis_x(ph[0])):
            out[f"clevis_{name[4:]}{k}"] = dk.box(x0, x1, *sorted((sy*(yo - 0.5), sy*(p.bar_y + 4.5))), p.pin_z - p.slot - 4.0, p.pin_z + 4.0)
    return out


def bars_hanger_boxes(p):
    z0, z1 = p.bar_z; out = {"bar": dk.box(0.0, p.bar_x1, -p.bar_half_w, p.bar_half_w, z0, z1)}
    for name, ph, pw in p.cord_sites():
        for k, (x0, x1) in enumerate(clevis_x(pw[0])):
            out[f"clevis_{name[4:]}{k}"] = dk.box(x0, x1, pw[1] - 4.0, pw[1] + 4.0, p.top_pin_z - 4.0, z0 + 0.01)
    return out


def cable_exit(p):
    """The cable's exit from the hood (hood frame): the left wall's outer face at the head's axis (or p.cable_out)."""
    if getattr(p, "cable_out", ()): return np.asarray(p.cable_out, float)
    pv, ph, pl = p.plug; al, aw, at = p.angle_plug
    return np.array([p.adapter_face_x - pl - (aw - at/2), p.side_in + p.wall_t, p.adapter_z])


def hanger_boxes(p):
    """The hanger's convex pieces (world mm): the plate in front of the slot (its front edge chamfered underneath) and the
    plate either side of it."""
    z0, z1 = p.bar_z; x1, c = p.slot_x1, p.chamfer
    front = np.array([[x, y, z] for y in (-p.bar_half_w, p.bar_half_w) for x, z in
                      ((x1, z0 + c), (x1, z1), (x1 + c, z0), (p.bar_x1, z0), (p.bar_x1, z1))], float)
    out = {"bar": front,
           "bar_l": dk.box(0.0, x1 + 0.01, p.slot_hw, p.bar_half_w, z0, z1),
           "bar_r": dk.box(0.0, x1 + 0.01, -p.bar_half_w, -p.slot_hw, z0, z1)}
    out.update(rods_ears(p))
    if hasattr(p, "hanger_extra_boxes"): out.update(p.hanger_extra_boxes())     # e.g. strapswing.Straps' ears
    return out


def rods_ears(p):
    """Rods: the plate's two ears past its front edge, carrying the front rods' seats (world mm boxes)."""
    if not isinstance(p, Rods) or p.rod_xw[1] + EAR_PAST <= p.bar_x1: return {}
    z0, z1 = p.bar_z; y0 = p.rod_y - EAR_IN
    return {"ear_l": dk.box(p.bar_x1 - 0.01, p.rod_xw[1] + EAR_PAST, y0, p.bar_half_w, z0, z1),
            "ear_r": dk.box(p.bar_x1 - 0.01, p.rod_xw[1] + EAR_PAST, -p.bar_half_w, -y0, z0, z1)}


def swing_hood_pieces(p):
    """dock.hood_pieces with the rear wall ending at the top plate and the tooth rising from its middle."""
    pc = dk.hood_pieces(p); xr = p.hood_rear; yo = p.side_in + p.wall_t; top1 = p.trunk_top + 3.0
    pc["rear"] = dk.box(xr, xr + 3.0, -yo, yo, 0.0, top1)
    pc["tooth"] = dk.box(xr, xr + 3.0, -p.tooth_hw, p.tooth_hw, top1 - 0.5, p.tooth_top)
    return pc


def tab_boxes(p):
    """The cord tabs on the hood's side walls (hood frame mm)."""
    out = {}
    for name, ph, _ in p.cord_sites():
        sy = np.sign(ph[1]); y0 = p.side_in + p.wall_t - 0.5
        out[name] = dk.box(ph[0] - 2.5, ph[0] + 2.5, *sorted((sy*y0, sy*(p.cord_y + 2.5))), p.cord_z - 1.5, p.cord_z + 1.5)
    return out


def build_scene(p, robot_dir, scene_path=None):
    """The dock with the hood on four cords (free in all six directions, held by the cords, the board and the slot)."""
    import mujoco
    if isinstance(p, SliderFriction): return slider_scene(p, robot_dir, scene_path)
    scene_path = scene_path or os.path.join(robot_dir, f"scene_microdock_swing_{os.getpid()}.xml")
    assets, world, hood, tendons = [], [], [], []
    bars = isinstance(p, Bars)
    static = p.static_pieces() if hasattr(p, "static_pieces") else dk.static_pieces(p)
    for k, P in list(static.items()) + list((bars_hanger_boxes(p) if bars else hanger_boxes(p)).items()):
        assets.append(dk._mesh_xml(f"dock_{k}", P))
        world.append(f'<geom name="dock_{k}" type="mesh" mesh="dock_{k}" rgba="{dk.LIGHT}" friction="0.35 0.005 0.0001"/>')
    world.append(f'<geom name="tag" type="box" pos="0.0003 0 {p.tag_cz/1e3:.5f}" size="0.0003 {p.tag_size/2e3:.5f} {p.tag_size/2e3:.5f}" '
                 f'rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>')
    hood_pc = swing_hood_pieces(p)
    if bars:
        hood_pc.pop("tooth"); hood_pc.update(bars_hood_extra(p))
    elif isinstance(p, Rods):
        hood_pc.update(rods_tab_boxes(p))
    elif hasattr(p, "hood_extra_pieces"):                                       # e.g. strapswing.Straps' tabs and sockets
        hood_pc.update(p.hood_extra_pieces())
    else:
        hood_pc.update(tab_boxes(p))
    for k, P in hood_pc.items():
        assets.append(dk._mesh_xml(f"hood_{k}", P))
        hood.append(f'<geom name="hood_{k}" type="mesh" mesh="hood_{k}" rgba="{dk.GREEN}" contype="5" conaffinity="5" solref="0.004 1" friction="0.25 0.005 0.0001"/>')
    assets.append(dk._mesh_xml("hood_adapter", dk.adapter_box(p)))
    hood.append(f'<geom name="hood_adapter" type="mesh" mesh="hood_adapter" rgba="0.15 0.17 0.25 1" contype="5" conaffinity="5" solref="0.004 1" friction="0.3 0.005 0.0001"/>')
    hood.append(f'<site name="adapter_face" pos="{p.adapter_face_x/1e3:.5f} 0 {p.adapter_z/1e3:.5f}" size="0.001"/>')
    ce = cable_exit(p); hood.append(f'<site name="cable_exit" pos="{ce[0]/1e3:.5f} {ce[1]/1e3:.5f} {ce[2]/1e3:.5f}" size="0.001"/>')
    chains, welds = p.chain_xml() if getattr(p, "strap_chains", False) else ([], [])
    for name, ph, pw in ([] if chains else p.cord_sites()):
        hood.append(f'<site name="{name}_h" pos="{ph[0]/1e3:.5f} {ph[1]/1e3:.5f} {ph[2]/1e3:.5f}" size="0.0008"/>')
        world.append(f'<site name="{name}_w" pos="{pw[0]/1e3:.5f} {pw[1]/1e3:.5f} {pw[2]/1e3:.5f}" size="0.0008"/>')
        lo = (p.cord_len() - p.slot)/1e3 if bars else 0.0
        tendons.append(f'<spatial name="{name}" limited="true" range="{lo:.5f} {p.cord_len()/1e3:.5f}" width="{0.002 if bars else 0.0015 if isinstance(p, Rods) else 0.0004}" '
                       f'rgba="0.95 0.95 0.95 1" solreflimit="0.003 1"><site site="{name}_w"/><site site="{name}_h"/></spatial>')
    h = p.hood_home(); c = p.hood_com
    xml = f'''<mujoco model="microdock_swing">
  <include file="robot_groundcontact.xml"/>
  <visual><headlight diffuse="0.5 0.5 0.5" ambient="0.38 0.38 0.38" specular="0.05 0.05 0.05"/><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.80 0.78 0.74" rgb2="0.74 0.72 0.68" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="14 14" reflectance="0.04"/>
    {chr(10).join("    "+a for a in assets)}
  </asset>
  <worldbody>
    <light pos="0.4 -0.6 1.4" dir="-0.25 0.4 -1" directional="true" castshadow="false" diffuse="0.55 0.55 0.55"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    {chr(10).join("    "+w for w in world)}
    <body name="hood" pos="{h[0]/1e3:.5f} 0 {h[2]/1e3:.5f}">
      {'' if bars else '<joint name="hood_x" type="slide" axis="1 0 0" damping="0.02"/>'}
      <joint name="hood_y" type="slide" axis="0 1 0" damping="0.02" frictionloss="{p.fric_y if bars else 0}"/>
      <joint name="hood_z" type="slide" axis="0 0 1" damping="0.02" frictionloss="{p.fric_z if bars else 0}"/>
      {'' if bars else f'<joint name="hood_rz" type="hinge" axis="0 0 1" damping="2e-5" pos="{c[0]/1e3:.5f} {c[1]/1e3:.5f} {c[2]/1e3:.5f}"/>'}
      <joint name="hood_ry" type="hinge" axis="0 1 0" damping="2e-5" pos="{c[0]/1e3:.5f} {c[1]/1e3:.5f} {c[2]/1e3:.5f}"/>
      <joint name="hood_rx" type="hinge" axis="1 0 0" damping="2e-5" pos="{c[0]/1e3:.5f} {c[1]/1e3:.5f} {c[2]/1e3:.5f}"/>
      <inertial pos="{c[0]/1e3:.5f} {c[1]/1e3:.5f} {c[2]/1e3:.5f}" mass="{p.hood_mass}" diaginertia="5.6e-5 5.6e-5 9.1e-5"/>
      {chr(10).join("      "+x for x in hood)}
    </body>
    {chr(10).join("    "+c for c in chains)}
  </worldbody>
  <tendon>
    {chr(10).join("    "+t for t in tendons)}
  </tendon>
  <equality>
    {chr(10).join("    "+w for w in welds)}
  </equality>
</mujoco>'''
    with open(scene_path, "w") as f:
        f.write(xml)
    spec = mujoco.MjSpec.from_file(scene_path)
    os.remove(scene_path)
    return spec


@dataclass
class SliderFriction(seated.Params):
    """Today's hood in the simulator, as printed: no centring spring (the V-seats can't centre it against PLA's
    friction), Coulomb friction on both slides, the print's mass, and the cable's spring at its exit."""
    hood_mass: float = 0.045
    friction_n: float = 0.5         # N, on each slide
    cable_k: float = 3.0
    cable_pre: tuple = (0.0, 0.0, 0.0)


def slider_scene(p, robot_dir, scene_path=None):
    import mujoco
    scene_path = scene_path or os.path.join(robot_dir, f"scene_microdock_slider_{os.getpid()}.xml")
    spec = _DOCK_BUILD(p, robot_dir, scene_path)
    os.remove(scene_path)
    for j in spec.joints:
        if j.name in ("hood_y", "hood_z"):
            j.stiffness = np.zeros((3, 1)); j.frictionloss = p.friction_n
    ce = cable_exit(p)
    spec.body("hood").add_site(name="cable_exit", pos=[v/1e3 for v in ce], size=[0.001, 0, 0])
    return spec


# --------------------------------------------------------------------------------------- the cable (both designs)
_REST = {}
def cable_force(model, data, p):
    """The cable at its exit: a spring toward where the exit rests (when the hood hangs or sits centred), plus a preload.
    Adds to data.qfrc_applied (call after dock.magnet, which clears it)."""
    import mujoco
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cable_exit")
    key = id(model)
    if key not in _REST: _REST[key] = rest_exit(model, p)
    F = -p.cable_k*(data.site_xpos[s] - _REST[key]) + np.asarray(p.cable_pre, float)
    mujoco.mj_applyFT(model, data, F, np.zeros(3), data.site_xpos[s], model.site_bodyid[s], data.qfrc_applied)


def rest_exit(model, p):
    """The exit's world position with the hood at rest (hanging; today's hood at the bottom of its slide)."""
    import mujoco
    d = mujoco.MjData(model)
    if isinstance(p, SliderFriction): d.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_z")]] = -p.hood_drop/1e3
    mujoco.mj_forward(model, d)
    return d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cable_exit")].copy()


# --------------------------------------------------------------------------------------- printable parts
# cad.py's conventions: manifold3d solids in the parts' assembly frames (the hood in the hood frame, the rest in world mm).
TONGUE_Y = (32.0, 48.0)            # the hanger's two tongues through the board (|y| range), and their length behind it
TONGUE_L = 12.0
CORD_HOLE = 1.6                    # braided fishing line (0.3-0.5 mm) and its knot's tail


def hood_cad(p):
    """cad.hood without its pegs and the rear wall's rise above the top plate, with the tooth and a tab for each cord
    on the side walls' outer faces. A 45-degree gusset above each tab holds it up when the hood prints upside down; the
    cord goes up through the tab and the gusset."""
    import cad, manifold3d as m3
    top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    s = cad.hood(p) - cad.cube(p.hood_rear - 40, p.hood_rear, -80, 80, -20, 90)      # the pegs (behind the rear wall)
    s = s - cad.cube(p.hood_rear - 1, p.hood_rear + 3.01, -yo - 1, yo + 1, top1, 90)  # the rear wall's rise
    s = s + cad.cube(p.hood_rear, p.hood_rear + 3.0, -p.tooth_hw, p.tooth_hw, top1 - 0.5, p.tooth_top)
    for name, ph, _ in p.cord_sites():
        sy = 1.0 if ph[1] > 0 else -1.0; ye = p.cord_y + 2.5
        z0, z1 = p.cord_z - 1.5, p.cord_z + 1.5
        pts = [[x, sy*y, z] for x in (ph[0] - 2.5, ph[0] + 2.5)
               for (y, z) in ((yo - 0.5, z0), (ye, z0), (ye, z1), (yo - 0.5, z1 + (ye - yo + 0.5)))]
        tab = cad.hull(np.array(pts)) - m3.Manifold.cylinder(40, CORD_HOLE/2, CORD_HOLE/2, 24).translate((ph[0], ph[1], z0 - 10))
        s = s + tab
    return s


def board_cad(p):
    """The board: cad.board's plate, tag pocket and corners, without the window, its V-seats and the cable clip, and with
    two slots for the hanger's tongues."""
    import cad
    T, w = p.T, p.board_half_w
    s = cad.cube(-p.board_t, 0.0, -w, w, T, p.board_top)
    tw = p.tag_size/2 + 2.0 + 0.15
    s = s - cad.cube(-1.6, 1, -tw, tw, p.tag_cz - tw, p.tag_cz + tw)
    r = 8.0
    for sy in (-1, 1):
        y0, y1 = (w - r, w + 1) if sy > 0 else (-w - 1, -w + r)
        s = s - (cad.cube(-p.board_t - 1, 1, y0, y1, p.board_top - r, p.board_top + 1) - cad.cyl_x(r, -p.board_t - 2, 2, sy*(w - r), p.board_top - r))
    z0, z1 = p.bar_z
    for sy in (-1, 1):
        y0, y1 = sorted((sy*(TONGUE_Y[0] - 0.15), sy*(TONGUE_Y[1] + 0.15)))
        s = s - cad.cube(-p.board_t - 1, 1, y0, y1, z0 - 0.1, z1 + 0.1)
    return s


def pin_x(p):
    """The hanger's filament pins (world x): just behind the board's back face."""
    import cad
    return -p.board_t - cad.PIN_GAP - cad.PIN_D/2


def hanger_cad(p):
    """World frame: the plate over the hood, with the cords' holes and the slot for the hood's tooth (its front edge, the
    forward stop, chamfered underneath), and two tongues through the board, each locked behind it by a filament pin
    (the plate's shoulders bear on the board's front). Prints flat, as it sits."""
    import cad, manifold3d as m3
    z0, z1 = p.bar_z; x1, c = p.slot_x1, p.chamfer
    s = cad.cube(0.0, p.bar_x1, -p.bar_half_w, p.bar_half_w, z0, z1)
    s = s - cad.cube(-1.0, x1, -p.slot_hw, p.slot_hw, z0 - 1, z1 + 1)
    s = s - cad.hull(np.array([[x, y, z] for y in (-p.slot_hw, p.slot_hw) for x, z in ((x1 - 0.01, z0 - 0.01), (x1 - 0.01, z0 + c), (x1 + c, z0 - 0.01))]))
    for sy in (-1, 1):
        y0, y1 = sorted((sy*TONGUE_Y[0], sy*TONGUE_Y[1]))
        s = s + cad.cube(-p.board_t - TONGUE_L, 0.01, y0, y1, z0, z1)
        s = s - m3.Manifold.cylinder(z1 - z0 + 2, cad.PIN_HOLE/2, cad.PIN_HOLE/2, 24).translate((pin_x(p), sy*sum(TONGUE_Y)/2, z0 - 1))
    for name, ph, pw in p.cord_sites():
        s = s - m3.Manifold.cylinder(z1 - z0 + 2, CORD_HOLE/2, CORD_HOLE/2, 24).translate((pw[0], pw[1], z0 - 1))
    return s


def stand_cad(p):
    """The setting stand: on the base plate, it holds the hood at its rest height, its rear wall rest_gap off the board,
    while the cords are tied; then it comes out."""
    import cad
    h = p.hood_home(); z_floor = h[2]                      # the hood's floor web bottom (hood frame z 0)
    x0 = h[0] + p.hood_rear                                 # the rear wall's back face at rest
    s = cad.cube(x0 + 6.0, x0 + 26.0, -12.0, 12.0, p.T, z_floor)
    s = s + cad.cube(0.2, x0 - 0.1, -12.0, 12.0, p.T, z_floor + 14.0)   # the gap fin between the board and the rear wall
    s = s + cad.cube(0.2, x0 + 26.0, -12.0, 12.0, p.T, p.T + 4.0)       # foot joining them
    return s


# ------------------------------------------------------------------------- printable parts for the bars (Bars)
PIN_SNUG, PIN_FREE = 1.85, 2.05    # a 1.75 mm filament pin: snug in the bars, free in the hanger's holes and the hood's slots
BAR_T = 3.0                        # a bar's thickness (x)
CLEVIS_T, CLEVIS_GAP = 2.0, 3.4    # each clevis: two plates this thick, this far apart (the bar between them); the rear
                                   # clevis fits between the rear wall's back face and the cable's slot


def clevis_x(x_c):
    """The two plates of a clevis centred at x_c: [(x0, x1), (x0, x1)]."""
    return [(x_c - CLEVIS_GAP/2 - CLEVIS_T, x_c - CLEVIS_GAP/2), (x_c + CLEVIS_GAP/2, x_c + CLEVIS_GAP/2 + CLEVIS_T)]


def bars_hood_cad(p):
    """cad.hood without its pegs and the rear wall's rise, with a clevis out from each side wall for a bar: two plates
    with a vertical slot. The pin, snug in the bar, rests at the slot's top end and the hood rides up it when the duck
    lifts the hood. The plates print as walls when the hood prints upside down."""
    import cad
    top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    s = cad.hood(p) - cad.cube(p.hood_rear - 40, p.hood_rear, -80, 80, -20, 90)
    s = s - cad.cube(p.hood_rear - 1, p.hood_rear + 3.01, -yo - 1, yo + 1, top1, 90)
    for name, ph, _ in p.cord_sites():
        sy = 1.0 if ph[1] > 0 else -1.0; ye = p.bar_y + 4.5
        z0, z1 = p.pin_z - p.slot - 4.0, p.pin_z + 4.0
        for x0, x1 in clevis_x(ph[0]):
            plate = cad.cube(x0, x1, *sorted((sy*(yo - 0.5), sy*ye)), z0, z1)
            plate = plate - cad.hull(np.array([[x, ph[1] + dy, z] for x in (x0 - 1, x1 + 1) for dy in (-PIN_FREE/2, PIN_FREE/2)
                                               for z in (p.pin_z - p.slot, p.pin_z)])) \
                          - cad.cyl_x(PIN_FREE/2, x0 - 1, x1 + 1, ph[1], p.pin_z) - cad.cyl_x(PIN_FREE/2, x0 - 1, x1 + 1, ph[1], p.pin_z - p.slot)
            s = s + plate
    return s


def bar_cad(p):
    """One bar, in its own frame (the top pin's axis is the x axis through the origin, the bar hangs along -z): a plate
    with a round hole at each end, cord_len apart. The pins are snug in it."""
    import cad
    L, w, t = p.cord_len(), p.bar_hw, BAR_T
    s = cad.cube(-t/2, t/2, -w, w, -L, 0.0) + cad.cyl_x(w, -t/2, t/2, 0.0, 0.0) + cad.cyl_x(w, -t/2, t/2, 0.0, -L)
    return s - cad.cyl_x(PIN_SNUG/2, -5, 5, 0.0, 0.0) - cad.cyl_x(PIN_SNUG/2, -5, 5, 0.0, -L)


def bars_hanger_cad(p):
    """World frame: the plate over the hood with a clevis under it for each bar (pins along x, free in its holes), and
    the two tongues through the board with their filament pins. Prints upside down (the clevises up)."""
    import cad, manifold3d as m3
    z0, z1 = p.bar_z
    s = cad.cube(0.0, p.bar_x1, -p.bar_half_w, p.bar_half_w, z0, z1)
    for sy in (-1, 1):
        y0, y1 = sorted((sy*TONGUE_Y[0], sy*TONGUE_Y[1]))
        s = s + cad.cube(-p.board_t - TONGUE_L, 0.01, y0, y1, z0, z1)
        s = s - m3.Manifold.cylinder(z1 - z0 + 2, cad.PIN_HOLE/2, cad.PIN_HOLE/2, 24).translate((pin_x(p), sy*sum(TONGUE_Y)/2, z0 - 1))
    for name, ph, pw in p.cord_sites():
        for x0, x1 in clevis_x(pw[0]):
            s = s + cad.cube(x0, x1, pw[1] - p.bar_hw, pw[1] + p.bar_hw, p.top_pin_z, z0 + 0.01) + cad.cyl_x(p.bar_hw, x0, x1, pw[1], p.top_pin_z)
        s = s - cad.cyl_x(PIN_FREE/2, pw[0] - 8, pw[0] + 8, pw[1], p.top_pin_z)
    return s


# ------------------------------------------------------------------------- printable parts for the rods (Rods, v3)
TILT_Y, TILT_X = 40.0, 24.0               # how far the seats' flares let a rod tilt sideways and front to back (deg)


def _seat(p, c, up, depth_below):
    """A cone seat for a ball at c (world or hood frame), opening toward the ball (up = the ball sits above it): the
    90-degree countersink to the throat, then a hole flaring away from the ball, oblong (TILT_Y sideways, TILT_X front to
    back) so the rod can tilt. depth_below: how far the flare runs (to the plate's or tab's far face)."""
    import cad
    sgn = 1.0 if up else -1.0
    zt = c[2] - sgn*p.seat_dz                                  # the throat's plane
    rt = THROAT_D/2; rc = rt + p.throat_depth + 0.6             # the countersink: throat -> past the near face
    cs = cad.hull(np.array([[c[0] + rt*np.cos(a), c[1] + rt*np.sin(a), zt] for a in np.linspace(0, 2*np.pi, 32, endpoint=False)] +
                           [[c[0] + rc*np.cos(a), c[1] + rc*np.sin(a), zt + sgn*(rc - rt)] for a in np.linspace(0, 2*np.pi, 32, endpoint=False)]))
    lever = p.seat_dz + depth_below                            # from the ball's centre to the far face
    ry = ROD_D/2 + 0.25 + lever*np.tan(np.radians(TILT_Y)); rx = ROD_D/2 + 0.25 + lever*np.tan(np.radians(TILT_X))
    far = zt - sgn*(depth_below + 0.5)
    fl = cad.hull(np.array([[c[0] + rt*np.cos(a), c[1] + rt*np.sin(a), zt + sgn*0.01] for a in np.linspace(0, 2*np.pi, 32, endpoint=False)] +
                           [[c[0] + rx*np.cos(a), c[1] + ry*np.sin(a), far] for a in np.linspace(0, 2*np.pi, 48, endpoint=False)]))
    return cs + fl


def rods_hood_cad(p):
    """cad.hood without its pegs and the rear wall's rise, with the tooth (as hood_cad) and a 3 mm tab for each rod out from
    the side wall: the bead's cone seat underneath, the rod's hole flaring upward."""
    import cad
    top1 = p.trunk_top + 3.0; yo = p.side_in + p.wall_t
    s = cad.hood(p) - cad.cube(p.hood_rear - 40, p.hood_rear, -80, 80, -20, 90)
    s = s - cad.cube(p.hood_rear - 1, p.hood_rear + 3.01, -yo - 1, yo + 1, top1, 90)
    s = s + cad.cube(p.hood_rear, p.hood_rear + 3.0, -p.tooth_hw, p.tooth_hw, top1 - 0.5, p.tooth_top)
    for name, ph, _ in p.cord_sites():
        sy = 1.0 if ph[1] > 0 else -1.0
        tab = cad.cube(ph[0] - TAB_HX, ph[0] + TAB_HX, *sorted((sy*(yo - 0.5), sy*(p.rod_y + TAB_HX + 0.5))), p.tab_z - 1.5, p.tab_z + 1.5)
        s = s + tab - _seat(p, ph, up=False, depth_below=3.0 - p.throat_depth)
        # above the throat a plain 45-degree cone too: with the hood up, the slack rod slides through anywhere in the
        # throat at up to ~36 deg (not about the bead's centre, which the flare is shaped for)
        rt, zt = THROAT_D/2, ph[2] + p.seat_dz; d = (3.0 - p.throat_depth) + 0.6
        s = s - cad.hull(np.array([[ph[0] + r*np.cos(a), ph[1] + r*np.sin(a), z] for r, z in ((rt, zt), (rt + d, zt + d))
                                   for a in np.linspace(0, 2*np.pi, 40, endpoint=False)]))
    return s


def rods_hanger_cad(p):
    """World frame: the cord swing's hanger (the plate, the slot for the hood's tooth, the two tongues through the board),
    wider, with a cone seat for each rod's ball on top and the rod's hole flaring underneath. Prints flat, as it sits."""
    import cad, manifold3d as m3
    z0, z1 = p.bar_z; x1, c = p.slot_x1, p.chamfer
    s = cad.cube(0.0, p.bar_x1, -p.bar_half_w, p.bar_half_w, z0, z1)
    s = s - cad.cube(-1.0, x1, -p.slot_hw, p.slot_hw, z0 - 1, z1 + 1)
    s = s - cad.hull(np.array([[x, y, z] for y in (-p.slot_hw, p.slot_hw) for x, z in ((x1 - 0.01, z0 - 0.01), (x1 - 0.01, z0 + c), (x1 + c, z0 - 0.01))]))
    for sy in (-1, 1):
        y0, y1 = sorted((sy*TONGUE_Y[0], sy*TONGUE_Y[1]))
        s = s + cad.cube(-p.board_t - TONGUE_L, 0.01, y0, y1, z0, z1)
        s = s - m3.Manifold.cylinder(z1 - z0 + 2, cad.PIN_HOLE/2, cad.PIN_HOLE/2, 24).translate((pin_x(p), sy*sum(TONGUE_Y)/2, z0 - 1))
    for P in rods_ears(p).values():
        s = s + cad.hull(np.asarray(P, float))
    for name, ph, pw in p.cord_sites():
        s = s - _seat(p, pw, up=True, depth_below=(z1 - z0) - p.throat_depth)
    return s


BOSS_D, BOSS_TOP = 4.6, 1.0      # the rod's end thickens inside the bead (around the pin; it still passes the throats)


def rod_cad(p):
    """One rod in its own frame (the top ball's centre at the origin, the rod hanging along -z to the bead's centre at
    -cord_len and a little past it), thickening to a boss inside the bead around the cross hole for the bead's filament
    pin. Prints standing on the top ball (a small flat on its top: the seat bears on its lower half)."""
    import cad, manifold3d as m3
    L = p.cord_len(); bot = -L - BALL_D/2 - 0.5
    s = m3.Manifold.sphere(BALL_D/2, 48) + m3.Manifold.cylinder(-bot, ROD_D/2, ROD_D/2, 32).translate((0, 0, bot))
    s = s + m3.Manifold.cylinder(BOSS_TOP - bot - L, BOSS_D/2, BOSS_D/2, 32).translate((0, 0, bot))
    s = s - cad.cube(-5, 5, -5, 5, BALL_D/2 - 1.0, 10)      # a 5.3 mm flat to stand on (print with a brim)
    return s - cad.cyl_x(cad.PIN_HOLE/2, -5, 5, 0.0, -L)


def bead_cad(p):
    """The bead in its own frame (centre at the origin): a ball bored for the rod's boss from below and the rod above it,
    with the filament pin's cross hole along x. Prints as it sits (a small flat underneath: the tab's seat bears on its
    upper half)."""
    import cad, manifold3d as m3
    s = m3.Manifold.sphere(BALL_D/2, 48) - cad.cube(-5, 5, -5, 5, -10, -BALL_D/2 + 0.5)
    s = s - m3.Manifold.cylinder(BALL_D + 2, ROD_D/2 + 0.15, ROD_D/2 + 0.15, 32).translate((0, 0, -BALL_D/2 - 1))
    s = s - m3.Manifold.cylinder(BALL_D/2 + 1 + BOSS_TOP + 0.3, BOSS_D/2 + 0.15, BOSS_D/2 + 0.15, 32).translate((0, 0, -BALL_D/2 - 1))
    return s - cad.cyl_x(cad.PIN_HOLE/2, -5, 5, 0.0, 0.0)

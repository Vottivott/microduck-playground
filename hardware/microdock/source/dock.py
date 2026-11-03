"""MicroDock: Microduck backs in; a floating hood on the dock's board settles onto its back, the hood's side walls
centre it, and the magnetic USB-C adapter in the hood meets the magnetic tip in Microduck's own USB-C port (on the
back, in a cut-out of the shell just above the battery, as on recent units). Everything prints; no screws, inserts
or feet.

This module holds the design parameters, the convex pieces the simulator collides with, the MuJoCo scene and the
magnet model. `cad.py` turns the same parameters into the printable parts.

Frames (mm): world x points away from the board (the duck approaches from +x), board front face at x = 0, floor
z = 0. The hood frame equals the duck's trunk_base frame when docked at the nominal standing height.
"""
import os
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    # robot (trunk_base frame, from the official model)
    bat_drop: float = 10.0          # battery lowered vs the official model: room for the USB-C port above it (recent units)
    bat_top_old: float = 31.7       # battery top end z in the official model
    port_above_bat: float = 4.8     # USB-C port centre above the battery's top end (4.5 mm above the shell's edge over it)
    # The shell halves cover the back above the battery; a tight cut-out in their bottom edge, centred on the seam,
    # shows the USB-C port, which sits slightly deeper inside (photos of recent units).
    shell_back_x: float = -47.13    # where the magnetic tip's head rests: the shells' back at the cut-out's top edge (seam ridge)
    port_recess: float = 1.0        # USB-C receptacle face behind the shell surface
    notch_w: float = 12.1           # cut-out width, centred on the seam (the 11.8 mm magnetic tip head just fits)
    notch_above_port: float = 1.85  # cut-out top above the port centre (it hugs the receptacle)
    # power button (white raised collar + yellow cap) and two status lights on the top face, rear of the shoulders
    button_xy: tuple = (-28.0, 13.0)
    button_r: float = 4.6           # yellow cap radius; collar outer radius = button_r + 1.2
    leds_xy: tuple = ((-28.0, -9.0), (-28.0, -15.5))
    skid_y: float = 22.5            # hood skids rest on the top face outside |y| > skid_y (clear of button and lights)
    back_face_x: float = -46.6      # battery back face
    tip_out: float = 3.4            # magnetic tip head length; the head rests on the shell around the cut-out
    tip_w: float = 11.8
    tip_h: float = 6.0
    trunk_half_w: float = 32.3
    trunk_top: float = 40.6         # skid contact height on the top face (|y| ~ 23-31)
    neck_x: float = -14.0           # front limit for the hood's skids (neck opening starts at ~-16, |y| < 12)
    brim_x: float = -40.0           # full-width part of the hood's top ends here; skids continue forward
    # base plate
    T: float = 3.0
    plate_x: tuple = (-30.0, 165.0)
    plate_half_w: float = 100.0
    ramp_len: float = 40.0
    ramp_edge: float = 0.6
    # board (stands in a socket at the plate's rear)
    board_t: float = 6.0
    board_half_w: float = 55.0
    board_top: float = 250.0        # world z
    socket_h: float = 16.0
    tag_size: float = 56.0
    tag_cz: float = 211.0           # world z of the tag centre
    # docking geometry
    dock_x: float = 81.5            # trunk origin x when docked (board face at trunk x = -81.5)
    z0: float = 118.6               # nominal trunk origin height when standing on the plate (measured in simulation)
    # hood (hood frame)
    hood_rear: float = -81.5        # rear wall back face (touches the board); room behind the adapter for the cable head
    wall_t: float = 2.4
    side_in: float = 33.0           # side wall inner face |y|
    wall_front: float = -6.0        # side walls run along the trunk to here (hood frame x), flares beyond
    side_bot: float = 14.0          # side wall bottom z
    flare_len: float = 22.0
    flare_out: float = 20.0         # extra half-width at the flare tip
    lip_len: float = 7.0
    lip_rise: float = 6.0
    stop_face_x: float = -47.0      # battery stop (just behind the battery's back face)
    plug: tuple = (8.0, 12.0, 14.0) # magnetic adapter (vertical, horizontal, length)
    # right-angle USB-C head (UGREEN 100 W, measured on its product photos): a round aluminium body along the cable with a
    # black elbow on its side that holds the plug. Length from the plug-side end to the end of the strain relief, depth
    # from the plug's base to the back of the body, body diameter. The cable leaves on the body's axis.
    angle_plug: tuple = (27.6, 9.5, 6.8)
    angle_elbow: tuple = (2.7, 10.3, 5.33)  # elbow: depth in front of the body, length along it, body end beyond the plug's axis
    hood_drop: float = 7.0          # rest position below docked when no robot
    hood_lift: float = 6.0
    hood_float_y: float = 12.0
    hood_travel_x: float = 0.0      # 0 = the hood is rigid in x (sits against the board); >0 adds a spring-loaded travel
    hood_kx: float = 25.0           # N/m (only with travel)
    hood_mass: float = 0.028

    @property
    def bat_top(self): return self.bat_top_old - self.bat_drop
    @property
    def port_z(self): return self.bat_top + self.port_above_bat
    @property
    def tip_face_x(self): return self.shell_back_x - self.tip_out
    @property
    def adapter_face_x(self): return self.tip_face_x - 0.1


def box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def hood_pieces(p: Params):
    """Convex pieces of the hood in the hood frame (mm)."""
    pc = {}
    top0, top1 = p.trunk_top, p.trunk_top + 3.0
    xr, xf = p.hood_rear, p.neck_x
    yo = p.side_in + p.wall_t
    pc["top"] = box(xr, p.brim_x, -yo, yo, top0, top1)                     # rear brim (the trunk's rounded rear edge passes under it)
    for sgn in (1, -1):                                                      # skids straddle the power button and the status lights
        y0, y1 = sorted((sgn*p.skid_y, sgn*yo))
        pc[f"skid{'L' if sgn > 0 else 'R'}"] = box(p.brim_x - 0.5, xf - p.lip_len, y0, y1, top0, top1)
        pc[f"lip{'L' if sgn > 0 else 'R'}"] = np.array([[xf - p.lip_len - 0.5, y, z] for y in (y0, y1) for z in (top0, top1)] +
                                                       [[xf, y, z] for y in (y0, y1) for z in (top0 + p.lip_rise, top1 + p.lip_rise)], float)
    for s in (1, -1):
        yi, yw = s*p.side_in, s*yo
        pc[f"wall{'L' if s > 0 else 'R'}"] = box(xr, p.wall_front, min(yi, yw), max(yi, yw), p.side_bot, top1)
        # flare beyond the wall: inner face opens from side_in to side_in+flare_out
        x0f, x1f = p.wall_front - 0.5, p.wall_front + p.flare_len
        pc[f"flare{'L' if s > 0 else 'R'}"] = np.array(
            [[x0f, y, z] for y in (s*p.side_in, s*(p.side_in + p.wall_t)) for z in (p.side_bot, top1)] +
            [[x1f, y, z] for y in (s*(p.side_in + p.flare_out), s*(p.side_in + p.flare_out + p.wall_t)) for z in (p.side_bot + 3.0, top1)], float)
    pc["rear"] = box(xr, xr + 3.0, -yo, yo, 0.0, top1 + 10.0)      # rises above the top plate: always covers the board window
    pv, ph, pl = p.plug
    ax0 = p.adapter_face_x
    # adapter sleeve (holds the magnetic adapter, face slightly proud of it), the web to the top plate is behind the head
    pc["sleeve_l"] = box(ax0 - pl + 0.4, ax0 - 0.6, ph/2 + 0.2, ph/2 + 2.2, p.port_z - pv/2 - 2.0, p.port_z + pv/2 + 2.0)
    pc["sleeve_r"] = box(ax0 - pl + 0.4, ax0 - 0.6, -ph/2 - 2.2, -ph/2 - 0.2, p.port_z - pv/2 - 2.0, p.port_z + pv/2 + 2.0)
    pc["sleeve_b"] = box(ax0 - pl + 0.4, ax0 - 0.6, -ph/2 - 2.2, ph/2 + 2.2, p.port_z - pv/2 - 2.0, p.port_z - pv/2 - 0.2)
    # battery stop below the sleeve, and the floor web back to the rear wall
    pc["stop"] = box(p.stop_face_x - 3.0, p.stop_face_x, -16.0, 16.0, 1.0, p.bat_top - 1.0)
    pc["floor"] = box(xr, p.stop_face_x - 3.0, -16.0, 16.0, 0.0, 2.6)
    pc["web"] = box(ax0 - pl + 0.4, p.stop_face_x - 3.0, -ph/2 - 2.2, ph/2 + 2.2, 1.0, p.port_z - pv/2 - 1.8)
    pc["sideweb_l"] = box(xr, ax0 - pl + 0.4, ph/2 + 0.2, ph/2 + 2.2, 0.0, top1)
    pc["sideweb_r"] = box(xr, ax0 - pl + 0.4, -ph/2 - 2.2, -ph/2 - 0.2, 0.0, top1)
    return pc


def adapter_box(p: Params):
    pv, ph, pl = p.plug
    ax0 = p.adapter_face_x
    return box(ax0 - pl, ax0, -ph/2, ph/2, p.port_z - pv/2, p.port_z + pv/2)


def static_pieces(p: Params):
    """Convex pieces of the base and board in world mm (for the simulator)."""
    x0, x1 = p.plate_x; w = p.plate_half_w; T = p.T; L = p.ramp_len
    pc = {}
    pc["plate"] = box(x0, x1 - L, -w, w, 0, T)
    pc["ramp"] = np.array([[x1 - L - 0.5, y, z] for y in (-w, w) for z in (0, T)] + [[x1, y, z] for y in (-w, w) for z in (0, p.ramp_edge)], float)
    bx0 = -p.board_t
    pc["board"] = box(bx0, 0.0, -p.board_half_w, p.board_half_w, T - 0.5, p.board_top)
    pc["socket_f"] = box(0.0, 4.0, -p.board_half_w - 4, p.board_half_w + 4, T - 0.5, T + p.socket_h)
    pc["socket_b"] = box(bx0 - 4.0, bx0, -p.board_half_w - 4, p.board_half_w + 4, T - 0.5, T + p.socket_h)
    return pc


# ------------------------------------------------------------------ simulator scene
GREEN = "0.0 0.68 0.26 1"; LIGHT = "0.90 0.90 0.88 1"


def _mesh_xml(name, P):
    v = " ".join(f"{x/1e3:.6f} {y/1e3:.6f} {z/1e3:.6f}" for x, y, z in np.asarray(P))
    return f'<mesh name="{name}" vertex="{v}"/>'


def build_scene(p: Params, robot_dir, scene_path=None):
    """The dock (convex collision pieces), the floating hood (slides in z and y, rigid in x) and the robot.
    The scene file is written into the robot model's directory (it includes robot_groundcontact.xml from there)."""
    import mujoco
    scene_path = scene_path or os.path.join(robot_dir, "scene_microdock.xml")
    assets, world, hood = [], [], []
    for k, P in static_pieces(p).items():
        assets.append(_mesh_xml(f"dock_{k}", P))
        world.append(f'<geom name="dock_{k}" type="mesh" mesh="dock_{k}" rgba="{LIGHT}" friction="0.7 0.005 0.0001"/>')
    # the tag: visual only (no collisions); renders use the printed tag's meshes from cad.py
    world.append(f'<geom name="tag" type="box" pos="0.0003 0 {p.tag_cz/1e3:.5f}" size="0.0003 {p.tag_size/2e3:.5f} {p.tag_size/2e3:.5f}" '
                 f'rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>')
    for k, P in hood_pieces(p).items():
        assets.append(_mesh_xml(f"hood_{k}", P))
        hood.append(f'<geom name="hood_{k}" type="mesh" mesh="hood_{k}" rgba="{GREEN}" contype="5" conaffinity="5" solref="0.004 1" friction="0.25 0.005 0.0001"/>')
    assets.append(_mesh_xml("hood_adapter", adapter_box(p)))
    hood.append(f'<geom name="hood_adapter" type="mesh" mesh="hood_adapter" rgba="0.15 0.17 0.25 1" contype="5" conaffinity="5" solref="0.004 1" friction="0.3 0.005 0.0001"/>')
    hood.append(f'<site name="adapter_face" pos="{p.adapter_face_x/1e3:.5f} 0 {p.port_z/1e3:.5f}" size="0.001"/>')
    xml = f'''<mujoco model="microdock">
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
    <body name="hood" pos="{p.dock_x/1e3:.5f} 0 {p.z0/1e3:.5f}">
      {f'<joint name="hood_x" type="slide" axis="1 0 0" range="{-p.hood_travel_x/1e3:.4f} 0" stiffness="{p.hood_kx}" damping="0.8"/>' if p.hood_travel_x > 0 else ''}
      <joint name="hood_z" type="slide" axis="0 0 1" range="{-p.hood_drop/1e3:.4f} {p.hood_lift/1e3:.4f}" damping="0.6"/>
      <joint name="hood_y" type="slide" axis="0 1 0" range="{-p.hood_float_y/1e3:.4f} {p.hood_float_y/1e3:.4f}" stiffness="4" damping="3.0"/>
      <inertial pos="-0.055 0 0.035" mass="{p.hood_mass}" diaginertia="1.5e-5 1.5e-5 1.5e-5"/>
      {chr(10).join("      "+h for h in hood)}
    </body>
  </worldbody>
  <contact><exclude body1="world" body2="hood"/></contact>
</mujoco>'''
    with open(scene_path, "w") as f:
        f.write(xml)
    return mujoco.MjSpec.from_file(scene_path)


def modify_robot(spec, p: Params):
    """Recent units' back: battery lowered, the shells cover the back above it (collision stand-in box), magnetic tip in
    the port, power button collar on the top face. Adds trunk-shell collision (own collision bit, so the legs are
    unaffected) for the hood to rest on. Masses and inertias are unchanged."""
    import mujoco
    tb = spec.body("trunk_base")
    for g in tb.geoms:
        if g.meshname == "np_f970":
            g.pos = [g.pos[0], g.pos[1], g.pos[2] - p.bat_drop/1e3]
    shells = [g for g in tb.geoms if g.meshname in ("left_shell", "right_shell")]
    for g in shells:
        c = tb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=g.meshname, pos=list(g.pos), quat=list(g.quat), mass=0.0)
        c.contype = 4; c.conaffinity = 4; c.group = 3; c.friction = [0.25, 0.005, 0.0001]
    x0, x1 = p.back_face_x, -25.0; z0, z1 = p.bat_top, p.bat_top_old + 1.0
    pan = tb.add_geom(name="back_panel", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0)/2e3, 19.6e-3, (z1 - z0)/2e3],
                      pos=[(x0 + x1)/2e3, 0, (z0 + z1)/2e3], rgba=[0.93, 0.92, 0.89, 1], mass=0.0)
    pan.contype = 4; pan.conaffinity = 4; pan.friction = [0.25, 0.005, 0.0001]
    bx, by = p.button_xy
    btn = tb.add_geom(name="power_button", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[(p.button_r + 1.2)/1e3, 2.0/1e3],
                      pos=[bx/1e3, by/1e3, (p.trunk_top + 0.4)/1e3], rgba=[0.95, 0.8, 0.1, 1], mass=0.0)
    btn.contype = 4; btn.conaffinity = 4
    tip = tb.add_geom(name="mag_tip", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.tip_out/2e3, p.tip_w/2e3, p.tip_h/2e3],
                      pos=[(p.shell_back_x - p.tip_out/2)/1e3, 0, p.port_z/1e3], rgba=[0.75, 0.76, 0.78, 1], mass=0.0)
    tip.contype = 4; tip.conaffinity = 4; tip.friction = [0.3, 0.005, 0.0001]
    tb.add_site(name="tip_face", pos=[p.tip_face_x/1e3, 0, p.port_z/1e3], size=[0.001, 0, 0])
    return spec


def tip_error(model, data):
    """Tip face relative to the adapter face, in the hood frame (mm): x = gap (positive = not touching), y, z."""
    import mujoco
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "adapter_face"); t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_face")
    R = data.site_xmat[a].reshape(3, 3)
    return R.T @ (data.site_xpos[t] - data.site_xpos[a]) * 1000


def connected(e):
    """USB power appears: the tip is on the adapter face (within the magnets' self-alignment range)."""
    return bool(-4.0 < e[0] < 2.0 and abs(e[1]) < 2.5 and abs(e[2]) < 2.5)


MAG_RANGE, MAG_LAT, MAG_F, MAG_K = 4.0, 3.5, 4.0, 250.0   # pull-in range (mm), lateral reach (mm), axial pull (N), centring (N/m)


def magnet(model, data):
    """The adapter's magnets: an axial pull plus a soft centring force when the tip is within a few mm (applied forces)."""
    import mujoco
    data.qfrc_applied[:] = 0
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "adapter_face"); t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_face")
    R = data.site_xmat[a].reshape(3, 3); e = R.T @ (data.site_xpos[t] - data.site_xpos[a]) * 1000
    if not (-4.0 < e[0] < MAG_RANGE and abs(e[1]) < MAG_LAT and abs(e[2]) < MAG_LAT): return False
    f_ax = -MAG_F*min(1.0, 1.25 - max(e[0], 0)/MAG_RANGE) if e[0] > -0.3 else 0.0
    f_lat = np.clip(-MAG_K*e[1]/1000, -1.0, 1.0); f_ver = np.clip(-MAG_K*e[2]/1000, -1.0, 1.0)
    F = R @ np.array([f_ax, f_lat, f_ver])           # force on the tip (toward the adapter); the hood gets the opposite
    tb = model.site_bodyid[t]; hb = model.site_bodyid[a]
    mujoco.mj_applyFT(model, data, F, np.zeros(3), data.site_xpos[t], tb, data.qfrc_applied)
    mujoco.mj_applyFT(model, data, -F, np.zeros(3), data.site_xpos[a], hb, data.qfrc_applied)
    return True

"""Visuals shared by the renders: the printable parts from cad.py in place of the simulator's collision pieces, and the
back of recent units on the robot (back_patch.py)."""
import os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "source")); sys.path.insert(0, HERE)
import dock as dk
import cad
import back_patch

HIDDEN = 5                                          # a geom group the default MjvOption doesn't draw (casts no shadow either)
PLA = (0.93, 0.93, 0.91, 1.0)
TAG_BLACK, TAG_WHITE = (0.05, 0.05, 0.05, 1.0), (0.95, 0.95, 0.95, 1.0)
ADAPTER = (0.14, 0.16, 0.23, 1.0)
CABLE = (0.07, 0.07, 0.08, 1.0)
CABLE_R = 0.002                                     # the charging cable's radius (m)
CABLE_X = -0.035                                    # its run along the floor behind the base (world x, m)


def _add_mesh(spec, body, name, man_or_vf, rgba=None, material="", pos=(0, 0, 0), scale=1e-3):
    if isinstance(man_or_vf, tuple):
        v, f = man_or_vf
    else:
        m = man_or_vf.to_mesh(); v, f = np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)
    mesh = spec.add_mesh(name=name)
    mesh.uservert = (np.asarray(v, float)*scale).ravel().tolist(); mesh.userface = np.asarray(f, int).ravel().tolist()
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=list(pos))
    g.contype = 0; g.conaffinity = 0; g.group = 1; g.mass = 0.0
    if material: g.material = material
    else: g.rgba = list(rgba)
    return g


def dress(spec, p: dk.Params, robot_dir, back=True):
    """Hide the simulator's collision pieces and the back's stand-in box; add the printed parts (base, board and tag in
    the world frame, hood and keeper on the hood body) and, if `back`, the newer back on the trunk."""
    for g in spec.geoms:
        if g.name.startswith("dock_") or g.name in ("tag", "back_panel") or (g.name.startswith("hood_") and g.name != "hood_adapter"):
            g.group = HIDDEN
        if g.name == "hood_adapter":
            g.rgba = list(ADAPTER)
    _add_mesh(spec, spec.worldbody, "print_base", cad.base(p), PLA)
    _add_mesh(spec, spec.worldbody, "print_board", cad.board(p), PLA)
    blk, white = cad.tag_insert(p)
    _add_mesh(spec, spec.worldbody, "print_tag_black", blk, TAG_BLACK, pos=(0, 0, p.tag_cz*1e-3))
    _add_mesh(spec, spec.worldbody, "print_tag_white", white, TAG_WHITE, pos=(0, 0, p.tag_cz*1e-3))
    _add_mesh(spec, spec.body("hood"), "print_hood", cad.hood(p), PLA)
    _add_mesh(spec, spec.body("hood"), "print_keeper", cad.keeper(p), PLA)
    if back:
        model = spec.compile()                      # the official shells, as compiled, in the trunk frame
        _add_mesh(spec, spec.body("trunk_base"), "back_patch", back_patch.back_patch(back_patch.shell_meshes(model), p),
                  material="left_shell_material", scale=1.0)
    return spec


def scene(p: dk.Params, robot_dir, back=True):
    return dress(dk.modify_robot(dk.build_scene(p, robot_dir), p), p, robot_dir, back=back)


def cable_points(p: dk.Params, hood_pos, hood_mat, far_y=3.0):
    """The charging cable's route (world, m), as in the README's assembly steps:
    - out of the right-angle head along its axis and through the slot in the hood's left side wall;
    - past the notch in the board's left edge and down behind the board;
    - over the base's rear edge onto the floor, then along the floor toward +y (the slot's side), out of the picture.
    The head rides in the hood, so the first points move with the hood body (hood_pos, hood_mat: its pose)."""
    al, aw, at = p.angle_plug; xe = p.angle_elbow[2]; pz = p.adapter_z
    xa = p.adapter_face_x - p.plug[2] - (aw - at/2)          # the head's body axis, which the cable leaves along (hood frame, mm)
    yo = p.side_in + p.wall_t                                  # the side wall's outer face
    def hood(x, y, z): return hood_pos + hood_mat @ (np.array([x, y, z])*1e-3)
    hy, hz = hood_pos[1]*1e3, hood_pos[2]*1e3
    dz = (p.z0 - dk.Params().z0)*1e-3                          # a lower hood (the seated v2) lowers the run behind the board
    def w(x, y, z): return np.array([x, y, z])
    return [hood(xa, al - xe + 0.2, pz), hood(xa, al - xe + 5.2, pz), hood(xa, yo + 3.0, pz), hood(p.hood_rear + 4.0, yo + 7.0, pz - 1.0),
            w(-0.003, (hy + yo + 9.0)*1e-3, (hz + pz - 2.0)*1e-3), w(-0.013, (hy + yo + 9.0)*1e-3, (hz + pz - 8.0)*1e-3),
            w(-0.014, 0.046, 0.085 + dz), w(-0.016, 0.044, (p.T + 2.0)*1e-3), w((p.plate_x[0] + 7.0)*1e-3, 0.052, (p.T + 2.0)*1e-3),
            w(CABLE_X + 0.001, 0.068, CABLE_R), w(CABLE_X, 0.110, CABLE_R), w(CABLE_X, 0.40, CABLE_R), w(CABLE_X, far_y, CABLE_R)]


def smooth_path(points, step=0.003, max_per_segment=60):
    """A smooth curve through the points (cubic Bezier segments, tangents along the neighbours, handles a third of the
    segment long), sampled about every `step` m."""
    P = [np.asarray(q, float) for q in points]; n = len(P)
    T = [P[1] - P[0]] + [P[i + 1] - P[i - 1] for i in range(1, n - 1)] + [P[-1] - P[-2]]
    T = [t/np.linalg.norm(t) for t in T]
    out = [P[0]]
    for i in range(n - 1):
        L = np.linalg.norm(P[i + 1] - P[i]); c1, c2 = P[i] + T[i]*L/3, P[i + 1] - T[i + 1]*L/3
        for s in np.linspace(0, 1, min(max_per_segment, max(1, int(np.ceil(L/step)))) + 1)[1:]:
            out.append((1 - s)**3*P[i] + 3*(1 - s)**2*s*c1 + 3*(1 - s)*s**2*c2 + s**3*P[i + 1])
    return out


def draw_cable(scn, pts, radius=CABLE_R, rgba=CABLE):
    """Add the cable to a scene already filled by update_scene: capsules along the polyline pts."""
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]; scn.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
        g.specular = 0.2; g.shininess = 0.2                    # braided: a dull sheen

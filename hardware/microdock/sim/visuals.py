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

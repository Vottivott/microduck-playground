"""Visuals of the egg dock for the MuJoCo renders (as the MicroDock's visuals.py): the printed parts in place of the
simulator's collision pieces (the base, the egg's front plate and dome and the tag in the world frame; the hood with
the magnetic adapter, the angle adapter and the cable's plug in it, on the hood body), the back of recent units, and
the white charging cable from the plug's strain relief through the egg's edge clip to the floor."""
import dataclasses, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import swing_trials as st                       # (importing it makes dock.build_scene build the egg dock for a FlexDock)
import flexdock as fd, flexdock_c as fc, strapswing as ss
import cad, visuals, back_patch

dk = st.dk
HIDDEN, PLA, TAG_BLACK, TAG_WHITE, ADAPTER = visuals.HIDDEN, visuals.PLA, visuals.TAG_BLACK, visuals.TAG_WHITE, visuals.ADAPTER
WHITE = (0.93, 0.93, 0.91, 1.0)                  # the cable, its plug and strain relief
CABLE_R = fc.CORD["cable_d"]/2*1e-3              # the cable's radius (m)
CABLE_X = -0.036                                 # its run along the floor behind the base (world x, m)
CLIP_DEPTH = 17.0                                # the egg's depth at its edge clip (mm): front plate and dome


def params(H):
    """The simulator's FlexDock from a run saved by swing_trials.py --save."""
    return fd.FlexDock(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in H["p"].items()})


def cad_params(p):
    """The parameters the printed parts are built with (flexdock_c_build.py): the simulator's, with the magnetic pair
    as it is (fc.PLUG_C). The simulator's adapter sits at the floating stack's front stop instead (face_offset)."""
    return dataclasses.replace(p, **fc.PLUG_C)


def stack(p):
    """The floating stack at its front stop (hood frame): the magnetic adapter and the angle adapter (dark), and the
    cable's plug, strain relief and first 6 mm of cable (white)."""
    pc = cad_params(p); s = fc.goobay_stack(pc, fc.float_x(pc)["face_fwd"])
    y_s = fc.GOOBAY["L"] - fc.GOOBAY["plug_at"]                  # where the angle adapter ends and the plug begins
    return s.trim_by_plane((0, -1, 0), -y_s), s.trim_by_plane((0, 1, 0), y_s)


def dress(spec, p, back=True):
    """Hide the simulator's collision pieces and the back's stand-in box; add the printed parts and, if `back`, the
    newer back on the trunk."""
    pc = cad_params(p)
    for g in spec.geoms:
        if g.name.startswith(("dock_", "guide_", "hood_")) or g.name in ("tag", "back_panel"):
            g.group = HIDDEN
    w = spec.worldbody
    visuals._add_mesh(spec, w, "print_base", ss.base_cad(pc), PLA)
    visuals._add_mesh(spec, w, "print_front", fc.front_plate_c(pc), PLA)
    visuals._add_mesh(spec, w, "print_dome", fc.dome_c(pc), PLA)
    blk, white = cad.tag_insert(pc)
    visuals._add_mesh(spec, w, "print_tag_black", blk, TAG_BLACK, pos=(0, 0, ss.EGG_TAG_CZ*1e-3))
    visuals._add_mesh(spec, w, "print_tag_white", white, TAG_WHITE, pos=(0, 0, ss.EGG_TAG_CZ*1e-3))
    hb = spec.body("hood"); dark, plug = stack(p)
    visuals._add_mesh(spec, hb, "print_hood", fc.hood_c_cad(pc), PLA)
    visuals._add_mesh(spec, hb, "adapters", dark, ADAPTER)
    visuals._add_mesh(spec, hb, "cable_plug", plug, WHITE)
    if back:
        model = spec.compile()                  # the official shells, as compiled, in the trunk frame
        visuals._add_mesh(spec, spec.body("trunk_base"), "back_patch", back_patch.back_patch(back_patch.shell_meshes(model), p),
                          material="left_shell_material", scale=1.0)
    return spec


def scene(p, robot_dir, back=True):
    return dress(dk.modify_robot(dk.build_scene(p, robot_dir), p), p, back=back)


def cable_points(p, hood_pos, hood_mat, far_y=3.0):
    """The charging cable's route (world, m), as in the README's assembly steps:
    - out of the plug's strain relief along the hood's +y (through its left wall);
    - in a loose loop down to the clip in the egg's edge, through the front plate and the dome;
    - down beside the egg onto the base, over its rear corner onto the floor, then along the floor toward +y, out of
      the picture.
    The plug rides in the hood, so the first points move with the hood body (hood_pos, hood_mat: its pose)."""
    R = np.asarray(hood_mat); out_d = R[:, 1]; dn = np.array([0.0, 0.0, -1.0])
    cs = hood_pos + R @ (np.array(p.cable_out)*1e-3)
    def w(x, y, z): return np.array([x, y, z])
    yk, zk = (v*1e-3 for v in ss.egg_clip(p)); bt = CLIP_DEPTH*1e-3; top = p.T*1e-3 + CABLE_R + 0.0004   # on the base
    return [cs, cs + out_d*0.006, cs + out_d*0.015 + dn*0.006, w(0.016, yk - 0.006, zk + 0.020), w(0.006, yk, zk + 0.003),
            w(0.0, yk, zk), w(-bt, yk, zk), w(-bt - 0.005, yk, zk - 0.005), w(-bt - 0.007, yk - 0.002, (p.T + p.socket_h + 3.0)*1e-3),
            w(-bt - 0.008, yk + 0.002, top), w(-0.026, 0.088, top), w(-0.031, 0.097, top - 0.0004),
            w(CABLE_X, 0.106, CABLE_R), w(CABLE_X, 0.140, CABLE_R), w(CABLE_X, 0.40, CABLE_R), w(CABLE_X, far_y, CABLE_R)]


def draw_cable(scn, p, hood_pos, hood_mat):
    visuals.draw_cable(scn, visuals.smooth_path(cable_points(p, hood_pos, hood_mat)), radius=CABLE_R, rgba=WHITE)

"""Studio gallery of the MicroDock (the studio of scripts/render_hardware_gallery.py): the assembled dock, its printed
parts pulled apart, and Microduck docked (the last frame of a run saved by docking.py --save).

  export MICRODUCK_RL_DIR=/path/to/microduck_rl
  MUJOCO_GL=egl uv run --with manifold3d python hardware/microdock/sim/render_gallery.py run.pkl hardware/microdock/renders
"""
import os, pickle, sys
import numpy as np
import mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import visuals
from visuals import cad, dk

STUDIO = (0.84, 0.87, 0.91, 1.0)
FLOOR = (0.69, 0.74, 0.80, 1.0)


def add_studio(spec, floor_z=0.0):
    """The neutral three-wall studio and soft shadows of scripts/render_hardware_gallery.py."""
    fm = spec.add_material(name="gallery_floor_material", rgba=FLOOR, roughness=0.82, specular=0.08)
    wm = spec.add_material(name="gallery_wall_material", rgba=STUDIO, roughness=0.9, specular=0.02)
    spec.worldbody.add_geom(name="gallery_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=(0.0, 0.0, floor_z), size=(4.0, 4.0, 0.05),
                            material=fm.name, contype=0, conaffinity=0)
    for name, pos, size in (("gallery_wall_x_pos", (2.5, 0.0, 1.3), (0.03, 2.5, 1.3)), ("gallery_wall_x_neg", (-2.5, 0.0, 1.3), (0.03, 2.5, 1.3)),
                            ("gallery_wall_y_pos", (0.0, 2.5, 1.3), (2.5, 0.03, 1.3)), ("gallery_wall_y_neg", (0.0, -2.5, 1.3), (2.5, 0.03, 1.3))):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, material=wm.name, contype=0, conaffinity=0)
    key = spec.worldbody.add_light(name="gallery_key")
    key.pos = (1.1, -0.9, 1.8); key.dir = (-0.45, 0.30, -1.0); key.type = mujoco.mjtLightType.mjLIGHT_SPOT
    key.castshadow = True; key.diffuse = (0.82, 0.84, 0.88); key.specular = (0.18, 0.18, 0.18)
    fill = spec.worldbody.add_light(name="gallery_fill")
    fill.pos = (-1.2, 0.8, 1.1); fill.dir = (0.55, -0.35, -0.7); fill.type = mujoco.mjtLightType.mjLIGHT_SPOT
    fill.castshadow = True; fill.diffuse = (0.46, 0.48, 0.52); fill.specular = (0.06, 0.06, 0.06)
    spec.visual.global_.offwidth = 1400; spec.visual.global_.offheight = 1050
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = (0.32, 0.32, 0.34); spec.visual.headlight.diffuse = (0.38, 0.38, 0.40); spec.visual.headlight.specular = (0.08, 0.08, 0.08)
    spec.visual.map.znear = 0.002; spec.visual.map.shadowclip = 3.0


def render(model, data, path, lookat, distance, azimuth, elevation):
    r = mujoco.Renderer(model, height=900, width=1200)
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt); opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
    r.update_scene(data, camera=cam, scene_option=opt); Image.fromarray(r.render()).save(path); r.close()
    print("wrote", path)


def parts_scene(p: dk.Params, explode=0.0):
    """Only the printed parts (and the adapter in the hood), at their assembled poses with the hood resting in its
    V-seats; `explode` pulls them apart (board out of its socket, tag, hood and keeper off the board)."""
    spec = mujoco.MjSpec(); add_studio(spec)
    lift = 55.0*explode                                  # board (and everything on it) out of the socket, mm
    hood_pos = np.array([p.dock_x, 0.0, p.z0 - p.hood_drop + lift])
    visuals._add_mesh(spec, spec.worldbody, "base", cad.base(p), visuals.PLA)
    visuals._add_mesh(spec, spec.worldbody, "board", cad.board(p), visuals.PLA, pos=(0, 0, lift*1e-3))
    blk, white = cad.tag_insert(p)
    for name, man, rgba in (("tag_black", blk, visuals.TAG_BLACK), ("tag_white", white, visuals.TAG_WHITE)):
        visuals._add_mesh(spec, spec.worldbody, name, man, rgba, pos=(35e-3*explode, 0, (p.tag_cz + lift)*1e-3))
    visuals._add_mesh(spec, spec.worldbody, "hood", cad.hood(p), visuals.PLA, pos=tuple((hood_pos + [70.0*explode, 0, 0])*1e-3))
    visuals._add_mesh(spec, spec.worldbody, "keeper", cad.keeper(p), visuals.PLA, pos=tuple((hood_pos + [-45.0*explode, 0, 0])*1e-3))
    if not explode:
        visuals._add_mesh(spec, spec.worldbody, "adapter", (dk.adapter_box(p), np.zeros((0, 3), int)),   # no faces: MuJoCo uses the hull
                          visuals.ADAPTER, pos=tuple(hood_pos*1e-3))
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    return model, data


def docked_scene(p: dk.Params, robot_dir, qpos):
    spec = visuals.scene(p, robot_dir)
    for g in spec.geoms:
        if g.name == "floor": g.group = visuals.HIDDEN          # the studio floor replaces the simulator's
    for l in spec.lights:
        l.active = False
    add_studio(spec)
    model = spec.compile(); data = mujoco.MjData(model)
    data.qpos[:] = qpos; mujoco.mj_forward(model, data)
    return model, data


if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit(__doc__)
    run, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    with open(run, "rb") as f:
        H = pickle.load(f)
    p = dk.Params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in H["p"].items()})
    m, d = parts_scene(p)
    render(m, d, os.path.join(out, "microdock_three_quarter.png"), (0.03, 0.0, 0.12), 0.48, 220.0, -20.0)
    m, d = parts_scene(p, explode=1.0)
    render(m, d, os.path.join(out, "microdock_parts.png"), (0.05, 0.0, 0.15), 0.58, 238.0, -14.0)
    robot_dir = os.path.join(os.environ.get("MICRODUCK_RL_DIR", ""), "src", "mjlab_microduck", "robot", "microduck")
    m, d = docked_scene(p, robot_dir, H["qpos"][-1])
    render(m, d, os.path.join(out, "microdock_docked.png"), (0.05, 0.0, 0.13), 0.50, 242.0, -12.0)

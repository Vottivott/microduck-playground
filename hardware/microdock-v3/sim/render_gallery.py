"""Studio gallery of the MicroDock v3 (the studio of the MicroDock's render_gallery.py): the assembled egg dock, its
printed parts pulled apart, and Microduck docked (the middle of the docked rest of a run saved by swing_trials.py
--save).

  export MICRODUCK_RL_DIR=/path/to/microduck_rl
  MUJOCO_GL=egl uv run --with manifold3d --with trimesh python hardware/microdock-v3/sim/render_gallery.py run.pkl hardware/microdock-v3/renders
"""
import importlib.util, os, pickle, sys
import numpy as np
import mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import egg_visuals as ev
from egg_visuals import fc, ss, cad, visuals

_g = importlib.util.spec_from_file_location("microdock_gallery", os.path.join(os.path.dirname(os.path.dirname(HERE)), "microdock", "sim", "render_gallery.py"))
g1 = importlib.util.module_from_spec(_g); _g.loader.exec_module(g1)            # the MicroDock's studio
LIFT = 40.0                                                                     # the egg out of the base's socket (mm)
# the parts pulled apart along the assembly's axis (world mm, with the lift): the hood and the tag forward, the spring
# stage and the dome back
EXPLODE = {"front": (0.0, 0.0), "tag": (45.0, 0.0), "hood": (115.0, 0.0), "stage": (-115.0, 0.0), "dome": (-235.0, 0.0)}


def parts_scene(p, explode=0.0):
    """Only the printed parts (and the adapters and the cable's plug in the hood), at their assembled poses with the
    hood at rest; `explode` pulls them apart (EXPLODE)."""
    pc = ev.cad_params(p); spec = mujoco.MjSpec(); g1.add_studio(spec)
    def at(name): dx, dy = EXPLODE[name]; return np.array([dx*explode, dy*explode, LIFT*explode])
    W = spec.worldbody; PLA = visuals.PLA
    visuals._add_mesh(spec, W, "base", ss.base_cad(pc), PLA)
    visuals._add_mesh(spec, W, "front", fc.front_plate_c(pc), PLA, pos=tuple(at("front")*1e-3))
    visuals._add_mesh(spec, W, "dome", fc.dome_c(pc), PLA, pos=tuple(at("dome")*1e-3))
    for k, m in fc.stage_c_parts(pc, 0.0, 0.0).items():
        visuals._add_mesh(spec, W, "stage_" + k, (np.asarray(m.vertices), np.asarray(m.faces)), PLA, pos=tuple(at("stage")*1e-3))
    blk, white = cad.tag_insert(pc)
    for name, man, rgba in (("tag_black", blk, visuals.TAG_BLACK), ("tag_white", white, visuals.TAG_WHITE)):
        visuals._add_mesh(spec, W, name, man, rgba, pos=tuple((at("tag") + [0, 0, ss.EGG_TAG_CZ])*1e-3))
    hp = tuple((p.hood_home() + at("hood"))*1e-3); dark, plug = ev.stack(p)
    visuals._add_mesh(spec, W, "hood", fc.hood_c_cad(pc), PLA, pos=hp)
    visuals._add_mesh(spec, W, "adapters", dark, visuals.ADAPTER, pos=hp)
    visuals._add_mesh(spec, W, "cable_plug", plug, ev.WHITE, pos=hp)
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    return model, data


def docked_scene(p, robot_dir, qpos):
    spec = ev.scene(p, robot_dir)
    for g in spec.geoms:
        if g.name == "floor": g.group = visuals.HIDDEN          # the studio floor replaces the simulator's
    for l in spec.lights:
        l.active = False
    g1.add_studio(spec)
    model = spec.compile(); data = mujoco.MjData(model)
    data.qpos[:] = qpos; mujoco.mj_forward(model, data)
    return model, data


def render(model, data, path, lookat, distance, azimuth, elevation, cable=None):
    """As the MicroDock's render(), with the charging cable (cable: the FlexDock) drawn from the hood's pose."""
    r = mujoco.Renderer(model, height=900, width=1200)
    opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt); opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
    r.update_scene(data, camera=cam, scene_option=opt)
    if cable is not None:
        hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
        ev.draw_cable(r.scene, cable, data.xpos[hb], data.xmat[hb].reshape(3, 3))
    Image.fromarray(r.render()).save(path); r.close()
    print("wrote", path)


if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit(__doc__)
    run, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    with open(run, "rb") as f:
        H = pickle.load(f)
    p = ev.params(H)
    m, d = parts_scene(p)
    render(m, d, os.path.join(out, "microdock_three_quarter.png"), (0.03, 0.0, 0.13), 0.62, 220.0, -20.0)
    m, d = parts_scene(p, explode=1.0)
    render(m, d, os.path.join(out, "microdock_parts.png"), (-0.045, 0.0, 0.15), 0.74, 238.0, -14.0)
    robot_dir = os.path.join(os.environ.get("MICRODUCK_RL_DIR", ""), "src", "mjlab_microduck", "robot", "microduck")
    Q = H["qpos"]; docked = np.where(Q[:, 0] < Q[:, 0].min() + 0.001)[0]      # the docked rest (the root's closest)
    m, d = docked_scene(p, robot_dir, Q[int(docked.mean())])
    render(m, d, os.path.join(out, "microdock_docked.png"), (0.06, 0.0, 0.14), 0.66, 242.0, -12.0, cable=p)

"""MuJoCo preview of a docking run, in the scene style of the playground's other previews: mjlab's checker floor and
visual defaults, one shadow-casting sun tilted toward the camera (as scripts/render_checkpoint.py), 50 fps, square.
Replays a run saved by docking.py --save, with the printable parts as the dock's visuals, the back of recent units and
the charging cable, which leaves the hood and runs along the floor out of the picture.

  export MICRODUCK_RL_DIR=/path/to/microduck_rl
  MUJOCO_GL=egl uv run --with manifold3d python hardware/microdock/sim/render_preview.py run.pkl preview.mp4
writes preview.mp4 (every simulation step, 50 fps, 1080 x 1080) and preview.gif (320 px, every 8th step at 125 ms,
ending 1.5 s after docking, or with --gif_to end at the end of the run). MUJOCO_GL=osmesa renders without a GPU.
"""
import argparse, math, os, pickle, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import visuals

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--size", type=int, default=1080); ap.add_argument("--frames", default="", help="comma list: write these frames as PNGs instead")
ap.add_argument("--azimuth", type=float, default=225.0); ap.add_argument("--elevation", type=float, default=-18.0)
# camera: wide on the approach (both the duck and the dock in view), easing in to the docked duck as it backs in
ap.add_argument("--start", default="0.19,-0.02,0.09,0.90", help="lookat x,y,z and distance at the start (m)")
ap.add_argument("--end", default="0.06,0.0,0.11,0.62", help="... and once docked")
ap.add_argument("--gif_to", default="docked", choices=["docked", "end"], help="the gif ends 1.5 s after docking, or at the run's end")
a = ap.parse_args()
RL = os.environ.get("MICRODUCK_RL_DIR", "")
ROBOT_DIR = os.path.join(RL, "src", "mjlab_microduck", "robot", "microduck")
if not os.path.isdir(ROBOT_DIR): sys.exit("set MICRODUCK_RL_DIR to the microduck_rl checkout the run was made with")

with open(a.run, "rb") as f:
    H = pickle.load(f)
p = visuals.dk.Params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in H["p"].items()})
spec = visuals.scene(p, ROBOT_DIR)

# ---- mjlab's look: scene.xml visual defaults and the terrain's groundplane (terrain_entity.py)
spec.visual.headlight.diffuse = [0.6]*3; spec.visual.headlight.ambient = [0.3]*3; spec.visual.headlight.specular = [0.0]*3
spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
spec.visual.global_.offwidth = a.size; spec.visual.global_.offheight = a.size
for t in spec.textures:
    if t.name == "floor":
        t.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER; t.mark = mujoco.mjtMark.mjMARK_EDGE
        t.rgb1 = [0.2, 0.3, 0.4]; t.rgb2 = [0.1, 0.2, 0.3]; t.markrgb = [0.8, 0.8, 0.8]; t.width = 300; t.height = 300
for m in spec.materials:
    if m.name == "floor": m.texrepeat = [4.0, 4.0]; m.texuniform = True; m.reflectance = 0.2
# the sun: directional, casting shadows, tilted toward the camera (render_checkpoint.py's light_dir="auto", tilt 0.43)
az = math.radians(a.azimuth); sun = np.array([0.43*math.cos(az), 0.43*math.sin(az), -1.0]); sun /= np.linalg.norm(sun)
for l in spec.lights:
    l.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL; l.castshadow = True; l.dir = sun.tolist(); l.pos = [0, 0, 1.5]
    l.diffuse = [0.7]*3; l.specular = [0.3]*3; l.ambient = [0.0]*3

model = spec.compile(); data = mujoco.MjData(model)
HOOD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
# Shadow map: 8192 texels as in render_checkpoint.py, but over a box sized to this small static scene (its +-16 m box
# gives ~4 mm texels: blocky shadows and acne on the dock's large flat faces). The box half-size is shadowclip*extent,
# fixed when the renderer's context is made. The far clip (zfar*extent) stays at the playground renders' 200 m (50 x 4).
SHADOW_BOX = 0.8
model.vis.quality.shadowsize = 8192; model.vis.map.shadowscale = 1.0; model.vis.map.shadowclip = 1.0
model.stat.extent = SHADOW_BOX; model.vis.map.zfar = 200.0/SHADOW_BOX
Q = H["qpos"]; n = len(Q)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.trackbodyid = -1; cam.fixedcamid = -1
cam.azimuth = a.azimuth; cam.elevation = a.elevation
# progress toward the dock from the duck's smoothed root x (Gaussian, sigma 1 s), so the camera never picks up the gait sway
k = np.exp(-0.5*(np.arange(-150, 151)/50.0)**2); k /= k.sum()
sx = np.convolve(np.pad(Q[:, 0], 150, mode="edge"), k, mode="valid")
u = np.clip((sx[0] - sx)/max(1e-6, sx[0] - sx.min()), 0, 1); u = u*u*(3 - 2*u)
C0, C1 = (np.array([float(v) for v in t.split(",")]) for t in (a.start, a.end))
def place(i):
    c = C0*(1 - u[i]) + C1*u[i]; cam.lookat[:] = c[:3]; cam.distance = c[3]

opt = mujoco.MjvOption()
r = mujoco.Renderer(model, height=a.size, width=a.size)
idx = [int(v) for v in a.frames.split(",")] if a.frames else list(range(n))
frames = []
for i in idx:
    data.qpos[:] = Q[i]; mujoco.mj_forward(model, data); place(i)
    r.update_scene(data, camera=cam, scene_option=opt)
    visuals.draw_cable(r.scene, visuals.smooth_path(visuals.cable_points(p, data.xpos[HOOD], data.xmat[HOOD].reshape(3, 3))))
    frames.append(r.render().copy())
from PIL import Image
stem = os.path.splitext(a.out)[0]
if a.frames:
    for i, f in zip(idx, frames): Image.fromarray(f).save(f"{stem}_{i:04d}.png")
    print("frames", idx)
else:
    import imageio
    imageio.mimsave(a.out, frames, fps=50, codec="libx264", quality=None, macro_block_size=8, ffmpeg_params=["-crf", "24", "-preset", "slow", "-movflags", "+faststart"])
    ph = H["log"]["phase"]; docked = ph.index("docked") if "docked" in ph else n
    g = [Image.fromarray(f).resize((320, 320), Image.LANCZOS) for f in frames[:n if a.gif_to == "end" else min(n, docked + 75)][::8]]
    g[0].save(stem + ".gif", save_all=True, append_images=g[1:], duration=125, loop=0, optimize=True)
    print("wrote", a.out, len(frames), "frames,", stem + ".gif", len(g), "frames")

"""Butt-walk command battery in the CPU MuJoCo docking scene (BAM actuators, the runtime's observation pipeline): for each
command, stand on the plate, sit with the official sitstand policy, hand over to the butt walk, run the command and
report what the trunk did. The sim-to-sim check before any docking trial.

  export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
  uv run python hardware/microdock-v2/sim/bw_eval.py hardware/microdock-v2/policy/butt_walk.onnx [--seconds 4] [--json out.json]
"""
import argparse, contextlib, io, json, math, os, sys
import numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import docking_seated as ds       # also puts the MicroDock's simulation on the path
dg = ds.dg; dk = dg.dk

ap = argparse.ArgumentParser()
ap.add_argument("policy"); ap.add_argument("--seconds", type=float, default=4.0); ap.add_argument("--seat_mu", type=float, default=ds.SEAT_MU)
ap.add_argument("--json", default="")
a = ap.parse_args()
import onnxruntime as ort
p = dk.Params()
model, data, bc, pol = dg.make(p); ds.seat_friction(model, a.seat_mu)
sess = ort.InferenceSession(a.policy)
tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
HEADG = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.body(model.geom_bodyid[g]).name in ("neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")]
NECK = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")]
def head_low():
    z = 1e9
    for g in HEADG:
        mid = model.geom_dataid[g]; a0 = model.mesh_vertadr[mid]; n = model.mesh_vertnum[mid]
        z = min(z, float((data.geom_xpos[g] + model.mesh_vert[a0:a0+n] @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min()))
    return z*1e3
def tilt():
    R = data.xmat[tb].reshape(3, 3); g = R.T @ np.array([0, 0, -1.0])
    return math.degrees(math.atan2(-g[0], -g[2])), math.degrees(math.atan2(g[1], -g[2]))
def step(n, vx=0.0, wz=0.0):
    for _ in range(n):
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, 0, wz); dg.step_control(model, data, bc, pol)
CMDS = [(-0.08, 0.0), (-0.05, 0.0), (-0.03, 0.0), (0.0, 0.0), (0.05, 0.0), (-0.05, 0.2), (-0.05, -0.2), (0.0, 0.3)]
out = []
for vx, wz in CMDS:
    with contextlib.redirect_stdout(io.StringIO()):
        dg.reset(model, data, bc, pol, x=0.30, y=0.0, yaw=0.0, z=0.125)
        pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = pol.walking_session
    step(50)
    with contextlib.redirect_stdout(io.StringIO()): pol.toggle_sit()
    step(200)
    pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = sess
    step(25)
    x0, y0, yaw0 = data.qpos[0], data.qpos[1], dg.yaw_deg(data); Z, PI, RO, HL, NK = [], [], [], [], []
    n = int(a.seconds*50)
    for _ in range(n):
        step(1, vx, wz); Z.append(data.qpos[2]*1e3); pi, ro = tilt(); PI.append(pi); RO.append(ro)
        if _ % 5 == 0: HL.append(head_low()); NK.append(data.qpos[NECK])
    T = n/50; c, s_ = math.cos(math.radians(yaw0)), math.sin(math.radians(yaw0))
    dx, dy = data.qpos[0] - x0, data.qpos[1] - y0
    fwd, side = dx*c + dy*s_, -dx*s_ + dy*c
    r = dict(cmd=[vx, wz], vx=round(fwd/T, 4), vy=round(side/T, 4), wz=round(math.radians(dg.wrap(dg.yaw_deg(data) - yaw0))/T, 3),
             z_mm=[round(float(np.mean(Z)), 1), round(float(np.min(Z)), 1), round(float(np.max(Z)), 1)],
             pitch=[round(float(np.mean(PI)), 1), round(float(np.min(PI)), 1), round(float(np.max(PI)), 1)],
             roll=round(float(np.mean(RO)), 1), fell=bool(abs(PI[-1]) > 45 or abs(RO[-1]) > 45),
             head_low_mm=round(float(np.min(HL)), 1), neck_pitch=round(float(np.mean(NK)), 3))
    out.append(r)
    print(f"cmd vx {vx:+.2f} wz {wz:+.2f} -> vx {r['vx']:+.4f} vy {r['vy']:+.4f} wz {r['wz']:+.3f}; trunk z {r['z_mm']} pitch {r['pitch']} roll {r['roll']:+.1f} head low {r['head_low_mm']:.0f} mm neck {r['neck_pitch']:+.2f}{' FELL' if r['fell'] else ''}", flush=True)
if a.json:
    json.dump(dict(policy=os.path.basename(a.policy), seat_mu=a.seat_mu, seconds=a.seconds, results=out), open(a.json, "w"), indent=1)

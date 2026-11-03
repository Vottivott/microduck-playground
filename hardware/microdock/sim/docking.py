"""MicroDock docking trials in MuJoCo: Microduck walks toward the dock's tag (detection emulated as truth + noise),
turns in place, backs in until the magnetic tip in its USB-C port meets the adapter in the floating hood, stops when
USB power appears and stands. Walking is the official `velstand.onnx` policy on the BAM XL330 actuator model.

Needs a checkout of pollen-robotics/microduck_rl (develop, validated at bae39f59: its scripts/infer_policy.py and robot
model) and the official policies (huggingface.co/pollen-robotics/microduck-policies: velstand.onnx, alpha_sitstand.onnx):

  export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
  uv run python hardware/microdock/sim/docking.py --seeds 41,42,43 --n 10 --stop_gap 0 --out results.json
  uv run python hardware/microdock/sim/docking.py --n 1 --seed 0 --stop_gap 0 --noise 0 --start 450,15,178 --hold 4 --save run.pkl

The scene file is written into the robot model's directory (scene_microdock.xml), next to the robot it includes.
"""
import argparse, contextlib, io, json, math, os, pickle, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "source"))
import dock as dk

RL = os.environ.get("MICRODUCK_RL_DIR", "")
POL = os.environ.get("MICRODUCK_POLICY_DIR", "")
ROBOT_DIR = os.path.join(RL, "src", "mjlab_microduck", "robot", "microduck")


def _infer_policy():
    if not (RL and os.path.exists(os.path.join(RL, "scripts", "infer_policy.py"))):
        sys.exit("set MICRODUCK_RL_DIR to a checkout of pollen-robotics/microduck_rl (develop, validated at bae39f59)")
    sys.path.insert(0, os.path.join(RL, "scripts"))
    import infer_policy
    return infer_policy


def make(p: dk.Params, vin=7.6):
    """The dock scene with the robot on BAM actuators and the official walk/stand policies."""
    ip = _infer_policy()
    if not all(os.path.exists(os.path.join(POL, f)) for f in ("velstand.onnx", "alpha_sitstand.onnx")):
        sys.exit("set MICRODUCK_POLICY_DIR to a folder with velstand.onnx and alpha_sitstand.onnx (pollen-robotics/microduck-policies)")
    spec = dk.modify_robot(dk.build_scene(p, ROBOT_DIR), p)
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, vin, 1.75)
    from bam.mujoco import MujocoController
    kt = bam_model.kt.value; R = bam_model.R.value; force_limit = bam_model.actuator.vin * kt / R
    names = []
    for act in spec.actuators:                     # as bam.mjlab.BamActuator: position actuators -> voltage-bounded motors
        tgt = act.target; tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"): continue
        act.set_to_motor(); act.forcelimited = True; act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False; act.gear = [1.0, 0, 0, 0, 0, 0]; names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1)); joint.frictionloss = 0.0
                joint.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION; joint.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
                break
    model = spec.compile(); model.opt.timestep = 0.005; data = mujoco.MjData(model)
    bam_ctrl = MujocoController(bam_model, names, model, data, vin_drop_gain=None, vin_min=ip.BAM_VIN_MIN)
    with contextlib.redirect_stdout(io.StringIO()):
        pol = ip.PolicyInference(model, data, walking_onnx_path=os.path.join(POL, "velstand.onnx"), action_scale=1.0,
                                 bam_ctrl=bam_ctrl, new_cmd_obs=True, use_projected_gravity=True,
                                 sitstand_onnx_path=os.path.join(POL, "alpha_sitstand.onnx"))
    return model, data, bam_ctrl, pol


def reset(model, data, bam_ctrl, pol, x=0.0, y=0.0, yaw=0.0, z=0.125):
    mujoco.mj_resetData(model, data)
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"); a = model.jnt_qposadr[j]
    data.qpos[a:a+3] = [x, y, z]
    data.qpos[a+3:a+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    for i, qi in enumerate(pol.joint_qpos_indices): data.qpos[qi] = pol.default_pose[i]
    bam_ctrl.reset(data.qpos); pol.set_position_targets(pol.default_pose); mujoco.mj_forward(model, data)
    pol.last_action[:] = 0


def step_control(model, data, bam_ctrl, pol, frame_cb=None):
    """One 50 Hz control step (4 x 5 ms physics)."""
    pol.update_behavior(0.02)
    pol.apply_action(pol.infer())
    for k in range(4):
        bam_ctrl.update(); mujoco.mj_step(model, data)
        if frame_cb: frame_cb(k)


def yaw_deg(data):
    q = data.qpos[3:7]; R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))
def wrap(a): return (a + 180) % 360 - 180


def run(model, data, bc, pol, p, a, x0, y0, yaw0, nx, ny, nyaw, rec=None):
    """One docking attempt from (x0 mm, y0 mm, yaw0 deg); nx, ny, nyaw: the tag estimate's error (mm, mm, deg)."""
    with contextlib.redirect_stdout(io.StringIO()):
        reset(model, data, bc, pol, x=x0/1000, y=y0/1000, yaw=math.radians(yaw0), z=0.125)
        pol.sit_mode = False; pol.current_policy = 'walking'; pol.ort_session = pol.walking_session
    log = {"phase": [], "err": [], "hood": []}
    hz = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_z")]
    hy = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_y")]
    btn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "power_button"); hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
    pressed = [0]
    def st(phase, vx, vy, wz):
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, vy, wz); step_control(model, data, bc, pol, frame_cb=lambda k: dk.magnet(model, data))
        for ci in range(data.ncon):
            c = data.contact[ci]
            if btn in (c.geom1, c.geom2) and hb in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]): pressed[0] += 1; break
        e = dk.tip_error(model, data)
        log["phase"].append(phase); log["err"].append(e.tolist()); log["hood"].append([data.qpos[hz]*1e3, data.qpos[hy]*1e3])
        if rec: rec(phase)
        return e
    X = lambda: data.qpos[0]*1000; Y = lambda: data.qpos[1]*1000
    for _ in range(40): st("look", 0, 0, 0)
    # approach: steer onto the dock axis (tag estimate = truth + noise), stop at the turn point
    for _ in range(1500):
        des = math.degrees(math.atan2(-(Y() + ny - a.yaim), -150.0)); e = wrap(des - (yaw_deg(data) + nyaw))
        st("approach", 0.3, 0, float(np.clip(0.035*e, -0.5, 0.5)))
        if X() + nx <= a.xturn: break
    for _ in range(20): st("stop", 0, 0, 0)
    for _ in range(600):
        e = wrap(a.yawt - (yaw_deg(data) + nyaw))
        if abs(e) < 10: break
        st("turn", 0, 0, 1.0 if e > 0 else -1.0)
    for _ in range(250):
        e = wrap(a.yawt - (yaw_deg(data) + nyaw))
        if abs(e) < 1.0: break
        st("turn", 0, 0, float(np.sign(e)*(0.6 if abs(e) > 4 else 0.45)))
    for _ in range(25): st("settle", 0, 0, 0)
    after_turn = (X(), Y(), yaw_deg(data))
    # back in with heading hold until the tip meets the adapter
    con_step = None
    for k in range(700):
        e_yaw = wrap(a.yawt - (yaw_deg(data) + nyaw))
        err = st("backin", a.vback, 0, float(np.clip(0.04*e_yaw, -0.4, 0.4)))
        if con_step is None and dk.connected(err): con_step = len(log["phase"])
        if a.stop_gap > 0 and err[0] < a.stop_gap: break        # (oracle) stop commanding before contact
        if a.stop_gap <= 0 and dk.connected(err): break         # realistic: stop when USB power appears (tip on the adapter)
        if X() < p.dock_x - 15: break
    for _ in range(int(a.hold*50)): err = st("docked", a.vhold, 0, 0)
    ok = dk.connected(err)
    jx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_x")
    return dict(ok=ok, button_contacts=pressed[0], err=np.array(err).round(2).tolist(), after_turn=[round(v, 1) for v in after_turn], con_step=con_step,
                hood=[round(v, 2) for v in log["hood"][-1]], hood_x=(round(data.qpos[model.jnt_qposadr[jx]]*1e3, 1) if jx >= 0 else 0.0),
                trunk_z=round(data.qpos[2]*1000, 1), log=log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=6, help="trials per seed")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", default="", help="comma list; overrides --seed")
    ap.add_argument("--xturn", type=float, default=240.5, help="turn point: trunk x (mm from the board) where the duck stops and turns")
    ap.add_argument("--yaim", type=float, default=-12.0, help="aim this far to one side (mm): the turn shifts the duck back by about as much")
    ap.add_argument("--yawt", type=float, default=0.0); ap.add_argument("--vback", type=float, default=-0.4)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds standing docked at the end")
    ap.add_argument("--save", default="", help="pickle the first trial's qpos at 50 Hz (for render_preview.py)")
    ap.add_argument("--out", default="", help="write the results (and a summary) as JSON")
    ap.add_argument("--set", nargs="*", default=[], help="override Params fields, e.g. --set hood_float_y=10")
    ap.add_argument("--noise", type=float, default=1.0, help="scale of the tag estimate's error (sigma 5 mm along, 3 mm across, 1 deg)")
    ap.add_argument("--start", default="", help="x,y,yaw start (mm, mm, deg) instead of a random one")
    ap.add_argument("--vhold", type=float, default=0.0)
    ap.add_argument("--stop_gap", type=float, default=14.0, help="0: stop walking when USB power appears (realistic)")
    a = ap.parse_args()
    p = dk.Params()
    for kv in a.set:
        k, v = kv.split("="); setattr(p, k, type(getattr(p, k))(float(v)) if not isinstance(getattr(p, k), tuple) else tuple(map(float, v.split(","))))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    res = []
    for seed in seeds:
        model, data, bc, pol = make(p)                # fresh per seed: each seed's trials are independent of the others
        rng = np.random.default_rng(seed)
        for t in range(a.n):
            x0, y0, yaw0 = rng.uniform(380, 460), rng.uniform(-40, 40), 180 + rng.uniform(-12, 12)
            if a.start: x0, y0, yaw0 = map(float, a.start.split(","))
            nx, ny, nyaw = rng.normal(0, 5)*a.noise, rng.normal(0, 3)*a.noise, rng.normal(0, 1)*a.noise
            frames = []
            save_this = bool(a.save) and t == 0 and seed == seeds[0]
            r = run(model, data, bc, pol, p, a, x0, y0, yaw0, nx, ny, nyaw, rec=(lambda ph: frames.append(data.qpos.copy())) if save_this else None)
            r.update(seed=seed, start=[round(x0), round(y0), round(yaw0)], noise=[round(nx, 1), round(ny, 1), round(nyaw, 2)])
            print(f"seed {seed} #{t} start ({x0:.0f},{y0:+.0f},{yaw0:.0f}) turn@({r['after_turn'][0]:.0f},{r['after_turn'][1]:+.0f}) yaw {r['after_turn'][2]:+.1f} "
                  f"-> err {r['err']} btn {r['button_contacts']} hood z/y {r['hood'][0]:+.2f}/{r['hood'][1]:+.2f} trunk z {r['trunk_z']} {'OK' if r['ok'] else 'MISS'} con@{r['con_step']}", flush=True)
            if save_this:
                with open(a.save, "wb") as f:
                    pickle.dump(dict(qpos=np.array(frames), log=r["log"], p=p.__dict__), f)
            r.pop("log"); res.append(r)
    docked = sum(r["ok"] for r in res); touching = sum(r["button_contacts"] > 0 for r in res)
    print(f"docked {docked} / {len(res)}, runs where the hood touched the power button: {touching}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(cmd=" ".join(["docking.py"] + sys.argv[1:]), docked=docked, runs=len(res), runs_touching_button=touching,
                           results=res), f, indent=1)

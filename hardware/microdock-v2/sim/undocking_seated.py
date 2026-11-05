"""MicroDock v2 undocking in MuJoCo: from the dock, Microduck scoots forward out of it against the magnet (the undocking
policy, policy/scoot_undock.onnx), stands up (the official alpha_sitstand) and walks off (the official velstand).

Both scoot policies run with the training's latencies (latency.py): the undocking policy only breaks free with them.
--mag sets the magnet's pull in newtons; the real adapter's has not been measured. Needs the same checkout and
official policies as docking_seated.py:

  export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
  # trials: dock as docking_seated.py does (then rest 2 s on the docking policy), undock, stand up, walk off
  uv run python hardware/microdock-v2/sim/undocking_seated.py --seeds 41,42,43 --n 10 --mag 4 --out undock.json
  # the showcased run: continue docking_seated.py's saved run (--rest limp) through the undocking, for the preview
  uv run python hardware/microdock-v2/sim/undocking_seated.py --continue run.pkl --save full.pkl

The experimental single policy docks and undocks: set MICRODUCK_SCOOT_DOCK and --undock to
policy/experimental/scoot_dock_undock.onnx.
"""
import argparse, contextlib, io, json, math, os, pickle, sys
import numpy as np, mujoco, onnxruntime as ort
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import docking_seated as ds
import latency
dg, dk, seated = ds.dg, ds.dk, ds.seated
UNDOCK = os.path.join(os.path.dirname(HERE), "policy", "scoot_undock.onnx")
# How the trials dock: docking_seated.py's defaults, handing over to the docking policy as soon as the sit is done,
# then resting 2 s on it.
DOCK = argparse.Namespace(xturn=200.0, seat_mu=ds.SEAT_MU, yaim=-29.0, sit_yaw=6.6, yawt=0.0, sit_s=4.0, handover_s=0.0,
                          vback=-0.05, kyaw=0.03, max_back_s=12.0, hold=2.0, rest="policy", doze_pause=0.1,
                          doze_push=False, sit_until_still=True)


def tilt_deg(data):
    w, x, y, z = data.qpos[3:7]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2*(x*x + y*y)))))


def hood_force(model, data, hb):
    """The contact force between the hood and the duck (N, normal components)."""
    f6 = np.zeros(6); tot = 0.0
    for i in range(data.ncon):
        c = data.contact[i]; b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 == hb) != (b2 == hb) and 0 not in (b1, b2):
            mujoco.mj_contactForce(model, data, i, f6); tot += abs(f6[0])
    return tot


def undock(model, data, bc, pol, sess, out_mm=70.0):
    """From the docked rest: scoot forward (0.05 m/s, heading held) until the trunk is out_mm out or 10 s pass, rest
    0.5 s, stand up (the sitstand, 3.5 s), walk off (the velstand at 0.3 m/s, 3 s)."""
    hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood"); L = []
    def step(ph, vx=0.0, wz=0.0):
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, 0.0, wz)
            dg.step_control(model, data, bc, pol, frame_cb=lambda k: dk.magnet(model, data))
        e = dk.tip_error(model, data)
        L.append((ph, data.qpos[0]*1000, data.qpos[2]*1000, tilt_deg(data), float(e[0]), hood_force(model, data, hb),
                  dg.yaw_deg(data), data.qpos[0]*1000, data.qpos[1]*1000))
    x0 = data.qpos[0]*1000; yaw_d = dg.yaw_deg(data)
    pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = sess
    for _ in range(500):
        step("scoot", 0.05, float(np.clip(DOCK.kyaw*dg.wrap(DOCK.yawt - dg.yaw_deg(data)), -0.3, 0.3)))
        if data.qpos[0]*1000 >= x0 + out_mm: break
    out = data.qpos[0]*1000 - x0
    for _ in range(25): step("still")
    pol.sit_mode = False; pol.current_policy = "sit"; pol.ort_session = pol.sit_session
    for _ in range(175): step("stand")
    pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = pol.walking_session
    for _ in range(150): step("walk", 0.3)             # the velstand stands still at 0.15 m/s; 0.3 as on the approach
    sc = [q for q in L if q[0] == "scoot"]; st = [q for q in L if q[0] == "stand"]; wk = [q for q in L if q[0] == "walk"]
    rel = next((i for i, q in enumerate(sc) if q[4] > 4.0), None)          # the tip more than 4 mm off the adapter
    walk_mm = math.hypot(*(np.array(wk[-1][7:9]) - np.array(wk[0][7:9])))
    return dict(out_mm=round(out, 1), release_s=None if rel is None else round(rel/50, 2), scoot_s=round(len(sc)/50, 2),
                scoot_max_yaw=round(max(abs(dg.wrap(q[6] - yaw_d)) for q in sc), 1),
                scoot_max_tilt=round(max(q[3] for q in sc), 1), scoot_max_hood_force=round(max(q[5] for q in sc), 1),
                stood=bool(st[-1][2] > 100 and st[-1][3] < 20), stand_max_hood_force=round(max(q[5] for q in st), 1),
                walk_mm=round(walk_mm, 1), walked=bool(wk[-1][2] > 100 and max(q[3] for q in wk) < 45 and walk_mm > 100.0))


def trials(a):
    p = seated.Params(); res = []
    for seed in [int(s) for s in a.seeds.split(",")]:   # each seed from scratch: the same as one run per seed
        model, data, bc, pol = ds.make(p, seat_mu=DOCK.seat_mu)
        latency.enable(pol, bc, seed=seed, official=False, scoot=True)
        sess = ort.InferenceSession(a.undock); pol.undock_session = sess
        rng = np.random.default_rng(seed)
        for t in range(a.n):
            x0, y0, yaw0 = rng.uniform(380, 460), rng.uniform(-40, 40), 180 + rng.uniform(-12, 12)
            nx, ny, nyaw = rng.normal(0, 5), rng.normal(0, 3), rng.normal(0, 1)
            r = ds.run(model, data, bc, pol, p, DOCK, x0, y0, yaw0, nx, ny, nyaw)
            u = undock(model, data, bc, pol, sess) if r["ok"] else {}
            row = dict(seed=seed, t=t, docked=bool(r["ok"]), err=r["err"], after_sit=r["after_sit"], con_step=r["con_step"],
                       start=[round(x0), round(y0), round(yaw0, 1)], **u)
            res.append(row); print(json.dumps(row, default=float), flush=True)
    ok = [r for r in res if r["docked"]]
    summ = dict(undock=os.path.basename(a.undock), dock=os.path.basename(ds.SCOOT_DOCK), mag_N=a.mag, docked=len(ok),
                runs=len(res), released=sum(r["release_s"] is not None for r in ok), out=sum(r["out_mm"] >= 69 for r in ok),
                stood=sum(r["stood"] for r in ok), walked=sum(r["walked"] for r in ok),
                release_s_max=max((r["release_s"] for r in ok if r["release_s"] is not None), default=None),
                scoot_max_yaw=max((r["scoot_max_yaw"] for r in ok), default=None),
                stand_max_hood_force=max((r["stand_max_hood_force"] for r in ok), default=None))
    print("summary", json.dumps(summ, default=float), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(cmd=" ".join(["undocking_seated.py"] + sys.argv[1:]), summary=summ, results=res), f, indent=1, default=float)


def continue_run(a):
    """Take on from a saved docked run that went to sleep (docking_seated.py --rest limp --save):
    1. wake: the servos back on at their dozed angles, then a minimum-jerk return to the docked seated pose (--wake s);
    2. scoot: the undocking policy until the trunk is --out_mm out, heading held;
    3. halt: --still_gap control steps still;
    4. stand: the official sitstand (flag 0), handed over once up and still;
    5. walk: the official velstand at --walk_vx m/s, heading held, for --walk s.
    Saves the run and its continuation in docking_seated.py's --save format, with a summary under "undock"."""
    H = pickle.load(open(a.cont, "rb")); p = seated.Params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in H["p"].items()})
    dk.MAG_F = a.mag
    model, data, bc, pol = ds.make(p)
    assert H["qpos"].shape[1] == model.nq, (H["qpos"].shape, model.nq)
    pol.undock_session = ort.InferenceSession(a.undock)
    latency.enable(pol, bc, seed=41, official=False, scoot=True)
    hz = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_z")]
    hy = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_y")]
    hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
    Q, L = [], dict(phase=[], err=[], hood=[])
    def rec(ph):
        e = dk.tip_error(model, data); Q.append(data.qpos.copy()); L["phase"].append(ph); L["err"].append(e.tolist())
        L["hood"].append([data.qpos[hz]*1e3, data.qpos[hy]*1e3])
    data.qpos[:] = H["qpos"][-1]; data.qvel[:] = 0.0; mujoco.mj_forward(model, data)       # asleep, at rest
    qi = np.array(pol.joint_qpos_indices); q_doze = data.qpos[qi].copy()
    last_docked = max(i for i, ph in enumerate(H["log"]["phase"]) if ph == "docked")
    q_seat = H["qpos"][last_docked][qi].copy()
    bc.reset(data.qpos); bc.q_target[:] = q_doze
    nw = int(round(a.wake*50))
    for k in range(nw):
        u = (k + 1)/nw; w = 10*u**3 - 15*u**4 + 6*u**5
        bc.q_target[:] = q_doze + (q_seat - q_doze)*w
        for _ in range(4):
            bc.update(); mujoco.mj_step(model, data); dk.magnet(model, data)
        rec("wake")
    def step(ph, vx=0.0, wz=0.0):
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, 0.0, wz); dg.step_control(model, data, bc, pol, frame_cb=lambda k: dk.magnet(model, data))
        rec(ph)
    def hold_heading(yawt=0.0, k=0.03, lim=0.3): return float(np.clip(k*dg.wrap(yawt - dg.yaw_deg(data)), -lim, lim))
    pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = pol.undock_session
    pol.last_action[:] = (bc.q_target - pol.default_pose)/pol.action_scale             # its last action = the current targets
    x0 = data.qpos[0]*1000; release = None; fmax = 0.0
    for k in range(500):
        step("scoot", 0.05, hold_heading()); fmax = max(fmax, hood_force(model, data, hb))
        if release is None and not dk.connected(dk.tip_error(model, data)): release = len(Q) - 1
        if data.qpos[0]*1000 >= x0 + a.out_mm: break
    for _ in range(a.still_gap): step("halt", 0.0, 0.0)
    pol.sit_mode = False; pol.current_policy = "sit"; pol.ort_session = pol.sit_session
    still = 0; fstand = 0.0
    for k in range(250):
        step("stand"); fstand = max(fstand, hood_force(model, data, hb))
        v = np.linalg.norm(data.qvel[:3])
        still = still + 1 if (data.qpos[2] > 0.108 and v < 0.08 and tilt_deg(data) < 12.0) else 0
        if k >= 40 and still >= 3: break
    stand_s = (k + 1)/50
    pol.sit_mode = False; pol.current_policy = "walking"; pol.ort_session = pol.walking_session
    xw = data.qpos[:2].copy()
    for k in range(int(round(a.walk*50))): step("walk", a.walk_vx, hold_heading())
    ph = L["phase"]; n0 = len(H["qpos"])
    summ = dict(fell_in_walk=any(Q[i][2] < 0.09 for i in range(len(Q)) if ph[i] == "walk"),
                min_z_stand=round(min(Q[i][2] for i in range(len(Q)) if ph[i] == "stand")*1000, 1), wake_steps=nw,
                release_step=None if release is None else n0 + release,
                release_s_into_scoot=None if release is None else (release - nw)/50, scoot_steps=ph.count("scoot"),
                stand_s=stand_s, walk_mm=round(float(np.linalg.norm(data.qpos[:2] - xw))*1000, 1),
                end_z=round(data.qpos[2]*1000, 1), end_yaw=round(dg.yaw_deg(data), 1), hood_force_scoot_N=round(fmax, 1),
                hood_force_stand_N=round(fstand, 1))
    print(json.dumps(summ), flush=True)
    out = dict(qpos=np.concatenate([H["qpos"], np.array(Q)]), p=H["p"],
               log=dict(phase=list(H["log"]["phase"]) + ph, err=list(H["log"]["err"]) + L["err"],
                        hood=list(H["log"]["hood"]) + L["hood"]), undock=summ)
    if a.save: pickle.dump(out, open(a.save, "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--undock", default=UNDOCK, help="the undocking policy")
    ap.add_argument("--mag", type=float, default=4.0, help="the magnet's pull (N)")
    ap.add_argument("--seeds", default="41"); ap.add_argument("--n", type=int, default=10, help="trials per seed")
    ap.add_argument("--out", default="", help="write the trials' results (and a summary) as JSON")
    ap.add_argument("--continue", dest="cont", default="", help="a saved docked run to continue instead of the trials")
    ap.add_argument("--save", default="", help="with --continue: pickle the whole run at 50 Hz (for the renders)")
    ap.add_argument("--wake", type=float, default=1.0); ap.add_argument("--walk", type=float, default=3.5)
    ap.add_argument("--walk_vx", type=float, default=0.3); ap.add_argument("--out_mm", type=float, default=70.0)
    ap.add_argument("--still_gap", type=int, default=15, help="control steps still between the scoot and the stand-up")
    a = ap.parse_args()
    if a.cont: continue_run(a)
    else:
        dk.MAG_F = a.mag; trials(a)

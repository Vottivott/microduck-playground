"""MicroDock v2 docking trials in MuJoCo: Microduck walks toward the dock's tag (detection emulated as truth + noise),
turns in place, SITS (the official sitstand policy), then butt-walks backward (the butt-walk policy) until the magnetic
tip in its USB-C port meets the adapter in the floating hood, stops when USB power appears and rests seated.

Walking is the official velstand.onnx and sitting the official alpha_sitstand.onnx; the butt walk is policy/butt_walk.onnx
(seated locomotion on command, trained for this dock: ../training). All three run on the BAM XL330 actuator model. Needs
a checkout of pollen-robotics/microduck_rl (develop, validated at bae39f59: its scripts/infer_policy.py and robot model)
and the official policies (huggingface.co/pollen-robotics/microduck-policies):

  export MICRODUCK_RL_DIR=/path/to/microduck_rl MICRODUCK_POLICY_DIR=/path/to/microduck-policies
  uv run python hardware/microdock-v2/sim/docking_seated.py --seeds 41,42,43 --n 10 --out results.json
  uv run python hardware/microdock-v2/sim/docking_seated.py --n 1 --seed 0 --noise 0 --start 450,15,178 --hold 4 --save run.pkl

The MicroDock's render_preview.py replays a saved run (hardware/microdock/sim).
"""
import argparse, contextlib, io, json, math, os, pickle, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE); V1 = os.path.join(os.path.dirname(V2), "microdock")
sys.path.insert(0, os.path.join(V2, "source")); sys.path.insert(0, os.path.join(V1, "sim"))
import docking as dg              # the MicroDock's simulation: scene, BAM actuators, official policies, magnet
import seated
dk = dg.dk
BUTT_WALK = os.environ.get("MICRODUCK_BUTT_WALK", os.path.join(V2, "policy", "butt_walk.onnx"))


SEAT_MU = 0.45                      # the seat (shell, battery cover) on PLA or a floor; the butt walk trained with 0.3-0.6


def seat_friction(model, mu=SEAT_MU):
    """Contacts as the butt walk trained them: every robot collision geom uses its own friction against the floor and the
    dock (priority 1), the seat `mu`, the feet 1.0. The geoms that meet the hood (contype 4) keep their low friction."""
    for g in range(model.ngeom):
        b = model.body(model.geom_bodyid[g]).name
        if (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0) or b in ("world", "hood"):
            continue
        model.geom_priority[g] = 1; model.geom_condim[g] = 3
        if model.geom_contype[g] == 4:
            continue
        model.geom_friction[g][0] = 1.0 if "foot" in model.geom(g).name else mu


def make(p, vin=7.6, seat_mu=SEAT_MU):
    model, data, bc, pol = dg.make(p, vin)
    seat_friction(model, seat_mu)
    if not os.path.exists(BUTT_WALK):
        sys.exit(f"butt-walk policy not found: {BUTT_WALK}")
    import onnxruntime as ort
    pol.butt_walk_session = ort.InferenceSession(BUTT_WALK)
    return model, data, bc, pol


def run(model, data, bc, pol, p, a, x0, y0, yaw0, nx, ny, nyaw, rec=None):
    """One docking attempt from (x0 mm, y0 mm, yaw0 deg); nx, ny, nyaw: the tag estimate's error (mm, mm, deg)."""
    with contextlib.redirect_stdout(io.StringIO()):
        dg.reset(model, data, bc, pol, x=x0/1000, y=y0/1000, yaw=math.radians(yaw0), z=0.125)
        pol.sit_mode = False; pol.current_policy = 'walking'; pol.ort_session = pol.walking_session
    log = {"phase": [], "err": [], "hood": []}
    hz = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_z")]
    hy = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hood_y")]
    btn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "power_button"); hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hood")
    pressed = [0]
    def st(phase, vx, vy, wz):
        with contextlib.redirect_stdout(io.StringIO()):
            pol.set_vel_cmd(vx, vy, wz); dg.step_control(model, data, bc, pol, frame_cb=lambda k: dk.magnet(model, data))
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
        des = math.degrees(math.atan2(-(Y() + ny - a.yaim), -150.0)); e = dg.wrap(des - (dg.yaw_deg(data) + nyaw))
        st("approach", 0.3, 0, float(np.clip(0.035*e, -0.5, 0.5)))
        if X() + nx <= a.xturn: break
    for _ in range(20): st("stop", 0, 0, 0)
    yaw_turn = a.yawt + a.sit_yaw                     # the sit turns the duck by about -sit_yaw: turn that much further first
    for _ in range(600):
        e = dg.wrap(yaw_turn - (dg.yaw_deg(data) + nyaw))
        if abs(e) < 10: break
        st("turn", 0, 0, 1.0 if e > 0 else -1.0)
    for _ in range(250):
        e = dg.wrap(yaw_turn - (dg.yaw_deg(data) + nyaw))
        if abs(e) < 1.0: break
        st("turn", 0, 0, float(np.sign(e)*(0.6 if abs(e) > 4 else 0.45)))
    for _ in range(25): st("settle", 0, 0, 0)
    after_turn = (X(), Y(), dg.yaw_deg(data))
    # sit (official sitstand: posture flag 1), then hand over to the butt walk
    with contextlib.redirect_stdout(io.StringIO()):
        pol.toggle_sit()
    for _ in range(int(a.sit_s*50)): st("sit", 0, 0, 0)
    after_sit = (X(), Y(), dg.yaw_deg(data), data.qpos[2]*1000)
    pol.sit_mode = False; pol.current_policy = 'walking'; pol.ort_session = pol.butt_walk_session
    for _ in range(int(a.handover_s*50)): st("seated", 0, 0, 0)
    # butt-walk backward with heading hold until the tip meets the adapter (USB power)
    con_step = None
    for k in range(int(a.max_back_s*50)):
        e_yaw = dg.wrap(a.yawt - (dg.yaw_deg(data) + nyaw))
        err = st("backin", a.vback, 0, float(np.clip(a.kyaw*e_yaw, -0.3, 0.3)))
        if con_step is None and dk.connected(err): con_step = len(log["phase"])
        if dk.connected(err): break
        if X() < p.dock_x - 15: break
    for _ in range(int(a.hold*50)): err = st("docked", 0, 0, 0)
    ok = dk.connected(err)
    return dict(ok=ok, button_contacts=pressed[0], err=np.array(err).round(2).tolist(), after_turn=[round(v, 1) for v in after_turn],
                after_sit=[round(v, 1) for v in after_sit], con_step=con_step, hood=[round(v, 2) for v in log["hood"][-1]],
                trunk_z=round(data.qpos[2]*1000, 1), log=log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=6, help="trials per seed")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", default="", help="comma list; overrides --seed")
    ap.add_argument("--xturn", type=float, default=200.0, help="turn point: trunk x (mm from the board) where the duck stops and turns")
    ap.add_argument("--seat_mu", type=float, default=SEAT_MU, help="seat friction (the butt walk trained with 0.3-1.0)")
    ap.add_argument("--yaim", type=float, default=-29.0, help="aim this far to one side (mm): the turn and the sit shift the duck back by about as much")
    ap.add_argument("--sit_yaw", type=float, default=6.6, help="turn this much past the target heading (deg): the sit turns the duck back by about as much")
    ap.add_argument("--yawt", type=float, default=0.0)
    ap.add_argument("--sit_s", type=float, default=4.0, help="seconds for the sit"); ap.add_argument("--handover_s", type=float, default=0.5)
    ap.add_argument("--vback", type=float, default=-0.05, help="butt-walk speed (m/s, negative = backward)")
    ap.add_argument("--kyaw", type=float, default=0.03, help="heading hold gain (rad/s per deg)")
    ap.add_argument("--max_back_s", type=float, default=12.0)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds resting seated, docked, at the end")
    ap.add_argument("--save", default="", help="pickle the first trial's qpos at 50 Hz (for the renders)")
    ap.add_argument("--out", default="", help="write the results (and a summary) as JSON")
    ap.add_argument("--set", nargs="*", default=[], help="override Params fields, e.g. --set z0=64")
    ap.add_argument("--noise", type=float, default=1.0, help="scale of the tag estimate's error (sigma 5 mm along, 3 mm across, 1 deg)")
    ap.add_argument("--start", default="", help="x,y,yaw start (mm, mm, deg) instead of a random one")
    a = ap.parse_args()
    p = seated.Params()
    for kv in a.set:
        k, v = kv.split("="); setattr(p, k, type(getattr(p, k))(float(v)) if not isinstance(getattr(p, k), tuple) else tuple(map(float, v.split(","))))
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    res = []
    for seed in seeds:
        model, data, bc, pol = make(p, seat_mu=a.seat_mu)
        rng = np.random.default_rng(seed)
        for t in range(a.n):
            x0, y0, yaw0 = rng.uniform(380, 460), rng.uniform(-40, 40), 180 + rng.uniform(-12, 12)
            if a.start: x0, y0, yaw0 = map(float, a.start.split(","))
            nx, ny, nyaw = rng.normal(0, 5)*a.noise, rng.normal(0, 3)*a.noise, rng.normal(0, 1)*a.noise
            frames = []
            save_this = bool(a.save) and t == 0 and seed == seeds[0]
            r = run(model, data, bc, pol, p, a, x0, y0, yaw0, nx, ny, nyaw, rec=(lambda ph: frames.append(data.qpos.copy())) if save_this else None)
            r.update(seed=seed, start=[round(x0), round(y0), round(yaw0)], noise=[round(nx, 1), round(ny, 1), round(nyaw, 2)])
            print(f"seed {seed} #{t} start ({x0:.0f},{y0:+.0f},{yaw0:.0f}) turn@({r['after_turn'][0]:.0f},{r['after_turn'][1]:+.0f}) "
                  f"sat@({r['after_sit'][0]:.0f},{r['after_sit'][1]:+.0f},{r['after_sit'][2]:+.1f}deg,z{r['after_sit'][3]:.0f}) -> err {r['err']} "
                  f"btn {r['button_contacts']} hood z/y {r['hood'][0]:+.2f}/{r['hood'][1]:+.2f} trunk z {r['trunk_z']} {'OK' if r['ok'] else 'MISS'} con@{r['con_step']}", flush=True)
            if save_this:
                with open(a.save, "wb") as f:
                    pickle.dump(dict(qpos=np.array(frames), log=r["log"], p=p.__dict__), f)
            r.pop("log"); res.append(r)
    docked = sum(r["ok"] for r in res); touching = sum(r["button_contacts"] > 0 for r in res)
    print(f"docked {docked} / {len(res)}, runs where the hood touched the power button: {touching}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(cmd=" ".join(["docking_seated.py"] + sys.argv[1:]), docked=docked, runs=len(res), runs_touching_button=touching,
                           results=res), f, indent=1)

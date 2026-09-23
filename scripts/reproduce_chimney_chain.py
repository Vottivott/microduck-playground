"""The whole corridor sequence in one take: side-step in, climb, exit onto a
platform 3 m up.

Three policies in one episode, hosted in the EXIT environment because that is
the only one that has the platform and the overhanging walls.  The climb and
the exit are both built on the chimney recipe, so they share its default pose
and run natively; only the approach is a guest and needs remapping:

  * its action offset is the STANCE, not the brace;
  * `joint_pos_rel` is measured from the default pose, so its slice of the
    observation has to be shifted by the same difference;
  * the command block carries a different meaning for each policy.

Phase 1 -> 2 fires on the approach's own handover test, 2 -> 3 when the duck
reaches platform height.
"""
import dataclasses
import json
import math
import os
import re
import sys

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.robot import corridor_stage as cs
from mjlab_microduck.robot import exit_stage as ex
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.tasks import approach_mdp as ap
from mjlab_microduck.tasks import exit_mdp as em
from mjlab_microduck.tasks.mdp import _servo_joint_ids
from mjlab_microduck.tasks import symmetry as sym
from mjlab_microduck.tasks.microduck_chimney_env_cfg import BRACE_FRAME
from chimney_presentation import DUCK_PALETTES, recolored_robot_cfg
from chimney_presentation import configure_video_cfg, fix_render_shadows

ap_ck, ch_ck, ex_ck, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
DEV = "cuda:0"
WIDTH = 0.125
ENTER_NEAR = os.getenv("ENTER_NEAR", "0") == "1"   # walk in from the SAME side
# the exit leaves by, i.e. the camera side (user 2026-09-21).  The corridor is
# symmetric under a 180 deg yaw rotation, so the approach policy does not need
# retraining: spawn it on the far side turned around, and present it the
# rotated frame.  `_root` is the single place every approach function reads
# pose from, so rotating THERE flips the command obs and the handover test
# together and nothing else has to know.
ENTRY_SIDE = float(os.getenv("ENTRY_SIDE", "-1"))   # -1: far side (a11); +1: camera side
# with a policy TRAINED for it (a13, the mirrored gait) - spawn on +y facing
# +x, no rotation, no frame flip.  ENTER_NEAR (rotate the duck) stays off.
START_Y = 0.45 if ENTER_NEAR else (-0.45 * -ENTRY_SIDE if ENTRY_SIDE > 0 else -0.45)
SECONDS = float(os.getenv("FULL_SECONDS", "60"))
SEED = int(os.getenv("FULL_SEED", "3"))
RES = int(os.getenv("FULL_RES", "720"))
HEIGHT_OFFSET = -0.12
STAND = {2: -0.4579, 3: -0.0049, 4: 0.4529, 11: 0.4579, 12: 0.0049, 13: -0.4529}

# camera keyframes, reusing the angles arrived at for the individual clips
# All three shots live on the PLATFORM side of the corridor, so the duck walks
# out toward the viewer at the end rather than away from it (user 2026-09-21).
CAM_ENTRY = dict(az=118.0, el=-14.0, dist=1.0, dz=0.06)
CAM_CLIMB = dict(az=92.0, el=-4.0, dist=1.15, dz=0.02)
CAM_EXIT = dict(az=72.0, el=-15.0, dist=1.65, dz=0.02)

# Isolated presentation variant of the original shelf. The reference overlap
# is 45 cm, followed by 40 cm of open platform. Reflect geometry along y so
# the exit comes toward the reference camera while the duck still faces +x.
ex.PLATFORM_THICK_M = cs.WALL_THICKNESS_M
ex.WALL_ABOVE_M = 0.40
ex._RGBA = "0.60 0.55 0.50 1"
ex.PLATFORM_CLEAR_M = float(os.getenv("REFERENCE_CLEAR", "0.0"))
_wall_spec = ex._stepped_wall_spec
def reference_wall(name, rgba):
    spec = _wall_spec(name, ex._RGBA)
    for geom in spec.geoms:
        geom.pos[1] *= -1
    return spec
ex._stepped_wall_spec = reference_wall
_platform_pos = ex.platform_mocap_pos
def reference_platform_pos():
    x, y, z = _platform_pos()
    return x, -y, z
ex.platform_mocap_pos = reference_platform_pos


def place_outside(env, env_ids=None, **_):
    if env_ids is None or len(env_ids) == 0:
        return
    n = len(env_ids)
    dev = env.device
    org = env.scene.env_origins[env_ids]
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
    for side, name in ((-1.0, cs.WALL_LEFT), (1.0, cs.WALL_RIGHT)):
        pos = torch.zeros(n, 3, device=dev)
        pos[:, 0] = side * (0.5 * WIDTH + 0.5 * cs.WALL_THICKNESS_M)
        # the stepped wall's body origin is at the FLOOR: its two segments
        # carry their own heights, so no half-height offset here
        pos[:, 1] = 0.0
        pos[:, 2] = ex.stepped_wall_mocap_z()
        env.scene[name].write_mocap_pose_to_sim(torch.cat([org + pos, q0], -1), env_ids=env_ids)
    ppos = torch.tensor(ex.platform_mocap_pos(), device=dev).repeat(n, 1)
    env.scene[ex.PLATFORM].write_mocap_pose_to_sim(
        torch.cat([org + ppos, q0], -1), env_ids=env_ids)
    yaw = (torch.rand(n, device=dev) * 2 - 1) * 0.20
    if ENTER_NEAR:
        yaw = yaw + math.pi
    a = 0.5 * yaw
    env.sim.data.qpos[env_ids, 0] = org[:, 0] + (torch.rand(n, device=dev) * 2 - 1) * 0.01
    env.sim.data.qpos[env_ids, 1] = org[:, 1] + START_Y
    env.sim.data.qpos[env_ids, 2] = org[:, 2] + 0.115
    env.sim.data.qpos[env_ids, 3] = torch.cos(a)
    env.sim.data.qpos[env_ids, 4:6] = 0.0
    env.sim.data.qpos[env_ids, 6] = torch.sin(a)
    env.sim.data.qvel[env_ids, :] = 0.0
    jn = env.scene["robot"].joint_names
    for i, nm in enumerate(jn):
        for pat, val in HOME_FRAME.joint_pos.items():
            if re.fullmatch(pat, nm) or re.search(pat, nm):
                env.sim.data.qpos[env_ids, 7 + i] = val
    ids = _servo_joint_ids(env, env.scene["robot"])
    for j, val in STAND.items():
        env.sim.data.qpos[env_ids, 7 + ids[j]] = val


def build(render: bool = True):
    # Always constructed identically, capture or not: building one env with a
    # renderer and one without gave the two passes different random draws, so
    # the dry run reached 3.11 m and the rendered run fell at 2.18 m.
    cfg = load_env_cfg("Mjlab-Exit-MicroDuck", play=True)
    palette = DUCK_PALETTES["cream"]

    def _paint(spec):
        by_mat = {m.name: tuple(float(v) for v in m.rgba) for m in spec.materials}
        for g in spec.geoms:
            rgba = by_mat.get(g.material) if g.material else None
            if rgba is None:
                nm = (g.name or "").lower()
                if "foot" in nm:
                    rgba = palette["foot"]
                elif "ankle" in nm:
                    rgba = palette["ankle"]
            if rgba is not None:
                g.rgba[:] = rgba
        return spec

    e = recolored_robot_cfg(cfg.scene.entities["robot"], "cream")
    b = e.spec_fn
    e.spec_fn = lambda base=b: _paint(base())
    cfg.scene.entities["robot"] = e
    cfg.scene.num_envs = 1
    cfg.seed = SEED
    cfg.episode_length_s = SECONDS + 5.0
    cfg.auto_reset = False
    cfg.sim.nconmax = 200
    configure_video_cfg(cfg, RES, RES)
    for k in list(cfg.terminations):
        if k != "nan_state":
            cfg.terminations.pop(k)
    for k in list(cfg.events):
        if k.startswith("set_"):
            cfg.events.pop(k)
    cfg.events["place_outside"] = EventTermCfg(func=place_outside, mode="reset", params={})
    return ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array")


def run(render: bool, entry_end: float):
    """`render` now means CAPTURE FRAMES; the environment is identical either
    way, so a seed that works in the sweep works in the take."""
    raw = build()
    rmodel = None
    if True:
        az0 = math.radians(CAM_ENTRY["az"])
        # Exact sun from enter_climb_v3, not the later oblique override.
        fix_render_shadows(raw, light_dir=(0.43 * math.cos(az0), 0.43 * math.sin(az0), -1.0))
        rmodel = raw._offline_renderer._model
        rmodel.vis.map.shadowclip = 4.0
        rmodel.stat.extent = 4.0

    agent = load_rl_cfg("Mjlab-Exit-MicroDuck")
    env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    def fresh_observations():
        # compute() defaults to returning a cached observation. Explicitly
        # invalidate after state/history changes; do not advance physics.
        raw.observation_manager._obs_buffer = None
        return env.get_observations()

    def _load(ck, task):
        a = load_rl_cfg(task)
        r = (load_runner_cls(task) or OnPolicyRunner)(env, dataclasses.asdict(a), device=DEV)
        r.load(ck, load_cfg={"actor": True}, map_location=DEV)
        return r.get_inference_policy(device=DEV)

    pol_ap = _load(ap_ck, "Mjlab-Approach-MicroDuck")
    pol_ch = _load(ch_ck, "Mjlab-Chimney-MicroDuck")
    pol_ex = _load(ex_ck, "Mjlab-Exit-MicroDuck")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    getup = ort.InferenceSession(os.environ["GETUP_POLICY"], sess_options=opts,
                                providers=["CPUExecutionProvider"])
    meta = getup.get_modelmeta().custom_metadata_map
    # Phase 3, render-side: once the exit is out past the line and upright, hand
    # over to the APPROACH policy with a zero goal vector.  Its handover pose is
    # tall + level + still and it holds it (99 %); the exit lineage (brace
    # default pose) never learned to hold an upright stance (2026-09-22).
    pol_stand = _load(os.environ["STAND_POLICY"], "Mjlab-Approach-MicroDuck") if os.getenv("STAND_POLICY") else None
    from mjlab_microduck.tasks import exit_mdp as em
    robot = raw.scene["robot"]
    names = robot.joint_names

    def offsets(frame):
        out = torch.zeros(len(names), device=DEV)
        for i, n in enumerate(names):
            for pat, v in frame.joint_pos.items():
                if re.fullmatch(pat, n) or re.search(pat, n):
                    out[i] = v
        return out

    off_ap, off_ch = offsets(HOME_FRAME), offsets(BRACE_FRAME)
    getup_names = meta["joint_names"].split(",")
    assert list(names) == getup_names, (names, getup_names)
    off_getup = torch.tensor(np.fromstring(meta["default_joint_pos"], sep=","), device=DEV, dtype=torch.float32)
    op = torch.tensor(sym._OBS_PERM, device=DEV)
    osign = torch.tensor(sym._OBS_SIGN, device=DEV)
    jp = torch.tensor(sym._JOINT_PERM, device=DEV)
    js = torch.tensor(sym._JOINT_SIGN, device=DEV)
    alpha = torch.tensor([.5 if n.startswith(("neck", "head")) else .7 for n in names], device=DEV)
    gains = [a.kp_scale.clone() for a in robot.actuators]
    previous_raw = torch.zeros(1, 14, device=DEV)
    previous_executed = previous_raw.clone()
    trace = []
    contact_run = 0
    if ENTER_NEAR and not getattr(ap, "_flipped", False):
        _root_orig = ap._root

        def _root_flipped(env, asset_cfg=ap._DEFAULT_ASSET_CFG):
            pos, quat, rest = _root_orig(env, asset_cfg)
            pos2 = pos.clone()
            pos2[:, 0] = -pos[:, 0]
            pos2[:, 1] = -pos[:, 1]
            # q <- q_z(-pi) * q, with q_z(-pi) = (0, 0, 0, -1)
            w, x, y, z = quat.unbind(-1)
            return pos2, torch.stack([z, y, -x, -w], dim=-1), rest

        ap._root = _root_flipped
        ap._flipped = True
    act_term = raw.action_manager.get_term("joint_pos")
    # Hand over on FOOT height, not trunk height.  Measured 2026-09-21: at the
    # old trunk-based gate the feet were 18-29 cm ABOVE the platform (mean
    # 22.6) - about a duck-height too high, and on screen it reads as climbing
    # out of the top of the corridor.  The exit trains with its feet +6..+26 cm
    # above the platform, so the BOTTOM of its own range is where to switch.
    _ankles = [i for i, b in enumerate(robot.body_names) if "ankle" in b.lower()]
    assert _ankles, robot.body_names
    st = ap._st(raw)
    st["width"][:] = WIDTH
    # Recovery remains controlled; presentation effects must never cut motors.
    obs = env.get_observations()
    okey = "actor" if hasattr(obs, "keys") else None
    cam = None
    if True:
        cam = raw._offline_renderer._cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1

    EXIT_Z = ex.PLATFORM_TOP_M + 0.12       # switch to the exit at platform level
    phase = 0
    # EXIT_ONLY: start the episode from a banked climb arrival and run only the
    # exit, inside THIS renderer's environment.  The probe injects the same
    # states but builds its own env, so it cannot tell "the env differs" from
    # "the live trajectory differs".  This can.
    if os.getenv("EXIT_ONLY"):
        _bank = torch.load(os.getenv("EXIT_ONLY"), map_location=DEV)
        _k = int(os.getenv("EXIT_ONLY_IDX", "0")) % _bank["qpos"].shape[0]
        _q = _bank["qpos"][_k:_k + 1].clone()
        _v = _bank["qvel"][_k:_k + 1].clone()
        _q[:, :3] += raw.scene.env_origins[0:1] - _bank["origins"][_k:_k + 1].to(DEV)
        _q[:, 2] += ex.PLATFORM_TOP_M - float(_bank.get("platform_top", ex.PLATFORM_TOP_M))
        raw.sim.data.qpos[0:1] = _q
        raw.sim.data.qvel[0:1] = _v
        raw.sim.forward()
        raw.action_manager.reset()
        obs = fresh_observations()
        phase, t_entry_end, t_exit_start = 2, 0.0, 0.0
        print("EXIT_ONLY spawned from %s idx %d" % (os.getenv("EXIT_ONLY"), _k))
    t_entry_end, t_exit_start, t_reach, t_stand_start = -1.0, -1.0, -1.0, -1.0
    if os.getenv("RECOVERY_STATE"):
        bank = np.load(os.environ["RECOVERY_STATE"])
        k = int(float(os.getenv("RECOVERY_TIME", "26.04")) * 50)
        raw.sim.data.qpos[0] = torch.as_tensor(bank["qpos"][k], device=DEV)
        raw.sim.data.qvel[0] = torch.as_tensor(bank["qvel"][k], device=DEV)
        raw.sim.forward()
        raw.action_manager.reset()
        previous_raw = robot.data.joint_pos.clone() - off_getup
        previous_executed = previous_raw.clone()
        obs = fresh_observations()
        phase, t_stand_start = 3, 0.
    frames, heights, tilts = [], [], []
    smooth = None
    steps = int(SECONDS / raw.step_dt)
    for i in range(steps):
        t = i * raw.step_dt
        with torch.inference_mode():
            p, _, _ = ap._root(raw, ap._DEFAULT_ASSET_CFG)
            jv = robot.data.joint_vel.abs().mean(dim=-1)
            _foot_z = float(robot.data.body_link_pos_w[0, _ankles, 2].min()
                            - raw.scene.env_origins[0][2])
            if phase == 0 and bool((ap.in_handover_pose(raw) & (jv < 0.35))[0]):
                phase, t_entry_end = 1, t
                # Same rule as the climb->exit switch: a policy swap is not a
                # reset, and the observation carries the action history.  The
                # climb's first steps were reading the approach's last actions
                # as its own.  Measured 2026-09-22: from the very states this
                # handover delivers, w17 climbs 94.5 % to 3 m with a clean
                # history, vs 71 % in the live chain.
                raw.action_manager.reset()
                obs = fresh_observations()
                if os.getenv("DUMP_ENTRY"):
                    # the REAL state the approach hands the climb, to measure the
                    # chain's climb from its actual start (chain 10/16 to 3 m vs 97 % isolated)
                    torch.save({"qpos": raw.sim.data.qpos[0:1].clone().cpu(),
                                "qvel": raw.sim.data.qvel[0:1].clone().cpu(),
                                "origins": raw.scene.env_origins[0:1].clone().cpu()},
                               os.getenv("DUMP_ENTRY") % SEED)
            elif phase == 1 and _foot_z > ex.PLATFORM_TOP_M + FOOT_OPEN_M:
                # Prefer to hand over when the climb has SETTLED at platform
                # height - the exit policy trains from a wedged spawn at rest -
                # but do not insist: requiring a settle inside a 25 cm window
                # meant the climb blew straight through it (one run carried on
                # to 5.9 m) and the exit never fired at all.  Wait up to a
                # second for calm, then switch regardless.
                if t_reach < 0.0:
                    t_reach = t
                # Wait for a NEAR-UPRIGHT moment before handing over.  The
                # climb ENTERS the window at ~60 deg of trunk tilt (median
                # 60.5), which is nothing like the 6-17 deg the exit trains
                # from - handing over on first entry gave it a posture it has
                # never seen, and it fell every time.  But while it braces in
                # the window it dips below 30 deg in 85 % of runs (median
                # minimum 17.6) and lingers there ~11 s, so the good moment is
                # simply worth waiting for.
                _q = robot.data.root_link_quat_w[0]
                _up = 1.0 - 2.0 * (float(_q[1]) ** 2 + float(_q[2]) ** 2)
                _tilt = math.degrees(math.acos(max(-1.0, min(1.0, _up))))
                _v = robot.data.root_com_lin_vel_w[0]
                _w = t - t_reach
                # The whole gate lives inside `z > EXIT_Z`, and measurement says
                # the climb only holds that for 2-4 s before it drops back or
                # falls - so TILT_HARD can NEVER fire on a 9/18 s schedule and
                # the run simply never hands over.  Relax inside the window we
                # actually get: 22 deg at first, then the exit's own tolerance
                # (26 deg, TILT_MAX_DEG) with a looser settle, then take it.
                _gate = TILT_GATE if _w < TILT_WAIT else TILT_RELAX
                _vz = 0.10 if _w < TILT_WAIT else 0.30
                if (_tilt < _gate and abs(float(_v[2])) < _vz) or _w > TILT_HARD:
                    phase, t_exit_start = 2, t
                    # The observation carries the ACTION HISTORY, and a policy
                    # swap is not a reset: without this the exit reads the
                    # climb's last actions as its own for the first steps.
                    # Measured 2026-09-21: the identical qpos+qvel injected as a
                    # spawn holds 100 % standing at 15 s, while the live switch
                    # from that same state collapses - the physics state was
                    # never the problem, the history was.
                    raw.action_manager.reset()
                    # ...and RECOMPUTE the observation.  `obs` in hand was
                    # produced by the previous env.step(), i.e. before the
                    # reset, so without this the exit's very first action is
                    # still chosen while reading the climb's action history -
                    # the reset would not take effect until one step later.
                    obs = fresh_observations()
                    if os.getenv("DUMP_ARRIVAL"):
                        # the REAL state the climb hands the exit at 3 m, for
                        # training the exit on its actual arrival distribution
                        torch.save({"qpos": raw.sim.data.qpos[0:1].clone().cpu(),
                                    "qvel": raw.sim.data.qvel[0:1].clone().cpu(),
                                    "origins": raw.scene.env_origins[0:1].clone().cpu()},
                                   os.getenv("DUMP_ARRIVAL") % SEED)

            o = obs[okey] if okey else obs
            if phase == 0:
                oc = o.clone()
                oc[:, 6:20] += (off_ch - off_ap)[None, :]
                oc[:, -13:] = 0.0
                oc[:, -6:] = ap.approach_command_obs(raw, dim=6)
                inp = obs.clone()
                if okey:
                    inp[okey] = oc
                if os.getenv("DUMP_ENTRY_OBS") and abs(t - 1.0) < 1e-6:
                    # the observation the approach is HANDED live at t=1 s, plus
                    # the physical state, so the approach env can be asked what
                    # it would have produced for the same state (guest-remap audit)
                    torch.save({"qpos": raw.sim.data.qpos[0:1].clone().cpu(),
                                "qvel": raw.sim.data.qvel[0:1].clone().cpu(),
                                "origins": raw.scene.env_origins[0:1].clone().cpu(),
                                "obs": oc[0:1].clone().cpu(),
                                "act_offset": (act_term._offset[0] if act_term._offset.dim() > 1 else act_term._offset).clone().cpu(),
                                "act_scale": (act_term._scale[0] if hasattr(act_term, "_scale") and torch.is_tensor(act_term._scale) and act_term._scale.dim() > 1 else getattr(act_term, "_scale", None))},
                               os.getenv("DUMP_ENTRY_OBS") % SEED)
                act = pol_ap(inp if okey else oc)
                act_term._offset[:] = off_ap[None, :]
            elif phase == 1:
                oc = o.clone()
                oc[:, -13:] = 0.0
                oc[:, -6] = (WIDTH - 0.13) / 0.02
                inp = obs.clone()
                if okey:
                    inp[okey] = oc
                act = pol_ch(inp if okey else oc)
                act_term._offset[:] = off_ch[None, :]
            elif phase == 2:
                if True:
                    _pp, _qq, _ = em._root(raw, em._DEFAULT_ASSET_CFG)
                    _car = float(em._carriage(raw)[0])
                    _upq = float(em._upright(_qq)[0])
                    # Contact is measured fresh from final forward, not reward caches.
                    c = raw.sim.data.contact
                    nc = int(raw.sim.data.nacon[0])
                    pairs = c.geom[:nc]
                    model = raw.sim.mj_model
                    platform_ids = [g for g in range(model.ngeom) if "exit_platform" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
                    pc = torch.isin(pairs, torch.tensor(platform_ids, device=DEV)).any(-1) & (c.dist[:nc] <= 0)
                    landed = bool(pc.any()) and float(_pp[0, 1]) < -float(os.getenv("RECOVERY_Y", str(ex.wall_end_y())))
                    contact_run = contact_run + 1 if landed else 0
                    if contact_run >= 2 and (_car < .13 or _upq < .85 or os.getenv("RECOVERY_ON_CONTACT") == "1"):
                        phase, t_stand_start = 3, t
                        previous_raw = raw.action_manager.action.clone() + off_ch - off_getup
                        if os.getenv("RECOVERY_HISTORY", "pose") == "pose":
                            previous_raw = robot.data.joint_pos.clone() - off_getup
                        elif os.getenv("RECOVERY_HISTORY") == "zero":
                            previous_raw.zero_()
                        previous_executed = previous_raw.clone()
                        print("GETUP_SWITCH", t, _pp[0].tolist(), flush=True)
                if phase == 2:
                    oc = o[:, op] * osign
                    # Mirror absolute joints about a possibly asymmetric brace.
                    oc[:, 6:20] = (o[:, 6:20] + off_ch)[:, jp] * js - off_ch
                    oc[:, 34:48] = (o[:, 34:48] + off_ch)[:, jp] * js - off_ch
                    oc[:, 48:] = 0
                    oc[:, 55] = ex.goal_y() + p[:, 1]
                    oc[:, 56] = ex.PLATFORM_TOP_M + em.STAND_Z - p[:, 2]
                    oc[:, 57] = -p[:, 0]
                    oc[:, 59] = (WIDTH - .13) / .02
                    inp = obs.clone()
                    inp[okey] = oc
                    virtual = pol_ex(inp)
                    act = (virtual + off_ch)[:, jp] * js - off_ch
                    act_term._offset[:] = off_ch[None, :]
            if phase == 3:
                oc = o.clone()
                oc[:, 6:20] += (off_ch - off_getup)[None, :]
                expected_joint_obs = robot.data.joint_pos - off_getup
                if not torch.allclose(oc[:, 6:20], expected_joint_obs, atol=.04):
                    print("OBS_AUDIT", {"given": oc[:,6:20].tolist(), "expected": expected_joint_obs.tolist(), "default": robot.data.default_joint_pos.tolist(), "brace": off_ch.tolist()}, flush=True)
                    raise RuntimeError("Recovery joint observation/default-pose mismatch")
                oc[:, -13:] = 0.0
                oc[:, 34:48] = previous_raw
                previous_raw = torch.tensor(getup.run(None, {getup.get_inputs()[0].name: oc.cpu().numpy()})[0], device=DEV)
                if os.getenv("RECOVERY_CONTRACT") == "approach":
                    oc[:, 6:20] += off_getup - off_ap
                    oc[:, -3] = 1.
                    oc[:, -2] = (WIDTH - .13) / .02
                    inp = obs.clone()
                    inp[okey] = oc
                    previous_raw = pol_ap(inp) + off_ap - off_getup
                _alpha = alpha if os.getenv("RECOVERY_CONTRACT", "official") == "desk" else torch.ones_like(alpha)
                act = _alpha * previous_raw + (1 - _alpha) * previous_executed
                previous_executed = act.clone()
                act_term._offset[:] = off_getup[None, :]
                for actuator, gain in zip(robot.actuators, gains):
                    actuator.kp_scale.copy_(gain * (.8 if os.getenv("RECOVERY_CONTRACT", "official") == "desk" else 1.))
            obs, _, _, _ = env.step(act)
            assert torch.isfinite(raw.sim.data.qpos).all(), "Non-finite physics state"
            trace.append((raw.sim.data.qpos[0].cpu().numpy().copy(), raw.sim.data.qvel[0].cpu().numpy().copy(), phase))

            p, _, _ = ap._root(raw, ap._DEFAULT_ASSET_CFG)
            h = 0.0 if phase == 0 else float(p[0, 2]) + HEIGHT_OFFSET
            heights.append(round(max(h, 0.0), 4))
            _qq = robot.data.root_link_quat_w[0]
            _uu = 1.0 - 2.0 * (float(_qq[1]) ** 2 + float(_qq[2]) ** 2)
            tilts.append(round(math.degrees(math.acos(max(-1.0, min(1.0, _uu)))), 1))

            if render:
                # entry -> climb over the walk-in, then ease back for the exit
                f1 = min(max(t / max(entry_end, 1e-6), 0.0), 1.0)
                f1 = f1 * f1 * (3.0 - 2.0 * f1)
                zf = (float(p[0, 2]) - (ex.PLATFORM_TOP_M - 1.2)) / 1.2
                f2 = min(max(zf, 0.0), 1.0)
                f2 = f2 * f2 * (3.0 - 2.0 * f2)
                def mix(k):
                    a1 = CAM_ENTRY[k] + (CAM_CLIMB[k] - CAM_ENTRY[k]) * f1
                    return a1 + (CAM_EXIT[k] - a1) * f2
                cam.azimuth, cam.elevation = mix("az"), mix("el")
                cam.distance = mix("dist")
                org = raw.scene.env_origins[0]
                want = [float(org[0]),
                        float(org[1]) + (1.0 - f1) * 0.5 * float(p[0, 1]) - f1 * f2 * 0.35,
                        float(org[2] + p[0, 2] + mix("dz"))]
                if smooth is None:
                    smooth = want
                smooth = [smooth[k] + 0.10 * (want[k] - smooth[k]) for k in range(3)]
                cam.lookat[:] = smooth
                rmodel.stat.center[:] = smooth
                for li in range(rmodel.nlight):
                    if int(rmodel.light_type[li]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                        rmodel.light_pos[li][2] = smooth[2] + 3.0
                frames.append(raw.render())
    _carf = float(em._carriage(raw)[0])
    print("STAND t_stand_start %.2f final_carriage %.3f final_tilt %.1f" % (t_stand_start, _carf, tilts[-1] if tilts else -1.0))
    np.savez_compressed(out_path.replace(".mp4", "_trajectory.npz"), qpos=np.stack([v[0] for v in trace]), qvel=np.stack([v[1] for v in trace]), phase=np.array([v[2] for v in trace]), mocap_pos=raw.sim.data.mocap_pos.cpu().numpy(), mocap_quat=raw.sim.data.mocap_quat.cpu().numpy(), dt=raw.step_dt)
    return t_entry_end, t_exit_start, frames, heights, raw.step_dt, float(p[0, 2]), tilts


TILT_GATE = float(os.getenv("TILT_GATE", "22.0"))     # deg, the exit's own posture
TILT_WAIT = float(os.getenv("TILT_WAIT", "0.6"))      # s, fall back if it never settles
TILT_RELAX = float(os.getenv("TILT_RELAX", "26.0"))   # deg, relaxed gate after TILT_WAIT
TILT_HARD = float(os.getenv("TILT_HARD", "1.2"))
STAND_CARRY = float(os.getenv("STAND_CARRY", "0.15"))   # hand to the stand policy when this tall...
STAND_TILT = float(os.getenv("STAND_TILT", "30.0"))     # ...and this upright, out past the line
FOOT_OPEN_M = float(os.getenv("FOOT_OPEN", "0.05"))  # open the window with the
# feet 5 cm clear of the platform; with TILT_HARD=1.2 s he can gain at most ~12 cm     # s, hand over regardless
ENTRY_END = float(os.getenv("ENTRY_END", "1.80"))   # measured: consistently ~1.75 s
if os.getenv("REPLAY"):
    import imageio.v2 as iio
    recording = np.load(os.environ["REPLAY"])
    raw = build()
    rm = raw._offline_renderer._model
    angle = math.radians(118)
    sun = np.array([.43 * math.cos(angle), .43 * math.sin(angle), -1.])
    fix_render_shadows(raw, min_extent=4., light_dir=sun)
    rm.stat.extent = 4.
    rm.vis.map.shadowclip = 4.
    rm.vis.quality.shadowsize = int(os.getenv("SHADOW_SIZE", "4096"))
    fixed_dirs = rm.light_dir.copy()
    cam = raw._offline_renderer._cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = -1
    raw.sim.data.mocap_pos[:] = torch.as_tensor(recording["mocap_pos"], device=DEV)
    raw.sim.data.mocap_quat[:] = torch.as_tensor(recording["mocap_quat"], device=DEV)
    phases = recording["phase"]
    switched = np.flatnonzero(phases > 0)
    entry_end = switched[0] * .02 if len(switched) else ENTRY_END
    heights = []
    audits = []
    physical = raw.sim.mj_model
    foot_ids = [mujoco.mj_name2id(physical, mujoco.mjtObj.mjOBJ_GEOM, "robot/" + side + "_foot_collision") for side in ("left", "right")]
    platform_id = mujoco.mj_name2id(physical, mujoco.mjtObj.mjOBJ_GEOM, "exit_platform/exit_platform_collision")
    if platform_id < 0:
        platform_id = next(g for g in range(physical.ngeom) if "exit_platform_collision" in (mujoco.mj_id2name(physical, mujoco.mjtObj.mjOBJ_GEOM,g) or ""))
    smooth = None
    origin_y = float(raw.scene.env_origins[0, 1])
    cleared = np.flatnonzero(-(recording["qpos"][:, 1] - origin_y) > ex.wall_end_y())
    orbit_start = max(0., cleared[0] * .02) if len(cleared) else float("inf")
    print("RIGHT_ORBIT_START", orbit_start, "duration", 6., flush=True)
    stride = int(os.getenv("REPLAY_STRIDE", "2"))
    first = int(os.getenv("REPLAY_FIRST", "0"))
    last = int(os.getenv("REPLAY_LAST", str(len(phases))))
    with iio.get_writer(out_path, fps=50 / stride, macro_block_size=1, codec="libx264", quality=8) as writer:
        for i in range(0, min(last, len(phases)), stride):
            p = recording["qpos"][i, :3] - raw.scene.env_origins[0].cpu().numpy()
            f1 = min(i * .02 / max(entry_end, .02), 1.)
            f1 = f1*f1*(3-2*f1)
            f2 = np.clip((p[2] - (ex.PLATFORM_TOP_M - .6)) / .6, 0, 1)
            f2 = f2*f2*(3-2*f2)
            def mix(k):
                a = CAM_ENTRY[k] + f1*(CAM_CLIMB[k]-CAM_ENTRY[k])
                return a + f2*(CAM_EXIT[k]-a)
            az_entry = CAM_ENTRY["az"] + f1 * (CAM_CLIMB["az"] - CAM_ENTRY["az"])
            # Six-second right reveal beginning at wall clearance.
            orbit = np.clip((i*.02-orbit_start)/6., 0., 1.)
            orbit = orbit*orbit*(3-2*orbit)
            cam.azimuth = az_entry + orbit * (140. - az_entry)
            cam.elevation, cam.distance = mix("el"), mix("dist")
            want = np.array([.5*p[0]*(1-f1), .5*p[1]*(1-f1)-.40*f2, p[2]+mix("dz")])
            # Same entry framing as reference; smooth the high-level exit view.
            smoothing = 1 - .8 ** (stride / 2)
            smooth = want if smooth is None or not f2 else smooth + smoothing*(want-smooth)
            if i < first:
                continue
            raw.sim.data.qpos[0] = torch.as_tensor(recording["qpos"][i], device=DEV)
            raw.sim.data.qvel[0] = torch.as_tensor(recording["qvel"][i], device=DEV)
            raw.sim.forward()
            nc = int(raw.sim.data.nacon[0])
            contacts = raw.sim.data.contact
            pairs = contacts.geom[:nc]
            platform_touch = (pairs == platform_id).any(-1) & (contacts.dist[:nc] <= 0)
            support = [bool((platform_touch & (pairs == gid).any(-1)).any()) for gid in foot_ids]
            quat = recording["qpos"][i, 3:7]
            tilt = math.degrees(math.acos(float(np.clip(1-2*(quat[1]**2+quat[2]**2),-1,1))))
            audits.append({"time": i*.02, "phase": int(phases[i]), "feet_on_platform": support,
                           "tilt_deg": tilt, "carriage_m": float(em._carriage(raw)[0]),
                           "speed_mps": float(np.linalg.norm(recording["qvel"][i,:3]))})
            cam.lookat[:] = smooth + raw.scene.env_origins[0].cpu().numpy()
            rm.stat.center[:] = cam.lookat
            for li in range(rm.nlight):
                if int(rm.light_type[li]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                    rm.light_pos[li] = cam.lookat + np.array([0.,0.,5.])
            assert np.array_equal(rm.light_dir, fixed_dirs), "Sun direction changed"
            writer.append_data(raw.render())
            heights.append(float(max(0., p[2]-.12)) if phases[i] else 0.)
            if i % 250 == 0:
                print("REPLAY", i, "of", len(phases), flush=True)
    json.dump({"dt": .02 * stride, "heights_m": heights, "sun_direction": fixed_dirs.tolist()}, open(out_path.replace(".mp4", "_heights.json"), "w"))
    json.dump(audits, open(out_path.replace(".mp4", "_audit.json"), "w"))
    print("REFERENCE_REPLAY_DONE", out_path, flush=True)
    raise SystemExit
if os.getenv("SWEEP", "0") != "0":
    te, tx, _f, hh, _dt, zf, tilts = run(False, ENTRY_END)
    print("SWEEP seed %d  entry_end %.2f  exit_start %.2f  final_z %.2f  max_cm %.1f"
          % (SEED, te, tx, zf, max(hh) * 100))
    raise SystemExit(0)
te, tx, frames, heights, dt, zf, tilts = run(True, ENTRY_END)
json.dump({"dt": dt, "heights_m": heights, "tilt_deg": tilts}, open(out_path.replace(".mp4", "_heights.json"), "w"))
import imageio.v2 as iio  # noqa: E402

iio.mimsave(out_path, frames, fps=50, macro_block_size=1)
print("FULL_RENDER %s frames %d  entry_end %.2f  exit_start %.2f  final_z %.2f  max_cm %.1f"
      % (out_path, len(frames), te, tx, zf, max(heights) * 100))

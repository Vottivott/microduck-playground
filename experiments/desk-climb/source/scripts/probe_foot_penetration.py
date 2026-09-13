"""Measure foot/tread interpenetration for a ladder checkpoint.

Two independent measures on env 0 of a CPU MuJoCo mirror of the GPU sim:
  * contact penetration: negative ``contact.dist`` for foot-side geoms vs tread geoms
  * vertex test: fraction/depth of sole + foot-upper + ankle-shell mesh vertices
    lying inside any tread box (catches overlaps contacts would not report).
"""
import sys, json, math
from dataclasses import asdict
import numpy as np, torch, mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner
import mjlab_microduck.tasks  # noqa

ck, level, n_envs, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
task = "Mjlab-LadderClimb-MicroDuck"
cfg = load_env_cfg(task, play=True); cfg.scene.num_envs = n_envs; cfg.terminations.clear()
cfg.events["reset_stair_ladder"].params["fixed_level"] = level
cfg.events["reset_stair_ladder"].params["floor_spawn_prob"] = 0.5
for c in cfg.commands.values():
    if hasattr(c, "debug_vis"): c.debug_vis = False
agent = load_rl_cfg(task)
raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
runner = (load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent), device="cuda:0")
runner.load(ck, map_location="cuda:0"); policy = runner.get_inference_policy(device="cuda:0")
m = raw.sim._mj_model if hasattr(raw.sim, "_mj_model") else raw.sim.mj_model
d = mujoco.MjData(m)
names = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""
foot_geoms = [g for g in range(m.ngeom) if any(k in names(mujoco.mjtObj.mjOBJ_GEOM, g) for k in ("foot", "ankle_shell", "sole"))]
tread_geoms = [g for g in range(m.ngeom) if names(mujoco.mjtObj.mjOBJ_GEOM, g).startswith("tread_") or "tread_" in names(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])]
print("foot geoms", [names(mujoco.mjtObj.mjOBJ_GEOM, g) for g in foot_geoms])
print("tread geoms", len(tread_geoms), "types", sorted(set(int(m.geom_type[g]) for g in tread_geoms)))
# mesh vertices of foot geoms (local frame)
foot_verts = {}
for g in foot_geoms:
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        foot_verts[g] = m.mesh_vert[a:a + n].copy()
obs = env.get_observations()
worst_contact = 0.0; worst_pair = None; worst_vertex = 0.0; worst_vpair = None
hist_contact = []; hist_vertex = []; clashes = {}
e0 = 0
for step in range(steps):
    with torch.inference_mode():
        obs, *_ = env.step(policy(obs))
    d.qpos[:] = raw.sim.data.qpos[e0].cpu().numpy(); d.qvel[:] = raw.sim.data.qvel[e0].cpu().numpy()
    if m.nmocap: d.mocap_pos[:] = raw.sim.data.mocap_pos[e0].cpu().numpy(); d.mocap_quat[:] = raw.sim.data.mocap_quat[e0].cpu().numpy()
    mujoco.mj_forward(m, d)
    step_c = 0.0
    for i in range(d.ncon):
        c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in foot_geoms and g2 in tread_geoms) or (g2 in foot_geoms and g1 in tread_geoms):
            pen = max(0.0, -float(c.dist)); step_c = max(step_c, pen)
            if pen > worst_contact: worst_contact, worst_pair = pen, (names(mujoco.mjtObj.mjOBJ_GEOM, g1), names(mujoco.mjtObj.mjOBJ_GEOM, g2), step)
    hist_contact.append(step_c)
    step_v = 0.0
    for g, verts in foot_verts.items():
        w = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        for t in tread_geoms:
            if m.geom_type[t] != mujoco.mjtGeom.mjGEOM_BOX: continue
            R = d.geom_xmat[t].reshape(3, 3); loc = (w - d.geom_xpos[t]) @ R  # into box frame
            size = m.geom_size[t]
            inside = np.all(np.abs(loc) < size, axis=1)
            if inside.any():
                depth = float(np.min(size - np.abs(loc[inside]), axis=1).max())  # how deep inside the box
                step_v = max(step_v, depth)
                if depth > worst_vertex: worst_vertex, worst_vpair = depth, (names(mujoco.mjtObj.mjOBJ_GEOM, g), names(mujoco.mjtObj.mjOBJ_GEOM, t) or names(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[t]), step, int(inside.sum()), len(verts))
    hist_vertex.append(step_v)
    # classify: for each foot side, the stance tread = tread with the most sole vertices inside/touching;
    # overlaps of ANY foot geom with a different tread are geometric clashes (heel into the tread
    # below, toe/shell into the tread above).
    for side in ("left", "right"):
        sole = [g for g in foot_verts if f"{side}_foot_collision" in names(mujoco.mjtObj.mjOBJ_GEOM, g)][0]
        w = foot_verts[sole] @ d.geom_xmat[sole].reshape(3, 3).T + d.geom_xpos[sole]
        counts = {}
        for t in tread_geoms:
            R = d.geom_xmat[t].reshape(3, 3); loc = (w - d.geom_xpos[t]) @ R; size = m.geom_size[t].copy(); size[2] += 0.002
            counts[t] = int(np.all(np.abs(loc) < size, axis=1).sum())
        stance = max(counts, key=counts.get) if max(counts.values()) > 0 else None
        for g, verts in foot_verts.items():
            if side not in names(mujoco.mjtObj.mjOBJ_GEOM, g): continue
            wg = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            for t in tread_geoms:
                if t == stance: continue
                R = d.geom_xmat[t].reshape(3, 3); loc = (wg - d.geom_xpos[t]) @ R; size = m.geom_size[t]
                inside = np.all(np.abs(loc) < size, axis=1)
                if inside.any():
                    depth = float(np.min(size - np.abs(loc[inside]), axis=1).max())
                    key = names(mujoco.mjtObj.mjOBJ_GEOM, g).split("/")[-1]
                    rec = clashes.setdefault(key, {"steps": 0, "max_mm": 0.0, "verts": 0})
                    rec["steps"] += 1; rec["max_mm"] = max(rec["max_mm"], 1e3 * depth); rec["verts"] = max(rec["verts"], int(inside.sum()))
hc, hv = np.array(hist_contact), np.array(hist_vertex)
print("PROBE contact penetration mm: max %.2f p95 %.2f mean-when-touching %.2f  worst=%s" % (1e3*hc.max(), 1e3*np.percentile(hc,95), 1e3*hc[hc>0].mean() if (hc>0).any() else 0, worst_pair))
print("PROBE vertex-inside-tread mm: max %.2f p95 %.2f steps-with-any %.0f%%  worst=%s" % (1e3*hv.max(), 1e3*np.percentile(hv,95), 100*(hv>0).mean(), worst_vpair))

print("PROBE clashes with NON-stance treads (per foot geom): %s  (of %d steps)" % (json.dumps(clashes), steps))
print("PROBE timestep %.4f  tread solref %s solimp %s  foot solref %s" % (m.opt.timestep, m.geom_solref[tread_geoms[0]].tolist(), m.geom_solimp[tread_geoms[0]].tolist(), m.geom_solref[foot_geoms[0]].tolist()))

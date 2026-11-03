"""The back of recent Microduck units, for renders: the official model's battery reaches the top of the body, while on
recent units the white shell halves cover the back above a lowered battery, with a tight cut-out around the USB-C port
(the port sits slightly deeper inside). This builds that region as one visual mesh in the trunk_base frame: the filled
area, the cut-out and a recessed floor with the receptacle's opening.

The filled area continues the official shells' own back surface. It is ray-cast from the shell meshes; each half is a
panel angled about 3 degrees in plan (they meet in a shallow ridge at the seam), curved in z. The residual from that
reference is smoothed and filled biharmonically across the opening, so the fill matches the surrounding surface's
height and slope with no crease.

  uv run --with manifold3d --with trimesh python hardware/microdock/sim/back_patch.py OUT.obj   (needs MICRODUCK_RL_DIR)
The mesh sits 0.05 mm proud of the official shells, so it draws over the old back without editing the robot's meshes.
"""
import os, sys
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spl
import manifold3d as m3
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "source"))
import dock as dk


def shell_meshes(model):
    """The official shell halves in the trunk_base frame (mm): {mesh name: (vertices, faces)}."""
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"); out = {}
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != tb or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mid = model.geom_dataid[g]; name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        if name not in ("left_shell", "right_shell") or name in out: continue
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
        f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        R = np.zeros(9); mujoco.mju_quat2Mat(R, model.geom_quat[g]); R = R.reshape(3, 3)
        out[name] = ((v @ R.T + model.geom_pos[g])*1000.0, f.copy())
    return out


def back_patch(shells, p: dk.Params, verbose=False):
    """Vertices (m) and faces of the new back region in the trunk_base frame."""
    import trimesh
    from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    # ---- 1) the shells' back surface x = h(y, z), filled across the battery opening
    both = trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f in shells.values()])
    G = 0.2; ys = np.arange(-23.0, 23.0 + 1e-6, G); zs = np.arange(17.0, 36.0 + 1e-6, G)
    Y, Z = np.meshgrid(ys, zs)
    org = np.stack([np.full(Y.size, -70.0), Y.ravel(), Z.ravel()], 1)
    loc, ri, _ = both.ray.intersects_location(org, np.tile([1.0, 0, 0], (Y.size, 1)), multiple_hits=False)
    hx = np.full(Y.size, np.nan); hx[ri] = loc[:, 0]; hx = hx.reshape(Y.shape)
    # Each half's back is a panel angled ~3 deg in plan: the halves meet in a shallow ridge at the seam (measured on the
    # band above the opening, where the shell spans the whole width). Reference surface: x = ref(z) + KY*(|y| - 21).
    band = (Z >= 32.8) & (Z <= 34.0) & (np.abs(Y) >= 2.0) & (np.abs(Y) <= 18.0) & ~np.isnan(hx)
    KY = float(np.polyfit(np.abs(Y[band]), hx[band] - np.array([np.nanmedian(hx[j][(np.abs(ys) >= 2) & (np.abs(ys) <= 18)]) for j in range(len(zs))])[:, None].repeat(len(ys), 1)[band], 1)[0])
    side = (np.abs(Y) >= 21.2) & (np.abs(Y) <= 22.8)
    ref = np.array([np.nanmedian((hx[j] - KY*(np.abs(ys) - 21.0))[side[j]]) for j in range(len(zs))])   # profile x(z) at |y| = 21
    # the official mesh is faceted (irregular triangles ~1-2.5 mm): smooth the profile so the fill has no facet kinks
    okr = ~np.isnan(ref)
    ref = UnivariateSpline(zs[okr], ref[okr], k=3, s=okr.sum()*0.012**2)(zs)
    REF = ref[:, None] + KY*(np.abs(Y) - 21.0)
    valid = np.abs(hx - REF) < 0.25
    valid &= ~((np.abs(Y) < 21.8) & (Z < 34.4))           # the whole replaced area: always filled, never raw facets
    valid &= ~(np.abs(Y) < 0.6)                            # the seam groove reads deeper than the surface
    # Fill the RESIDUAL from that reference, biharmonically (least-squares bilaplacian): it matches the surrounding
    # surface's height and slope at the edge of the hole. Below the shells' new edge (z < 21, where the battery sits)
    # there is no shell: pin the fill to the panel shape there, so the unconstrained band cannot drift.
    anchor = (np.abs(Y) < 21.8) & (Z <= 21.0)
    res = np.where(valid, hx - REF, 0.0)
    wv = gaussian_filter(valid.astype(float), 4.0); res = np.where(valid, gaussian_filter(res*valid, 4.0)/np.maximum(wv, 1e-6), 0.0)
    valid = valid | anchor
    nz_, ny_ = Y.shape
    def lap1d(n):
        m = sp.lil_matrix((n, n))
        for i in range(n):
            if 0 < i < n - 1: m[i, i - 1] = 1; m[i, i] = -2; m[i, i + 1] = 1
        return m.tocsr()
    Lap = sp.kron(sp.eye(nz_), lap1d(ny_)) + sp.kron(lap1d(nz_), sp.eye(ny_))      # interior Laplacian (grid-edge rows are zero)
    U = np.flatnonzero(~valid.ravel()); K = np.flatnonzero(valid.ravel())
    LU, LK = Lap[:, U], Lap[:, K]
    A = (LU.T @ LU).tocsc(); b = -(LU.T @ (LK @ res.ravel()[K]))
    r_ = res.ravel().copy(); r_[U] = spl.spsolve(A + 1e-9*sp.eye(len(U)), b); r_ = r_.reshape(Y.shape)
    h = REF + r_
    hf = RegularGridInterpolator((zs, ys), h, bounds_error=False, fill_value=None)
    def H_(y, z): return float(hf([[z, y]])[0])
    if verbose:
        print("panel angle in plan: slope %.4f (%.2f deg); fill residual %.3f..%.3f mm" % (KY, np.degrees(np.arctan(KY)), r_.ravel()[U].min(), r_.ravel()[U].max()))

    # ---- 2) the filled area for each half, with the cut-out, as a slab following h
    ZB = p.bat_top + 0.3                          # the shells' new bottom edge (battery opening's top edge), just above the battery
    PZ = p.port_z; NT = PZ + p.notch_above_port; NW = p.notch_w/2
    T_SLAB = 2.6; OUT = -0.05                     # slab thickness; outer face 0.05 mm proud of the official shell
    PY1, PZ1 = 21.8, 34.4                         # replaced region: beyond the old opening and its rounded edge
    def slab(y0, y1, z0, z1, g=0.2):
        yy = np.linspace(y0, y1, int(round((y1 - y0)/g)) + 1); zz = np.linspace(z0, z1, int(round((z1 - z0)/g)) + 1)
        ny, nz = len(yy), len(zz); YY, ZZ = np.meshgrid(yy, zz)
        hh = hf(np.stack([ZZ.ravel(), YY.ravel()], 1)).reshape(ZZ.shape)
        vo = np.stack([hh + OUT, YY, ZZ], -1).reshape(-1, 3)
        vi = np.stack([hh + T_SLAB, YY, ZZ], -1).reshape(-1, 3)
        V = np.vstack([vo, vi]); n = ny*nz; F = []
        def q(a, b_, c, d, flip=False):
            F.extend([(a, b_, c), (a, c, d)] if not flip else [(a, c, b_), (a, d, c)])
        for j in range(nz - 1):
            for i in range(ny - 1):
                a = j*ny + i; q(a, a + 1, a + ny + 1, a + ny); q(n + a, n + a + 1, n + a + ny + 1, n + a + ny, flip=True)
        ring = [j*ny for j in range(nz)][::-1] + [i for i in range(ny)] + [j*ny + ny - 1 for j in range(nz)] + [(nz - 1)*ny + i for i in range(ny)][::-1]
        ring = [ring[0]] + [r for k, r in enumerate(ring[1:], 1) if r != ring[k - 1]]
        if ring[-1] == ring[0]: ring = ring[:-1]
        for k in range(len(ring)):
            a, b_ = ring[k], ring[(k + 1) % len(ring)]; q(a, b_, n + b_, n + a, flip=True)
        man = m3.Manifold(m3.Mesh(V.astype(np.float32), np.array(F, np.uint32)))
        if man.volume() < 0 or man.status() != m3.Error.NoError:
            man = m3.Manifold(m3.Mesh(V.astype(np.float32), np.array(F, np.uint32)[:, ::-1].copy()))
        assert man.status() == m3.Error.NoError and man.volume() > 0, man.status()
        return man
    def arc(cx, cy, r, a0, a1, n=24): return [(cx + r*np.cos(np.radians(a0 + (a1 - a0)*k/n)), cy + r*np.sin(np.radians(a0 + (a1 - a0)*k/n))) for k in range(n + 1)]
    RC, RF = 0.5, 2.4                              # cut-out top corner radius, flare radius into the opening edge
    def cutout_outline(sgn):
        """The cut-out in the (y, z) plane for one half (sgn=+1 left, y >= 0): straight sides from the top edge down,
        then curving outward (radius RF) into the opening's top edge."""
        pts = [(0.0, ZB - 1.0), (0.0, NT)]
        pts += arc(NW - RC, NT - RC, RC, 90, 0)
        pts += arc(NW + RF, ZB + RF, RF, 180, 270)[:-1] + [(NW + RF, ZB - 1.0)]
        return [(sgn*y, z) for y, z in pts]
    def _ccw(poly): a = np.array(poly); return 0.5*np.sum(a[:, 0]*np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1)*a[:, 1]) > 0
    def prism_yz(poly, x0, x1):
        """Extrude a (y, z) polygon along x from x0 to x1."""
        pl = poly if _ccw(poly) else poly[::-1]
        ex = m3.CrossSection([[(float(a), float(b_)) for a, b_ in pl]], m3.FillRule.Positive).extrude(x1 - x0)   # (y, z, x - x0)
        return ex.transform([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]])
    def stadium_yz(L, W, cy, cz, n=24):
        r = W/2; c = L/2 - r
        return arc(c + cy, cz, r, -90, 90, n) + arc(-c + cy, cz, r, 90, 270, n)

    X_FLOOR = H_(0, PZ) + p.port_recess           # recessed floor = the receptacle's face
    SEAM = 0.15
    V_all, F_all = [], []
    for sgn in (1, -1):
        y0, y1 = (SEAM, PY1) if sgn > 0 else (-PY1, -SEAM)
        fill = slab(y0, y1, ZB, PZ1) - prism_yz(cutout_outline(sgn), -60.0, -30.0)
        fy0, fy1 = (SEAM, NW + RF + 0.3) if sgn > 0 else (-(NW + RF + 0.3), -SEAM)
        floor = prism_yz([(fy0, ZB), (fy1, ZB), (fy1, NT + 0.5), (fy0, NT + 0.5)], X_FLOOR, X_FLOOR + 1.4)
        floor = floor - prism_yz(stadium_yz(8.92, 3.14, 0.0, PZ), X_FLOOR - 1.0, X_FLOOR + 2.0)   # the receptacle's opening
        m = (fill + floor).to_mesh(); v = np.asarray(m.vert_properties)[:, :3]/1000.0; f = np.asarray(m.tri_verts)
        F_all.append(f + sum(len(q) for q in V_all)); V_all.append(v)
    return np.vstack(V_all), np.vstack(F_all)


def build(robot_dir, p: dk.Params, verbose=False):
    """Shells from the official robot model (as the docking simulation loads it), then the patch."""
    model = dk.modify_robot(dk.build_scene(p, robot_dir), p).compile()
    return back_patch(shell_meshes(model), p, verbose=verbose)


if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    RL = os.environ.get("MICRODUCK_RL_DIR", "")
    V, F = build(os.path.join(RL, "src", "mjlab_microduck", "robot", "microduck"), dk.Params(), verbose=True)
    with open(sys.argv[1], "w") as fo:
        fo.writelines(f"v {x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in V); fo.writelines(f"f {a+1} {b+1} {c+1}\n" for a, b, c in F)
    print("back patch (trunk frame, m):", V.shape, F.shape, "x %.4f..%.4f y %.4f..%.4f z %.4f..%.4f" % (V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max(), V[:, 2].min(), V[:, 2].max()))

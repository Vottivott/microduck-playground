import numpy as np,trimesh,json
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from pathlib import Path
import argparse
a=argparse.ArgumentParser();a.add_argument('--directory',default='friction-fit');args=a.parse_args();p=Path(__file__).parent/args.directory;out={}
for name in sorted(x.stem for x in p.glob('module-??.stl')):
 m=trimesh.load(p/(name+'.stl'));v=m.convex_hull.vertices
 def f(a):return np.ptp(v@Rotation.from_euler('xyz',a).as_matrix().T,axis=0).max()
 r=differential_evolution(f,[(-np.pi,np.pi)]*3,seed=42,popsize=20,maxiter=200,tol=1e-6);rot=Rotation.from_euler('xyz',r.x);m.vertices=m.vertices@rot.as_matrix().T;m.apply_translation(-m.bounds[0]);m.export(p/(name+'-bed-oriented.stl'));out[name]={'rotation_xyz_deg':np.degrees(r.x).tolist(),'bounds_mm':m.extents.tolist(),'geometric_fit_256cube':bool(max(m.extents)<256),'supports_not_sliced':True};print(name,m.extents,flush=True)
if (p/'floor-0.stl').exists():
 name='floor-0';m=trimesh.load(p/(name+'.stl'));m.vertices=m.vertices@Rotation.from_euler('z',45,degrees=True).as_matrix().T;m.apply_translation(-m.bounds[0]);m.export(p/(name+'-bed-oriented.stl'));out[name]={'rotation_xyz_deg':[0,0,45],'bounds_mm':m.extents.tolist(),'geometric_fit_256cube':bool(max(m.extents)<256),'supports_not_sliced':True}
(p/'bed-orientation.json').write_text(json.dumps(out,indent=2))

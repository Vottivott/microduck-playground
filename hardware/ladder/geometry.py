"""Build evaluated scene primitives without exporting obsolete prototype parts."""
import numpy as np,trimesh
from scipy.spatial.transform import Rotation

def mesh(r):
 if r['type']=='mesh':m=trimesh.convex.convex_hull(np.asarray(r['vertices']))
 elif r['type']=='box':m=trimesh.creation.box(extents=np.asarray(r['size'])*2)
 elif r['type']=='cylinder':m=trimesh.creation.cylinder(radius=r['size'][0],height=r['size'][1]*2,sections=48)
 else:raise ValueError(r['type'])
 if r.get('quat'):
  q=r['quat'];tf=np.eye(4);tf[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix();m.apply_transform(tf)
 m.apply_translation(r.get('pos',[0,0,0]));return m

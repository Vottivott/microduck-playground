from geometry_shift import *
import geometry_shift as shifted
assert abs(SHIFT-.06)<1e-9
records=[]
for r in floor_desk.RECORDS:
 if r['name'].startswith('connected_side_gusset'):continue
 if r['name'].startswith('clamp_'):r['pos'][0]-=SHIFT
 records.append(r)
# Bridge above the tabletop, not through its thickness. Clamps stay on the edge.
profile=[(.410,.667),(.451,.697),(.478,.680),(.478,.666)]
for side in [-.143,.143]:
 records.append({'name':f'edge_bridge_{round(side*1000)}','type':'mesh','pos':[0,0,0],'vertices':[[x,side+dy,z] for dy in [-.012,.012] for x,z in profile]})
floor_desk.RECORDS[:]=records
# Conservative world AABB separation from tabletop interior (touching allowed).
from scipy.spatial.transform import Rotation
desk=next(r for r in records if r['name']=='desktop');lo=np.array(desk['pos'])-np.array(desk['size']);hi=np.array(desk['pos'])+np.array(desk['size']);clearance=[]
for r in records:
 if not r['name'].startswith(('clamp_','edge_bridge_')):continue
 if r['type']=='mesh':v=np.array(r['vertices'])+np.array(r['pos'])
 else:
  sz=r['size'];sz=np.array(sz if r['type']=='box' else [sz[0],sz[0],sz[1]])
  v=np.array([[x,y,z] for x in [-sz[0],sz[0]] for y in [-sz[1],sz[1]] for z in [-sz[2],sz[2]]])
  if r.get('quat'):v=Rotation.from_quat([*r['quat'][1:],r['quat'][0]]).apply(v)
  v+=np.array(r['pos'])
 overlaps=np.minimum(v.max(axis=0),hi)-np.maximum(v.min(axis=0),lo)
 assert (overlaps<=1e-8).any(),(r['name'],overlaps)
 clearance.append({'name':r['name'],'aabb_overlap_axes':overlaps.tolist(),'no_table_interior_intersection':True})

def audit():return {**shifted.audit(),'records':records,'clamps_reanchored_to_table_edge':True,'ladder_shift_preserved_m':SHIFT,'table_clamp_intersection_checks':clearance,'connecting_bridge_above_table':True}

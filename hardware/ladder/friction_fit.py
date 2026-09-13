"""Generate current pin-free rail-axis joints and matching floor docking.
Run generate_clamps.py first to supply the unchanged threaded clamp parts.
All dimensions mm. No physical fit/load validation is implied.
"""
import json,math,shutil,hashlib,os
from pathlib import Path
import numpy as np,trimesh as tm
import cad as c
from cad import box,cyl,union,diff,intersect,vol,moved,export_checked
P=Path(__file__).parent;O=P/('friction-fit' if c.args.treads_per_module==6 else 'friction-fit-3step');O.mkdir(exist_ok=True)
c.O=O;c.manifest={'revision':'pin-free friction rail ends','treads_per_module':c.args.treads_per_module,'units':'millimetres','parts':{},'assembly':{},'checks':{}};c.scenes={};manifest=c.manifest;scenes=c.scenes;save=c.save
axis=np.array([math.cos(math.radians(62)),0,math.sin(math.radians(62))]);normal=np.array([axis[2],0,-axis[0]]);B=np.column_stack([normal,[0,1,0],axis]);T=np.eye(4);T[:3,:3]=B

def orient(m,point):m=m.copy();m.apply_transform(T);m.apply_translation(point);return m

def obox(ext,point):return orient(box(ext,[0,0,0]),point)
def half(point,upper):return obox([2500,2500,2500],point+axis*(1250 if upper else -1250))
def center_at(rail,point):
 section=intersect(rail,obox([2500,30,.1],point));verts=section.vertices@B
 loc=(verts.min(0)+verts.max(0))/2;loc[2]=point@axis;return loc@B.T

# Side ribs take up the running clearance with a small nominal interference.
# This is a fit-coupon starting point, not a measured pull-out-force specification.
CLEARANCE=.30
INTERFERENCE=float(os.environ.get('FRICTION_INTERFERENCE_MM','.05'))
assert 0<=INTERFERENCE<=.15,'Choose interference from 0 to 0.15mm per face'

def tenon(dimensions,interference=INTERFERENCE):
 w,h=dimensions
 core=union([box([w,h,11],[0,0,4.5]),tm.convex.convex_hull(np.array([[sx*(w/2-inset),sy*(h/2-inset),z]for z,inset in [(10,0),(12,.8)]for sx in [-1,1]for sy in [-1,1]]))])
 ribs=[]
 for x in [-w*.23,w*.23]:
  for sign in [-1,1]:
   points=[]
   for z,extra in [(1,0),(3,CLEARANCE+interference),(9,CLEARANCE+interference),(11,0)]:
    for dx in [-.6,.6]:
     for offset in [-.1,extra]:points.append([x+dx,sign*(h/2+offset),z])
   ribs.append(tm.convex.convex_hull(np.array(points)))
 return core,union(ribs)

def ascii_export(m,path):
 # ASCII preserves coordinates at the precision needed for nearly coplanar
 # split faces; binary STL float32 can collapse their narrow triangles.
 m=m.copy();m.merge_vertices(digits_vertex=7);m.update_faces(m.nondegenerate_faces());m.update_faces(m.unique_faces());m.remove_unreferenced_vertices()
 assert m.is_watertight and m.is_winding_consistent,(str(path),'pre-export cleanup')
 m.export(path,file_type='stl_ascii')
 check=tm.load(path);assert check.is_watertight and check.is_winding_consistent,(str(path),'ASCII roundtrip')
c.export_checked=ascii_export
export_checked=ascii_export

def main():
 manifest['source_geometry_sha256']=hashlib.sha256((P/'evaluated-geometry.json').read_bytes()).hexdigest()
 j=json.loads((P/'evaluated-geometry.json').read_text());rs=[r for r in j['records']if not(r['name']in ['landing_plate','landing_taper']or(r['name'].startswith('tread_')and int(r['name'].split('_')[1])>=27))]
 meshes={r['name']:c.evaluated.mesh(r) for r in rs}
 for m in meshes.values():m.apply_scale(1000)
 rails={y:union([m for n,m in meshes.items()if n.startswith('spine_' if y==0 else f'rail_{y}_')])for y in [-143,0,143]}
 ladder=union([m for n,m in meshes.items()if n.startswith(('tread_','root_','rail_','spine_','bridge_'))]);treads=union([m for n,m in meshes.items()if n.startswith('tread_')])
 # Common section planes perpendicular to the straight rail direction. Keep all
 # stepping surfaces; cutting/reuniting solids may retriangulate their interior.
 cuts=[]
 for i in range(c.args.treads_per_module,27,c.args.treads_per_module):
  previous=meshes[f'tread_{i-1:02}'].vertices@axis;following=meshes[f'tread_{i:02}'].vertices@axis
  assert following.min()>previous.max(),'No tread-free section plane'
  cuts.append(axis*((previous.max()+following.min())/2))
 modules=[];remainder=ladder
 for pt in cuts:modules.append(intersect(remainder,half(pt,False)));remainder=intersect(remainder,half(pt,True))
 modules.append(remainder);jointdata=[];sockets=[];floorpegs={};allribs=[]
 def joint(lower,upper,point,y,label):
  centre=center_at(rails[y],point+np.array([0,y,0]));depth=12;dimensions=[14,7]if y else[9,5]
  core,ribs=tenon(dimensions);core=orient(core,centre);ribs=orient(ribs,centre)
  if label!='floor':core=intersect(core,rails[y])
  peg=union([core,ribs]);allribs.append(ribs)
  socket=obox([dimensions[0]+2*CLEARANCE,dimensions[1]+2*CLEARANCE,depth+1],centre+axis*(depth-.4)/2);sockets.append(socket)
  lower=union([lower,peg]);upper=diff(upper,socket)
  if label=='floor':floorpegs[y]=peg
  jointdata.append({'name':label,'rail_y':y,'center_mm':centre.tolist(),'insertion_axis':axis.tolist(),'nominal_tenon_mm':dimensions+[depth],'per_face_running_clearance_mm':CLEARANCE,'rib_interference_per_face_mm':INTERFERENCE,'retention':'friction ribs; no pin, hole or positive lock'})
  return lower,upper
 for n,pt in enumerate(cuts):
  for y in [-143,0,143]:modules[n],modules[n+1]=joint(modules[n],modules[n+1],pt,y,f'joint-{n+1}')
 # Floor modules dock using the same three integral rail-end joints. Docking
 # plane lies above the very bottom of the evaluated rails, without moving treads.
 floorparts={}
 for y in [-143,0,143]:
  basepoint=axis*(float((rails[y].vertices@axis).min())+.10)
  old=union([meshes[f'base_shoe_{y}'],meshes[f'base_root_{y}']])if y else union([meshes['base_crossbar'],meshes['base_center']])
  base=diff(old,modules[0])
  base,modules[0]=joint(base,modules[0],basepoint,y,'floor')
  floorparts[y]=base
 # Crossbar slides into the shoes; its tongues are captured once the three
 # ladder rails are docked. No overlapping solids masquerading as joints.
 bar=intersect(floorparts[0],box([2500,224,2500],[0,0,0]))
 for y in [-143,143]:
  sign=np.sign(y)
  bar=union([bar,box([16,24,3.5],[90,sign*123,6])])
  floorparts[y]=diff(floorparts[y],box([16.6,24.6,4.1],[90,sign*123,6]))
 floorparts[0]=bar
 # Clear the docking approach where the old monolithic feet wrapped around
 # the first module. These cuts affect the shoes, never the tread solids.
 for y in [-143,0,143]:
  collisions=[intersect(floorparts[y],moved(modules[0],axis*d))for d in np.arange(.5,14.5,.5)]
  cutters=[]
  for hit in collisions:
   if vol(hit)<1e-6:continue
   hit=diff(hit,union(allribs))
   if vol(hit)>.01:
    # Retain exact male geometry after relieving the shoe approach.
    cutters.append(box(hit.extents+.4,hit.bounds.mean(0)))
  if cutters:floorparts[y]=union([diff(floorparts[y],union(cutters).convex_hull),floorpegs[y]])
 floorparts[0]=union([floorparts[0],box([45,24,4],[75,0,6])])
 combined=union(modules);rib_union=union(allribs)
 def unintended(a,b):
  hit=intersect(a,b)
  return 0.0 if vol(hit)<1e-6 else vol(diff(hit,rib_union))
 manifest['checks']['socket_tread_intersection_mm3']=vol(intersect(treads,union(sockets)));assert manifest['checks']['socket_tread_intersection_mm3']<.1
 manifest['checks']['missing_tread_volume_mm3']=vol(diff(treads,combined));assert manifest['checks']['missing_tread_volume_mm3']<.2,manifest['checks']['missing_tread_volume_mm3']
 manifest['checks']['individual_tread_missing_from_single_module_mm3']=[min(vol(diff(tread,m))for m in modules)for name,tread in meshes.items()if name.startswith('tread_')]
 assert max(manifest['checks']['individual_tread_missing_from_single_module_mm3'])<.1
 manifest['checks']['module_overlap_mm3']=[unintended(a,b)for a,b in zip(modules,modules[1:])];assert max(manifest['checks']['module_overlap_mm3'])<.1
 manifest['checks']['floor_module_overlap_mm3']=[unintended(m,combined)for m in floorparts.values()];assert max(manifest['checks']['floor_module_overlap_mm3'])<.1
 for i,m in enumerate(modules):save(f'module-{i+1:02}',m,m)
 for y,m in floorparts.items():
  save(f'floor-{y}',m,m)
 # Threaded clamps use the shared current functional set.
 old=json.loads((P/'clamps/manifest.json').read_text())
 for name in ['clamp-screw','pressure-pad','clamp-body--143','clamp-body-143','thread-fit-coupon']:
  shutil.copy2(P/'clamps'/f'{name}.stl',O/f'{name}.stl');manifest['parts'][name]=old['parts'][name]
 for name in ['clamp-body--143','clamp-body-143','screw--143','screw-143','pad--143','pad-143']:
  shutil.copy2(P/'clamps'/f'{name}-assembly.stl',O/f'{name}-assembly.stl');scenes[name]=tm.load(O/f'{name}-assembly.stl')
 manifest['assembly']['joints']=jointdata;manifest['assembly']['thread']=old['assembly']['thread']
 manifest['assembly']['fit']=f'Slide along rails until shoulders seat. Four narrow ribs per tenon have nominal {INTERFERENCE:g}mm per-face interference. Print coupons before choosing fit; no positive retention.'
 manifest['checks']['module_insertion_unintended_overlap_mm3']=[unintended(a,moved(b,axis*d))for a,b in zip(modules,modules[1:])for d in [0,2,5,10,15,20]]
 assert max(manifest['checks']['module_insertion_unintended_overlap_mm3'])<.1
 floor_union=union(list(floorparts.values()))
 manifest['checks']['floor_insertion_unintended_overlap_mm3']=[unintended(floor_union,moved(modules[0],axis*d))for d in [0,2,5,10,15,20]]
 assert max(manifest['checks']['floor_insertion_unintended_overlap_mm3'])<.1
 manifest['checks']['floor_pair_overlap_mm3']=[vol(intersect(floorparts[a],floorparts[b]))for a,b in [(-143,0),(0,143),(-143,143)]]
 assert max(manifest['checks']['floor_pair_overlap_mm3'])<.1
 manifest['checks']['intentional_rib_interference_mm3']=[vol(intersect(a,b))for a,b in zip(modules,modules[1:])]+[vol(intersect(m,combined))for m in floorparts.values()]
 manifest['fit_coupon_variants_mm_per_face']=[0,.025,.05,.10]
 for value in manifest['fit_coupon_variants_mm_per_face']:
  for label,dimensions in [('side',[14,7]),('spine',[9,5])]:
   core,ribs=tenon(dimensions,value);save(f'coupon-{label}-{round(value*1000):03}-tenon',union([core,ribs,box([dimensions[0]+8,dimensions[1]+6,4],[0,0,-2])]))
 for label,dimensions in [('side',[14,7]),('spine',[9,5])]:
  save(f'coupon-{label}-socket',diff(box([dimensions[0]+8,dimensions[1]+6,16],[0,0,8]),box([dimensions[0]+.6,dimensions[1]+.6,12.6],[0,0,9.7])))
 for name,m in scenes.items():export_checked(m,O/f'{name}-assembly.stl')
 scene=tm.Scene()
 for name,m in scenes.items():scene.add_geometry(m,node_name=name,geom_name=name)
 scene.export(O/'hardware-assembled.glb');manifest['status']='Pin-free friction-fit prototype, not physically fit/load-tested. Exact evaluated tread geometry retained within boolean precision.'
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['checks'],indent=2))
if __name__=='__main__':main()

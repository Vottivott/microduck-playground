"""Generate the current printed table clamps, screws, pads and thread coupon."""
from cad import *

def main():
 j=json.loads((P/'evaluated-geometry.json').read_text())
 meshes={r['name']:evaluated.mesh(r) for r in j['records']}
 for m in meshes.values():m.apply_scale(1000)
 # Clamp body: keep top contact face, corrected edge backbone and rail bridge.
 # Join the old floating bridge to the top jaw with a rib ABOVE the tabletop.
 screw=union([helix(),cyl(18,8,-4),cyl(3,6,69),moved(tm.creation.icosphere(subdivisions=3,radius=5),[0,0,74])])
 # Four lobes give finger purchase without a separate knob fastener.
 screw=union([screw]+[moved(cyl(5,8,-4),[16*math.cos(a),16*math.sin(a),0])for a in np.arange(4)*math.pi/2])
 save('clamp-screw',screw)
 # Snap socket: broad flat pressure face, spherical cavity and four flex slots.
 pad=union([cyl(18,5,7.5),cyl(8,9,.5)])
 cavity=moved(tm.creation.icosphere(subdivisions=3,radius=5+GAP),[0,0,0]);cavity=union([cavity,cyl(4.65,12,-6)])
 pad=diff(pad,cavity)
 for angle in [0,math.pi/2]:
  slot=box([18,1,6],[0,0,-2]);tf=tm.transformations.rotation_matrix(angle,[0,0,1]);slot.apply_transform(tf);pad=diff(pad,slot)
 save('pressure-pad',pad)
 manifest['checks']['screw_pad_intersection_mm3']=vol(intersect(screw,moved(pad,[0,0,74])))
 assert manifest['checks']['screw_pad_intersection_mm3']<.1
 # Nominal 25mm desktop: pad face 635, upper jaw contact660.
 # Ball center625 => screw bottom551, thread engagement588..602 (14mm).
 threadcutter=helix(100,GAP)
 clampchecks=[]
 for side in [-143,143]:
  upper=meshes[f'clamp_upper_{side}'];cx=meshes[f'clamp_screw_{side}'].centroid[0];y=side
  body=union([m for name,m in meshes.items()if name.startswith('clamp_')and name.endswith('_'+str(side))and not any(s in name for s in ['screw','pad'])]+[meshes[f'edge_bridge_{side}'],box([40,24,10],[435,y,672])])
  # Thread at same global phase as assembled screw; no radial collision on turns.
  body=diff(body,moved(threadcutter,[cx,y,551]))
  save(f'clamp-body-{side}',body,body);scenes[f'screw-{side}']=moved(screw,[cx,y,551]);scenes[f'pad-{side}']=moved(pad,[cx,y,625])
  for travel in np.arange(-15,10.01,.5):
   angle=travel/4*2*math.pi;tf=tm.transformations.rotation_matrix(angle,[0,0,1]);test=screw.copy();test.apply_transform(tf);test.apply_translation([cx,y,551+travel]);v=vol(intersect(test,body));assert v<.1,(travel,v);clampchecks.append(v)
  # Outer body/table overlap must remain zero at evaluated25mm tabletop.
  desk=union([m for name,m in meshes.items()if name=='desktop'])
  manifest['checks'][f'clamp_table_intersection_{side}']=vol(intersect(body,desk));assert vol(intersect(body,desk))<.1
 # Coupon matches14mm jaw, including female helical clearance.
 coupon=diff(box([28,28,14],[0,0,7]),moved(threadcutter,[0,0,-8]));save('thread-fit-coupon',coupon)
 manifest['checks']['screw_body_intersection_mm3']=clampchecks
 manifest['checks']['evaluated_geometry_sha256']=hashlib.sha256((P/'evaluated-geometry.json').read_bytes()).hexdigest()
 manifest['assembly']['thread']={'pitch_mm':4,'major_diameter_mm':17,'root_diameter_mm':13.6,'radial_clearance_mm':GAP,'axial_flank_allowance_mm':GAP,'jaw_engagement_mm':14,'nominal_table_mm':25,'designed_table_range_mm':[15,40],'ball_radius_mm':5,'pad_diameter_mm':36,'pad_capture':'four-slotted snap socket; print fit must be tested'}
 # World-coordinate scene parts are separate from origin-normalized print STLs.
 scene=tm.Scene()
 for n,m in scenes.items():scene.add_geometry(m,node_name=n,geom_name=n)
 scene.export(O/'hardware-assembled.glb')
 for n,m in scenes.items():export_checked(m,O/(n+'-assembly.stl'))
 (O/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('Generated current threaded clamps in',O)

if __name__=='__main__':main()

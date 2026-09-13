import numpy as np,math,copy,json
from scipy.spatial.transform import Rotation
from mjlab_microduck.robot import floor_desk
from mjlab_microduck.tasks import mdp
OLD_Z=np.arange(1,31)*.024
RUN=.024/math.tan(math.radians(62))
angles=[62.]*24+[56.,48.,40.,32.,24.,16.]
# Equal 3D chord length between corresponding rail stations; lateral alternating
# offset is unchanged, so alternating tread-center distances are also equal.
SPACING=.720/np.sin(np.radians(angles)).sum()
NEW_Z=np.cumsum(SPACING*np.sin(np.radians(angles)))
NEW_X=np.cumsum(SPACING*np.cos(np.radians(angles)))
DX=NEW_X-np.arange(1,31)*RUN
DELTA=float(DX[-1]);assert abs(NEW_Z[-1]-.720)<1e-9
PITCH=np.radians(62.-np.asarray(angles))
assert np.allclose(np.hypot(np.diff(NEW_X),np.diff(NEW_Z)),SPACING,atol=1e-12)

def warp(v):
 v=np.asarray(v,dtype=float).copy();z=v[...,2].copy();v[...,0]+=np.interp(z,np.r_[0,OLD_Z,.95],np.r_[0,DX,DELTA]);v[...,2]+=np.interp(z,np.r_[0,OLD_Z,.95],np.r_[0,NEW_Z-OLD_Z,0]);return v
records=[]
for r in copy.deepcopy(floor_desk.RECORDS):
 n=r['name']
 if n=='landing_plate':r['pos'][0]=.462;r['size'][0]=.052
 if n.startswith(('desk_leg','clamp_')) or n in ['desktop','landing_plate','landing_taper','landing_crossmember'] or n.startswith('landing_support'):
  r['pos'][0]+=DELTA;records.append(r);continue
 if n.startswith('base_'):records.append(r);continue
 if n.startswith('root_'):
  i=int(n.split('_')[1]);old_c=np.asarray(next(t['pos'] for t in floor_desk.RECORDS if t['name']==f'tread_{i:02d}'))
  new_c=old_c+np.array([DX[i],0,NEW_Z[i]-OLD_Z[i]])
  rot=Rotation.from_euler('y',PITCH[i]);r['pos']=(new_c+rot.apply(np.asarray(r['pos'])-old_c)).tolist();r['quat']=[math.cos(PITCH[i]/2),0,math.sin(PITCH[i]/2),0]
  records.append(r);continue

 if r['type']=='mesh':
  points=np.asarray(r['vertices'])+np.asarray(r['pos']);r['vertices']=warp(points).tolist();r['pos']=[0,0,0];records.append(r);continue
 if r['type']=='box':
  R=Rotation.from_quat([*r['quat'][1:],r['quat'][0]]).as_matrix() if r.get('quat') else np.eye(3)
  segs=40 if n in ['rail_-143','rail_143','spine'] else 1
  sx,sy,sz=r['size']
  for k in range(segs):
   zlo=-sz+2*sz*k/segs;zhi=-sz+2*sz*(k+1)/segs
   v=np.array([[x,y,z] for x in [-sx,sx] for y in [-sy,sy] for z in [zlo,zhi]])@R.T+np.asarray(r['pos'])
   records.append({'name':n+(f'_segment{k}' if segs>1 else ''),'type':'mesh','pos':[0,0,0],'vertices':warp(v).tolist()})
 else:records.append(r)
floor_desk.RECORDS[:]=records
original=mdp.reset_floor_desk

def reset_gradual(env,env_ids,**kwargs):
 original(env,env_ids,**kwargs)
 import torch
 ids=env_ids.to(env.device,dtype=torch.long);s=mdp._stair_state(env);orig=env.scene.env_origins[ids]
 for i in range(30):
  shift=torch.tensor([DX[i],0,NEW_Z[i]-OLD_Z[i]],device=env.device)
  s.tread_centre[ids,i]+=shift;s.tread_top[ids,i]+=shift[2]+.018*abs(math.sin(PITCH[i]))+.003*(math.cos(PITCH[i])-1);s.tread_target_xy[ids,i,0]+=shift[0]
  q=torch.zeros(len(ids),4,device=env.device);q[:,0]=math.cos(PITCH[i]/2);q[:,2]=math.sin(PITCH[i]/2)
  env.scene[f'tread_{i:02d}'].write_mocap_pose_to_sim(torch.cat((s.tread_centre[ids,i],q),1),env_ids=ids)
 # Match upper-start robot pose to the moved support station; preserve all
 # pose noise, velocity and previous-action conventions. No support force.
 if __import__('os').environ.get('UPPER_TRANSLATE')=='1':
  start=s.start_tread[ids];assert not s.spawn_on_floor[ids].any()
  shift=torch.stack((torch.as_tensor(DX,device=env.device)[start],torch.zeros(len(ids),device=env.device),torch.as_tensor(NEW_Z-OLD_Z,device=env.device)[start]),1)
  env.sim.data.qpos[ids,:3]+=shift.to(env.sim.data.qpos.dtype)
 # Static records already include the outward desk displacement.
 for i in [30,31]:s.tread_centre[ids,i,0]+=DELTA;s.tread_target_xy[ids,i,0]+=DELTA
mdp.reset_floor_desk=reset_gradual
status=mdp.floor_desk_status

def shifted_status(env):
 s,robot,c,x,y,feet_x,upright,safe=status(env)
 x=x-DELTA;feet_x=feet_x-DELTA
 safe=(c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)&(feet_x>.55).all(dim=1)&(feet_x<.96).all(dim=1)&(y.abs()<.25)&upright
 safe &= robot.data.root_link_lin_vel_w.norm(dim=1)<.25
 safe &= robot.data.root_link_ang_vel_w.norm(dim=1)<2.5
 return s,robot,c,x,y,feet_x,upright,safe
mdp.floor_desk_status=shifted_status

def audit():return {'angles_deg':angles,'equal_station_spacing_m':float(SPACING),'original_station_spacing_m':float(.024/math.sin(math.radians(62))),'actual_station_distances_m':np.hypot(np.diff(NEW_X),np.diff(NEW_Z)).tolist(),'tread_pitch_about_y_deg':np.degrees(PITCH).tolist(),'old_top_z':OLD_Z.tolist(),'new_top_z':NEW_Z.tolist(),'new_station_x':NEW_X.tolist(),'dx':DX.tolist(),'desk_shift_x':DELTA,'tread_dimensions_unchanged':True,'all_station_spacings_equal':True,'fixed_tread_to_rail_mount_angle':True,'records':records}

def configure_cfg(cfg):return cfg

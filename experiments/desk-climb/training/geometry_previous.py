import os,copy,math
import numpy as np
from scipy.spatial import ConvexHull
import geometry_base as base
from mjlab_microduck.robot import floor_desk
from mjlab_microduck.tasks import mdp
ROLE=os.environ['ENDING_ROLE']
assert ROLE in ['control','trim','closer','above']
TRIM=ROLE!='control';XSHIFT=-.06 if ROLE in ['closer','above'] else 0.;ZSHIFT=-.08 if ROLE=='above' else 0.
CUT=float(base.NEW_Z[26]+.012)
CUT_NORMAL=np.array([math.cos(math.radians(40)),0,math.sin(math.radians(40))]);CUT_DOT=float(np.dot(CUT_NORMAL,[.393,0,CUT]))
records=[]
for r in copy.deepcopy(floor_desk.RECORDS):
 n=r['name']
 if TRIM and n.startswith('root_') and int(n.split('_')[1])>=27:continue
 if TRIM and n.startswith(('bridge_','spine_gusset_')) and int(n.split('_')[-1])>=13:continue
 if TRIM and (n.startswith('rail_') or n.startswith('spine_segment')) and r['type']=='mesh':
  v=np.array(r['vertices'])+np.array(r['pos']);inside=(v@CUT_NORMAL)<=CUT_DOT
  if not inside.any():continue
  if not inside.all():
   points=list(v[inside]);h=ConvexHull(v);edges={tuple(sorted((int(a),int(b)))) for tri in h.simplices for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]}
   for a,b in edges:
    if inside[a]!=inside[b]:points.append(v[a]+(v[b]-v[a])*(CUT_DOT-float(v[a]@CUT_NORMAL))/float((v[b]-v[a])@CUT_NORMAL))
   v=np.unique(np.round(points,10),axis=0)
   if len(v)<4:continue
  r['vertices']=v.tolist();r['pos']=[0,0,0]
 if n=='desktop' or n.startswith(('desk_leg','clamp_')):
  r['pos'][0]+=XSHIFT;r['pos'][2]+=ZSHIFT
 records.append(r)
floor_desk.RECORDS[:]=records
original=mdp.reset_floor_desk

def reset(env,env_ids,**kwargs):
 original(env,env_ids,**kwargs)
 import torch
 ids=env_ids.to(env.device,dtype=torch.long);s=mdp._stair_state(env)
 if TRIM:
  for i in [27,28,29,30]:
   s.tread_centre[ids,i,2]=-2;s.tread_top[ids,i]=-2
   if i<30:
    q=torch.zeros(len(ids),4,device=env.device);q[:,0]=1
    env.scene[f'tread_{i:02d}'].write_mocap_pose_to_sim(torch.cat((s.tread_centre[ids,i],q),1),env_ids=ids)
 s.tread_centre[ids,31,0]+=XSHIFT;s.tread_target_xy[ids,31,0]+=XSHIFT;s.tread_centre[ids,31,2]+=ZSHIFT;s.tread_top[ids,31]+=ZSHIFT
mdp.reset_floor_desk=reset
status=mdp.floor_desk_status

def shifted_status(env):
 s,robot,c,x,y,feet_x,upright,safe=status(env);x=x-XSHIFT;feet_x=feet_x-XSHIFT
 safe=(c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)&(feet_x>.55).all(dim=1)&(feet_x<.96).all(dim=1)&(y.abs()<.25)&upright
 safe &= robot.data.root_link_lin_vel_w.norm(dim=1)<.25
 safe &= robot.data.root_link_ang_vel_w.norm(dim=1)<2.5
 return s,robot,c,x,y,feet_x,upright,safe
mdp.floor_desk_status=shifted_status

def configure_cfg(cfg):
 if TRIM:
  for name in ['tread_27','tread_28','tread_29','tread_30']:
   fn=cfg.scene.entities[name].spec_fn
   def disabled(fn=fn):
    spec=fn()
    for g in spec.geoms:g.contype=0;g.conaffinity=0;g.rgba=[0,0,0,0]
    return spec
   cfg.scene.entities[name].spec_fn=disabled
 return cfg

def audit():
 return {**base.audit(),'role':ROLE,'active_treads':27 if TRIM else 30,'lower_ladder_unchanged':True,'desk_extra_x':XSHIFT,'desk_extra_z':ZSHIFT,'desktop_height':.740+ZSHIFT,'last_remaining_station_z':float(base.NEW_Z[26 if TRIM else 29]),'removed_platform':TRIM,'rail_cut_z':CUT if TRIM else None,'records':records}

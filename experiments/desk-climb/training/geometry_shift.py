from geometry_full import *
import geometry_full as full
SHIFT=float(os.environ.get('LADDER_SHIFT','0'))
assert SHIFT in (0.,.02,.04,.06)
for rec in floor_desk.RECORDS:
 if rec['name']=='desktop' or rec['name'].startswith('desk_leg'):continue
 rec['pos'][0]+=SHIFT
original_reset=mdp.reset_floor_desk

def reset_shift(env,env_ids,**kwargs):
 original_reset(env,env_ids,**kwargs)
 import torch
 ids=env_ids.to(env.device,dtype=torch.long);s=mdp._stair_state(env)
 for i in range(30):
  s.tread_centre[ids,i,0]+=SHIFT;s.tread_target_xy[ids,i,0]+=SHIFT
  q=torch.zeros(len(ids),4,device=env.device);q[:,0]=math.cos(base.PITCH[i]/2);q[:,2]=math.sin(base.PITCH[i]/2)
  env.scene[f'tread_{i:02d}'].write_mocap_pose_to_sim(torch.cat((s.tread_centre[ids,i],q),1),env_ids=ids)
 env.sim.data.qpos[ids,0]+=SHIFT
mdp.reset_floor_desk=reset_shift

def audit():
 return {**full.audit(),'records':floor_desk.RECORDS,'rigid_ladder_x_shift_m':SHIFT,'robot_spawn_x_shift_m':SHIFT,'table_fixed':True,'ladder_and_attachment_shapes_unchanged':True}

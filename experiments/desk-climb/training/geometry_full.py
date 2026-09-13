from geometry_connected import *
import geometry_connected as connected
for r in floor_desk.RECORDS:
 if r['name']=='desktop':
  r['pos'][0]=1.110+base.DELTA+XSHIFT;r['size'][0]=.700;r['size'][1]=.400
 if r['name'].startswith('desk_leg_'):
  r['pos'][0]=(.470 if r['pos'][0]<.7+base.DELTA+XSHIFT else 1.750)+base.DELTA+XSHIFT
  r['pos'][1]=.335 if r['pos'][1]>0 else -.335
previous_status=mdp.floor_desk_status

def full_status(env):
 s,robot,c,x,y,feet_x,upright,safe=previous_status(env)
 # Same edge margin, now matched to the full physical tabletop extent.
 safe=(c['foot_tread']==31).all(dim=1)&c['foot_support'].all(dim=1)&(feet_x>.55).all(dim=1)&(feet_x<1.76).all(dim=1)&(y.abs()<.35)&upright
 safe &= robot.data.root_link_lin_vel_w.norm(dim=1)<.25
 safe &= robot.data.root_link_ang_vel_w.norm(dim=1)<2.5
 return s,robot,c,x,y,feet_x,upright,safe
mdp.floor_desk_status=full_status

def audit():
 return {**connected.audit(),'records':floor_desk.RECORDS,'physical_table_dimensions_m':[1.4,.8,.025],'render_collision_match_required':True,'rail_cut_normal_xz':CUT_NORMAL[[0,2]].tolist(),'rail_cut_perpendicular_to_local40deg_direction':True}

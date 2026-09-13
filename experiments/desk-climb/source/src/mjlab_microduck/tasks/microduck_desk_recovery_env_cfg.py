from copy import deepcopy
from mjlab.managers import RewardTermCfg
from mjlab_microduck.tasks.microduck_floor_desk_env_cfg import make_floor_desk,FloorDeskRlCfg
from mjlab_microduck.tasks import mdp

def make_desk_recovery(play=False,floor_probability=.5,desk_probability=.5):
    c=make_floor_desk(play=play,floor_probability=floor_probability)
    t=c.events['reset_stair_ladder'];t.func=mdp.reset_floor_desk_recovery;t.params.update(desk_probability=desk_probability,min_start_tread=24)
    for name in ['bad_orientation','fallen','off_side','foot_fling','overstep','airborne']:
        t=c.terminations[name];t.params={'original':t.func,'original_params':t.params.copy()};t.func=mdp.floor_desk_recoverable_termination
    for name,t in c.rewards.items():
        if t.weight>0 and name!='stance':
            t.params={'original':t.func,'original_params':t.params.copy()};t.func=mdp.floor_desk_climbing_reward
    c.rewards['recovery_progress']=RewardTermCfg(func=mdp.floor_desk_recovery_progress,weight=100.)
    import os
    c.rewards['desk_standing_quality']=RewardTermCfg(func=mdp.floor_desk_standing_quality,weight=float(os.environ.get('DESK_STAND_WEIGHT','0')),params={'lin_vel_std':float(os.environ.get('DESK_STAND_LIN_STD','.2')),'ang_vel_std':float(os.environ.get('DESK_STAND_ANG_STD','2'))})
    if os.environ.get('DESK_DISABLE_STAIR_COSTS','0')=='1':
        for name in ['upward_progress','swing_overshoot']:
            t=c.rewards[name];t.params={'original':t.func,'original_params':t.params.copy()};t.func=mdp.floor_desk_disable_stair_cost
    if not play and os.environ.get('DESK_SHORT_EPISODES','0')=='1':
        from mjlab.managers import TerminationTermCfg
        c.terminations['desk_training_timeout']=TerminationTermCfg(func=mdp.floor_desk_short_episode,params={'seconds':3.},time_out=True)
    c.rewards['desk_leg_pose_cost']=RewardTermCfg(func=mdp.floor_desk_leg_pose_cost,weight=float(os.environ.get('DESK_LEG_POSE_WEIGHT','0')))
    c.rewards['desk_leg_action_cost']=RewardTermCfg(func=mdp.floor_desk_leg_action_cost,weight=float(os.environ.get('DESK_LEG_ACTION_WEIGHT','0')))
    return c
DeskRecoveryRlCfg=deepcopy(FloorDeskRlCfg);DeskRecoveryRlCfg.experiment_name='desk_recovery_blind'

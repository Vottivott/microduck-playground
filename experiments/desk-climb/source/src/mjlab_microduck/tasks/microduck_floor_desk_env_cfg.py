"""Blind 61D floor-to-desk task, rigid central-spine printable geometry."""
from copy import deepcopy
from mjlab.managers import RewardTermCfg,TerminationTermCfg
from mjlab_microduck.robot.floor_desk import GEOMETRY,structure_cfg
from mjlab_microduck.tasks.microduck_ladder_env_cfg import make_microduck_ladder_env_cfg,MicroduckLadderRlCfg
from mjlab_microduck.tasks import mdp

def make_floor_desk(play=False,floor_probability=.25):
    cfg=make_microduck_ladder_env_cfg(play=play,geometry=GEOMETRY,level_table=({'riser':(.024,.024),'angle':(62.,62.)},),episode_length_s=60.,max_start_tread=27)
    cfg.curriculum.clear();cfg.rewards['action_rate_l2'].weight=-.1
    cmd=cfg.commands['twist'];cmd.rel_standing_envs=0.;cmd.ranges.lin_vel_x=(.04,.04);cmd.ranges.lin_vel_y=(0.,0.);cmd.ranges.ang_vel_z=(0.,0.)
    reset=cfg.events['reset_stair_ladder'];reset.func=mdp.reset_floor_desk
    reset.params.update(fixed_level=0,level_mix_prob=0.,floor_spawn_prob=1. if play else floor_probability,swing_spawn_prob=0.)
    cfg.events.pop('push_robot',None)
    for name,width in [('command',3),('head_command',4),('body_command',6)]:
        t=cfg.observations['actor'].terms[name];t.func=mdp.blind_stair_zero_slots;t.params={'width':width};t.noise=None
    # Keep legacy rail entities for the inherited reset, but park them below the floor.
    cfg.scene.entities['rail_printed_structure']=structure_cfg('structure')
    cfg.scene.entities['tread_30']=structure_cfg('landing');cfg.scene.entities['tread_31']=structure_cfg('desk')
    cfg.terminations['reached_top']=TerminationTermCfg(func=mdp.floor_desk_success,time_out=True)
    cfg.terminations['off_side'].func=mdp.floor_desk_off_side
    cfg.rewards['desk_progress']=RewardTermCfg(func=mdp.floor_desk_progress,weight=300.)
    cfg.rewards['stance'].func=mdp.floor_desk_stance
    cfg.sim.nconmax=250
    return cfg
FloorDeskRlCfg=deepcopy(MicroduckLadderRlCfg)
FloorDeskRlCfg.experiment_name='floor_desk_blind';FloorDeskRlCfg.run_name='central_spine'

"""Long straight mini stairs, autonomous 61D proprioceptive actor."""
from copy import deepcopy
from mjlab_microduck.robot.ladder import StairLadderGeometry
from mjlab_microduck.tasks.microduck_ladder_env_cfg import make_microduck_ladder_env_cfg,MicroduckLadderRlCfg
from mjlab_microduck.tasks import mdp
GEOMETRY=StairLadderGeometry(num_treads=64,landing_every=0,rail_length_m=1.8)
LEVELS=({'riser':(.0231,.0231),'angle':(62.,62.)},)
def make_long_stairs(play=False,blind=True):
    cfg=make_microduck_ladder_env_cfg(play=play,geometry=GEOMETRY,level_table=LEVELS,episode_length_s=30.,max_start_tread=48)
    cfg.curriculum.clear()
    cfg.rewards['action_rate_l2'].weight=-.1
    cmd=cfg.commands['twist'];cmd.rel_standing_envs=0.;cmd.ranges.lin_vel_x=(.04,.04);cmd.ranges.lin_vel_y=(0.,0.);cmd.ranges.ang_vel_z=(0.,0.)
    cfg.events['reset_stair_ladder'].params.update(fixed_level=0,level_mix_prob=0.,floor_spawn_prob=1. if play else .25,swing_spawn_prob=0. if play else .3)
    if 'push_robot' in cfg.events:cfg.events.pop('push_robot')
    if blind:
        for name,width in [('command',3),('head_command',4),('body_command',6)]:
            term=cfg.observations['actor'].terms[name];term.func=mdp.blind_stair_zero_slots;term.params={'width':width};term.noise=None
    return cfg
LongStairRlCfg=deepcopy(MicroduckLadderRlCfg)
LongStairRlCfg.experiment_name='long_stairs_blind'
LongStairRlCfg.run_name='blind_source_comparison'

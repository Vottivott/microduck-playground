from pathlib import Path
import runpy
import mjlab_microduck.tasks
from mjlab.tasks.registry import load_env_cfg

def test_only_released_profiles():
    profiles=runpy.run_path(str(Path(__file__).parents[1]/'scripts/parkour_release.py'))['PROFILES']
    assert set(profiles)=={'long-jump','backflip'}
    assert profiles['backflip'][1]['MICRODUCK_FLIP_MAT_VISUAL']=='0'

def test_release_contact_capacity():
    for name in ('PlatformJump','Flip'):
        for play in (False,True):
            assert load_env_cfg(f'Mjlab-{name}-MicroDuck',play=play).sim.nconmax>=200

def test_backflip_starts_on_platform():
    cfg=load_env_cfg('Mjlab-Flip-MicroDuck',play=True)
    assert cfg.events['set_flip_state'].params['midflip_prob']==0
    assert 'flip_spawn_mix' not in cfg.curriculum

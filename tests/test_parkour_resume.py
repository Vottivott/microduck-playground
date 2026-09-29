import json
from pathlib import Path

def test_only_released_continuation_profiles():
    profiles=json.loads((Path(__file__).parents[1]/'experiments/parkour/resume.json').read_text())['profiles']
    assert set(profiles)=={'long-jump','backflip'}
    assert 'MICRODUCK_PJ_GAP' not in profiles['long-jump']['environment']
    assert profiles['backflip']['environment']['MICRODUCK_FLIP_CURRICULUM_SHIFT']=='3000'
    for p in profiles.values():
        assert len(p['checkpoint_sha256'])==64
        assert p['provenance'] and p['limitations']

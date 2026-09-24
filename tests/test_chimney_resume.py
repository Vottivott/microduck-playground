"""Resume recipe and handover-bank CPU regression tests."""
import json
from pathlib import Path
import torch
from mjlab_microduck.tasks import chimney_mdp as ch


def test_handover_cache_is_device_specific(tmp_path):
    path = tmp_path / 'bank.pt'
    torch.save({'qpos': torch.zeros(2, 21), 'qvel': torch.zeros(2, 20),
                'origins': torch.zeros(2, 3)}, path)
    cpu = ch._handover_bank(str(path), 'cpu')
    meta = ch._handover_bank(str(path), 'meta')
    assert cpu['qpos'].device.type == 'cpu'
    assert meta['qpos'].device.type == 'meta'
    assert ch._handover_bank(str(path), 'cpu') is cpu


def test_resume_recipe_is_explicit_and_honest():
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads((root / 'experiments/chimney-climb/resume.json').read_text())
    assert 'Not an exact w17' in recipe['provenance']
    assert recipe['environment']['MICRODUCK_CH_WALL_FRIC'] == '0.5,1.1'
    assert recipe['environment']['MICRODUCK_CH_HANDOVER_PROB'] == '0.40'
    assert recipe['environment']['MICRODUCK_CH_EPISODE_S'] == '40'
    assert len(recipe['bank_sha256']) == len(recipe['checkpoint_sha256']) == 64

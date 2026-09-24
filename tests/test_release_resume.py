"""CPU guards for explicit, non-destructive checkpoint continuation."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resume_release', ROOT/'scripts/resume_release.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_checkpoint_comparison_checks_nested_optimizer_tensors():
    resume.same({'state': [torch.ones(2), 7]}, {'state': [torch.ones(2), 7]})
    with pytest.raises(AssertionError):
        resume.same({'state': [torch.ones(2)]}, {'state': [torch.zeros(2)]})


def test_companions_are_explicit_and_do_not_claim_historical_bank():
    path = ROOT/'experiments/chimney-climb/companions.json'
    enter = resume.read_recipe(path, 'enter')
    exit = resume.read_recipe(path, 'exit')
    assert enter['seed'] == 137
    assert enter['environment']['MICRODUCK_AP_DASH_W'] == '150'
    assert exit['environment']['MICRODUCK_EX_ARRIVAL_PROB'] == '0.0'
    assert exit['environment']['MICRODUCK_EX_ARRIVAL_BANK'] == ''
    assert 'NOT exact x19' in exit['provenance']
    assert exit['environment']['MICRODUCK_EX_PLATFORM_TOP'] == '1.20'


def test_public_recipes_have_no_private_filesystem_dependencies():
    text = (ROOT/'experiments/chimney-climb/companions.json').read_text()
    for prefix in ['/scratch/', '/tmp/', '/net/', '/data/']:
        assert prefix not in text


def test_runner_applies_curriculum_after_load_and_before_learning():
    source = (ROOT/'scripts/resume_release.py').read_text()
    assert source.index('runner.load(') < source.index('raw.reset(seed=seed)') < source.index('runner.learn(')
    assert "runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']" in source
    assert "scripts/export.py" in source


def test_exit_bank_cache_separates_devices(tmp_path):
    from mjlab_microduck.tasks.exit_mdp import _arrival_bank
    path = tmp_path/'arrivals.pt'
    torch.save({'qpos': torch.zeros(2, 21), 'qvel': torch.zeros(2, 20),
                'origins': torch.zeros(2, 3), 'platform_top': 3.0}, path)
    assert _arrival_bank(str(path), 'cpu')['qpos'].device.type == 'cpu'
    assert _arrival_bank(str(path), 'meta')['qpos'].device.type == 'meta'

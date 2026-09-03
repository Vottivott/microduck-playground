import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_running_preview_is_the_released_policy_media() -> None:
    record = _load("experiments/running/eval/released_8749.json")
    media = ROOT / "experiments/running/media/preview.mp4"

    assert record["policy"]["checkpoint_iteration"] == 8749
    assert record["evaluation"]["num_envs"] == 256
    assert record["evaluation"]["survival_fraction"] == 0.9921875
    assert hashlib.sha256(media.read_bytes()).hexdigest() == record["media"]["sha256"]


def test_every_released_stilt_height_has_a_rollout_record() -> None:
    record = _load("experiments/stilts/eval/released_rollouts.json")
    rollouts = record["rollouts"]

    assert [item["height_cm"] for item in rollouts] == [10, 15, 20, 25, 50, 100, 140, 200]
    assert all(item["full_horizon"] for item in rollouts)
    assert all(not item["reset_or_termination"] for item in rollouts)
    assert all(not item["auxiliary_ground_contact"] for item in rollouts)
    for item in rollouts:
        assert item["root_z_range_m"][0] < item["root_z_range_m"][1]
        assert len(item["policy_onnx_sha256"]) == 64
        assert len(item["checkpoint_pt_sha256"]) == 64
        assert len(item["source_record_sha256"]) == 64


def test_pollen_runtime_swing_plane_adapter_compiles_and_passes(tmp_path: Path) -> None:
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc is unavailable in this environment")

    test_binary = tmp_path / "swing-plane-cue-test"
    subprocess.run(
        [
            rustc,
            "--edition=2024",
            "--test",
            str(ROOT / "integrations/pollen-microduck/swing_plane_cue.rs"),
            "-o",
            str(test_binary),
        ],
        check=True,
    )
    subprocess.run([test_binary], check=True)

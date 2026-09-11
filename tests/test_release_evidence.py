import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_running_preview_is_the_default_released_policy_media() -> None:
    record = _load("experiments/running/eval/released_12195.json")
    media = ROOT / "experiments/running/media/preview.mp4"

    assert record["policy"]["checkpoint_iteration"] == 12195
    assert record["evaluation"]["cases"][0]["num_envs"] == 512
    assert record["evaluation"]["cases"][0]["survival_fraction"] == 0.9921875
    assert hashlib.sha256(media.read_bytes()).hexdigest() == record["media"]["sha256"]


def test_running_release_contains_only_selected_policy_evidence() -> None:
    default = _load("experiments/running/eval/released_12195.json")

    assert default["policy"]["resumed_from_iteration"] == 11748
    assert default["checkpoint_onnx_parity"]["action_clipping_during_training"] is False
    assert not (ROOT / "experiments/running/eval/released_8749.json").exists()
    assert not (ROOT / "scripts/render_checkpoint.py").exists()


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


def test_stilt_preview_is_the_released_10cm_policy() -> None:
    media = ROOT / "experiments/stilts/media/preview.mp4"

    assert hashlib.sha256(media.read_bytes()).hexdigest() == (
        "589b67eb3bd7102a29b2f26b6b99897e7e0b16717196a76a9a3325cd18d8fd8e"
    )


def test_every_released_stilt_height_has_left_right_and_pair_meshes() -> None:
    release_dir = ROOT / "hardware/stilts/generated/release"

    heights = ("10p0", "15p0", "20p0", "25p0", "50p0", "100p0", "140p0", "200p0")
    for height in heights:
        for variant in ("left", "right", "pair"):
            mesh = release_dir / f"direct_replacement_{variant}_b0p50_h{height}cm.stl"
            assert mesh.is_file(), mesh.relative_to(ROOT)
            assert mesh.stat().st_size > 84, mesh.relative_to(ROOT)


def test_swing_release_stops_at_the_policy_shown_in_its_video() -> None:
    summary = _load("experiments/swing/eval/summary.json")
    seed27 = _load("experiments/swing/eval/seed27_full.json")
    checkpoint_dir = ROOT / "experiments/swing/checkpoints"
    checkpoint = checkpoint_dir / "alpha050.pt"
    video = ROOT / summary["video"]["path"]

    assert summary["policy"] == "alpha050"
    assert seed27["checkpoint"] == "experiments/swing/checkpoints/alpha050.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    assert hashlib.sha256(video.read_bytes()).hexdigest() == summary["video"]["sha256"]
    assert {path.name for path in checkpoint_dir.glob("*.pt")} == {
        "alpha050.pt",
        "frontier_source_3500.pt",
        "planar_target_3600.pt",
    }


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

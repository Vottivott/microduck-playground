"""Public command-line discovery must work for task-specific configurations."""

from __future__ import annotations

import subprocess
import sys


def test_swing_train_help_renders() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mjlab_microduck.train_cli", "Mjlab-SwingPump-MicroDuck", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--agent.max-iterations" in result.stdout
    assert "--agent.algorithm.symmetry-cfg.mirror-loss-coeff" in result.stdout

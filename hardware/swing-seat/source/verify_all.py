"""Generate every seat variant and run the complete clearance verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGES = (
    "generate_seat.py",
    "generate_padded_seat.py",
    "generate_retained_seat.py",
    "verify_retention_clearance.py",
    "verify_pumping_clearance.py",
)


def main() -> None:
    for stage in STAGES:
        print(f"\n==> {stage}", flush=True)
        subprocess.run([sys.executable, stage], cwd=HERE, check=True)

    print("\nAll swing-seat generation and clearance checks passed.")


if __name__ == "__main__":
    main()

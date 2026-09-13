#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <microduck-running-speed-final-model-8749.pt> [output-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT="$(realpath "$1")"
OUTPUT_DIR="${2:-${REPO_ROOT}/artifacts/recreated-running-crash-mat}"
ASSET_DIR="${REPO_ROOT}/docs/assets/running_crash_mat"
UV_BIN="${UV_BIN:-uv}"

mkdir -p "${OUTPUT_DIR}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MICRODUCK_RUNNING_TARGET_MAX_SPEED=1.65
export MICRODUCK_RUNNING_SPEED_CAP=1.8
export MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=-0.10
export MICRODUCK_RUNNING_ENABLE_SYMMETRY=0
export MICRODUCK_RUNNING_ENABLE_HEADING_FEEDBACK=0

cd "${REPO_ROOT}"
"${UV_BIN}" run \
  --with trimesh \
  --with scipy \
  --with fast-simplification \
  --with pillow \
  python scripts/render_running_crash_video.py \
  --checkpoint "${CHECKPOINT}" \
  --glb "${ASSET_DIR}/crash-mat-textureless.glb" \
  --output-dir "${OUTPUT_DIR}" \
  --speed 1.8 \
  --barrier-x 6.30 \
  --video-steps 400 \
  --startup-seed 0 \
  --seed 17 \
  --camera-final-distance 3.5 \
  --camera-start-fovy 22 \
  --camera-end-fovy 14 \
  --camera-zoom-start-s 0.15 \
  --camera-zoom-end-s 0.75 \
  --camera-pan-smoothing 0.12 \
  --camera-final-azimuth 35 \
  --camera-final-elevation -10 \
  --camera-lookahead 0.20 \
  --camera-impact-hold-s 0.60 \
  --power-off-s 5.16 \
  --spark-particles 72 \
  --spark-seed 20260829

"${UV_BIN}" run \
  --with imageio \
  --with imageio-ffmpeg \
  --with pillow \
  python scripts/add_running_crash_label.py \
  --input "${OUTPUT_DIR}/microduck_crash_mat_upright_telephoto.mp4" \
  --output "${OUTPUT_DIR}/microduck-running-crash-mat-1.6mps-silent.mp4" \
  --font "${ASSET_DIR}/Anton-Regular.ttf" \
  --x 30 \
  --y 26 \
  --primary-size 80 \
  --secondary-size 23

sha256sum "${OUTPUT_DIR}/microduck-running-crash-mat-1.6mps-silent.mp4"

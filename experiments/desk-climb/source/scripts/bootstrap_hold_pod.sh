#!/usr/bin/env bash
# Generic held-pod bootstrap: extract the source archive, install uv, sync the
# environment, run the CPU invariant tests, then idle so the orchestrator can
# start experiments with `kubectl exec`.  Nothing here is task specific; the
# run tag and persist root come from the environment.
set -euo pipefail

RUN_TAG="${RUN_TAG:-run}"
PERSIST_ROOT="${PERSIST_ROOT:-/net/gcp/cloud-storage/europe-west/training/hannes/genai/exp-hold/${RUN_TAG}}"
WORK_ROOT="/scratch/${RUN_TAG}"
REPO_ROOT="${WORK_ROOT}/repo"
TEST_FILES="${BOOTSTRAP_TEST_FILES:-tests/test_ladder_cfg.py}"

mkdir -p "${PERSIST_ROOT}/logs" "${PERSIST_ROOT}/eval" "${PERSIST_ROOT}/source" \
  "${WORK_ROOT}" /scratch/control
exec > >(tee -a "${PERSIST_ROOT}/logs/bootstrap.log") 2>&1
echo "BOOTSTRAP_START $(date -Iseconds)"
echo "EXP_PERSIST_PATH ${PERSIST_ROOT}"
nvidia-smi -L

test -s /scratch/input/source.tar.gz
cp /scratch/input/source.tar.gz "${PERSIST_ROOT}/source/source.tar.gz"
sha256sum /scratch/input/source.tar.gz | tee "${PERSIST_ROOT}/source/SHA256SUMS"
mkdir -p "${REPO_ROOT}"
tar -xzf /scratch/input/source.tar.gz -C "${REPO_ROOT}"
cd "${REPO_ROOT}"

mkdir -p /scratch/tools
curl -LsSf https://astral.sh/uv/0.12.7/install.sh | env UV_INSTALL_DIR=/scratch/tools sh
export PATH="/scratch/tools:${PATH}"
export UV_LINK_MODE=copy
export WANDB_MODE="${WANDB_MODE:-disabled}"
# Do not export MUJOCO_GL=egl here: the worker image has no loadable libEGL,
# and with that variable set `import mujoco` itself fails.  Rendering jobs
# install libegl1 and set it explicitly.
unset MUJOCO_GL
uv sync --no-progress
uv run --with pytest python -m pytest ${TEST_FILES} -q

touch /scratch/control/READY
echo "BOOTSTRAP_READY $(date -Iseconds)"
# Experiments are started explicitly by the orchestrator; hold the pod until
# the Job's active deadline or deletion.
while true; do
  sleep 30
done

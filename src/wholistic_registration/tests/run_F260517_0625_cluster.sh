#!/bin/bash
# Job payload for the F260517 0625 pipeline reproduction on a Janelia GPU node.
# Registers N_FRAMES_LIMIT forward-loop frames after the fixed 5-frame warmup.
# Follows the conventions of run_F260517_cluster.sh.
set -euo pipefail

unset XDG_RUNTIME_DIR || true

source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh
conda activate wholistic-registration

# Resolve the repo root from this script's own location, so the payload runs
against whichever checkout it was copied into.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"

# LSF exposes one GPU per job as device 0, so the script's default of device 1
# would raise on the allocated node.
export GPU_DEVICE="${GPU_DEVICE:-0}"
export N_FRAMES_LIMIT="${N_FRAMES_LIMIT:-3}"
export F260517_OUT_DIR="${F260517_OUT_DIR:-/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/repro_3frame_out}"

echo "host: $(hostname)"
echo "date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE=$GPU_DEVICE  N_FRAMES_LIMIT=$N_FRAMES_LIMIT"
echo "F260517_OUT_DIR=$F260517_OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "--- /nrs visibility from this compute node ---"
ls -la /nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/
echo "---------------------------------------------"

python -u src/wholistic_registration/tests/run_F260517_0625.py

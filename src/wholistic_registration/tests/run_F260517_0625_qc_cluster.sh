#!/bin/bash
# Job payload for the instrumented QC variant of the F260517 0625 pipeline.
# Registers N_FRAMES_LIMIT forward-loop frames after the fixed 5-frame warmup,
# and saves masks / motion field / alignment-QC diagnostics that
# run_F260517_0625.py does not persist. Follows the conventions of
# run_F260517_0625_cluster.sh (the non-instrumented sibling run).
set -euo pipefail

unset XDG_RUNTIME_DIR || true

source /groups/ahrens/home/ruttenv/miniforge3/etc/profile.d/conda.sh
conda activate wholistic-registration

# Resolve the repo root from this script's own location, so the payload runs
# against whichever checkout it was copied into.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"

# LSF exposes one GPU per job as device 0, so the script's default of device 1
# would raise on the allocated node.
export GPU_DEVICE="${GPU_DEVICE:-0}"
# 5 forward-loop frames after the fixed 0-4 warmup, per the requested smoke test.
export N_FRAMES_LIMIT="${N_FRAMES_LIMIT:-5}"
export F260517_OUT_DIR="${F260517_OUT_DIR:-/nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/registration_out/f260517_0625_qc}"

# Diagnostic saves: all on, phase/motion field saved every frame (small run).
export SAVE_ALIGNMENT_QC="${SAVE_ALIGNMENT_QC:-1}"
export SAVE_MASKS="${SAVE_MASKS:-1}"
export SAVE_MOTION_FIELD="${SAVE_MOTION_FIELD:-1}"
export PHASE_SAVE_STRIDE="${PHASE_SAVE_STRIDE:-1}"
export SAVE_PROJECTION_STATE="${SAVE_PROJECTION_STATE:-1}"

echo "host: $(hostname)"
echo "date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE=$GPU_DEVICE  N_FRAMES_LIMIT=$N_FRAMES_LIMIT"
echo "F260517_OUT_DIR=$F260517_OUT_DIR"
echo "SAVE_ALIGNMENT_QC=$SAVE_ALIGNMENT_QC  SAVE_MASKS=$SAVE_MASKS  SAVE_MOTION_FIELD=$SAVE_MOTION_FIELD  PHASE_SAVE_STRIDE=$PHASE_SAVE_STRIDE  SAVE_PROJECTION_STATE=$SAVE_PROJECTION_STATE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "--- /nrs visibility from this compute node ---"
ls -la /nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/
echo "---------------------------------------------"

python -u src/wholistic_registration/tests/run_F260517_0625_qc.py

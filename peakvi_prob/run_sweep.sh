#!/usr/bin/env bash
# =============================================================================
# PeakVI Probe - wandb sweep launcher
#
# Creates one sweep, then launches parallel agents across GPUs.
# Each agent is pinned to one GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash run_sweep.sh
# or in background:
#   nohup bash run_sweep.sh > run_sweep.log 2>&1 &
#
# Optional overrides:
#   NUM_GPUS=4 RUNS_PER_AGENT=2 bash run_sweep.sh
# =============================================================================
set -euo pipefail

WANDB="/root/project/chaohan/.conda/bin/wandb"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

echo "Creating sweep from sweep.yaml ..."
SWEEP_OUT=$("${WANDB}" sweep sweep.yaml 2>&1)
echo "${SWEEP_OUT}"

SWEEP_ID=$(echo "${SWEEP_OUT}" | grep "wandb agent" | awk '{print $NF}')

if [[ -z "${SWEEP_ID}" ]]; then
    echo "ERROR: could not parse sweep ID from wandb output." >&2
    exit 1
fi

echo ""
echo "Sweep ID : ${SWEEP_ID}"
echo "Starting parallel agents ..."
echo ""

RUNS_PER_AGENT="${RUNS_PER_AGENT:-1}"
NUM_GPUS="${NUM_GPUS:-8}"

echo "Launching ${NUM_GPUS} agents, one per GPU, ${RUNS_PER_AGENT} runs per agent ..."

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${SCRIPT_DIR}/sweep_agent_gpu${GPU_ID}.log"
    echo "  GPU ${GPU_ID} -> ${LOG_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${WANDB}" agent --count "${RUNS_PER_AGENT}" "${SWEEP_ID}" > "${LOG_FILE}" 2>&1 &
done

wait
echo "All sweep agents finished."

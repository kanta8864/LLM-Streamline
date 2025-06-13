#!/bin/bash
# File: run_job_smoke_test.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=smoke-test
#SBATCH --partition=general
#SBATCH --time=00:10:00              # Request only 10 minutes for the test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=smoke_test_output_%j.log
#SBATCH --gres=gpu:l40:1

set -e

# --- Load Required Modules ---
echo "Loading modules..."
module use /opt/insy/modulefiles
module load cuda/12.1

# --- Job Execution & Path Definitions ---
echo "Job started on $(hostname)"
CONTAINER_PATH="$HOME/my_deepspeed_env_v2.sif"
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container file not found at ${CONTAINER_PATH}"
    exit 1
fi

# --- Set Caches to the Node's Local, Fast /tmp Directory ---
JOB_TMP_DIR="/tmp/${USER}/${SLURM_JOB_ID}"
mkdir -p "${JOB_TMP_DIR}"
export HF_HOME="${JOB_TMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${JOB_TMP_DIR}/datasets"
export TRANSFORMERS_CACHE="${JOB_TMP_DIR}/transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
echo "Using temporary cache for this job: ${JOB_TMP_DIR}"

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up temporary directory: ${JOB_TMP_DIR}"
    rm -rf "${JOB_TMP_DIR}"
}
trap cleanup EXIT

# --- Run Code Inside Apptainer ---
echo "Starting container execution for smoke test..."
apptainer exec \
    --nv \
    --bind "$(pwd)":/app \
    --bind "${JOB_TMP_DIR}":"${JOB_TMP_DIR}" \
    --pwd /app \
    --env "HF_HOME=${HF_HOME}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    "$CONTAINER_PATH" \
    env -u CUDA_HOME python3 smoke_test.py # Changed to run the smoke test

echo "Job finished."
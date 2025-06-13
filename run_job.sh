#!/bin/bash
# File: run_job_test.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=real-run
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_test_output_%j.log
#SBATCH --gres=gpu:l40:1

set -e

# --- Load Required Modules ---
echo "Loading modules..."
module use /opt/insy/modulefiles
module load cuda/12.1


# --- Job Execution & Path Definitions ---
echo "Job started on $(hostname)"
CONTAINER_PATH="$HOME/my_deepspeed_env.sif"
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container file not found at ${CONTAINER_PATH}"
    exit 1
fi

# --- Set Caches to the Node's Local, Fast /tmp Directory ---
JOB_TMP_DIR="/tmp/${USER}/${SLURM_JOB_ID}"
mkdir -p "${JOB_TMP_DIR}"

# Set the cache directories
export HF_HOME="${JOB_TMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${JOB_TMP_DIR}/datasets"
export TRANSFORMERS_CACHE="${JOB_TMP_DIR}/transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
echo "Using temporary cache for this job: ${JOB_TMP_DIR}"

# --- Best Practice: Cleanup Function ---
cleanup() {
    echo "Cleaning up temporary directory: ${JOB_TMP_DIR}"
    rm -rf "${JOB_TMP_DIR}"
}
trap cleanup EXIT

# --- Run Code Inside Apptainer ---
echo "Starting container execution..."
apptainer exec \
    --nv \
    --bind "$(pwd)":/app \
    --bind "${JOB_TMP_DIR}":"${JOB_TMP_DIR}" \
    --bind /tudelft.net/staff-umbrella:/tudelft.net/staff-umbrella \
    --pwd /app \
    --env "HF_HOME=${HF_HOME}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    "$CONTAINER_PATH" \
    bash -c "CUDA_HOME= accelerate launch mseloss_entry.py --with_tracking"

echo "Job finished."
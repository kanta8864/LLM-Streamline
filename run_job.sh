#!/bin/bash
# File: run_job_test.sh

# ... (all SBATCH directives and module loads are the same) ...
#SBATCH --job-name=testtest-run
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_test_output_%j.log
#SBATCH --gres=gpu:l40:1

# --- Load Required Modules ---
echo "Loading modules..."
module use /opt/insy/modulefiles
module load cuda/12.1

# --- Job Execution & Path Definitions ---
echo "Job started on $(hostname)"
CONTAINER_PATH="/tudelft.net/staff-umbrella/llmstreamline88/containers/my_env_v4.sif"

# --- Set Caches and Local Output Directory ---
JOB_TMP_DIR="/tmp/${USER}/${SLURM_JOB_ID}"
LOCAL_OUTPUT_DIR="${JOB_TMP_DIR}/final_output" # This is the single source of truth
mkdir -p "${LOCAL_OUTPUT_DIR}"

export HF_HOME="${JOB_TMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${JOB_TMP_DIR}/datasets"
export TRANSFORMERS_CACHE="${JOB_TMP_DIR}/transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
echo "Using temporary job directory: ${JOB_TMP_DIR}"
echo "Local output will be saved to: ${LOCAL_OUTPUT_DIR}"

# --- Best Practice: Cleanup Function ---
# (See improvement note below)
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
    --env-file <(env | grep -E '^(HF_|TRANSFORMERS_|SLURM_JOB_ID)') \
    "$CONTAINER_PATH" \
    accelerate launch mseloss_entry.py \
        --with_tracking \
        --output-dir "${LOCAL_OUTPUT_DIR}" # Pass the correct path here

echo "Container execution finished."

# --- Copy results from local temp to network storage ---
echo "Copying results..."
FINAL_DESTINATION="/tudelft.net/staff-umbrella/llmstreamline88/"

if [ -d "${LOCAL_OUTPUT_DIR}" ] && [ "$(ls -A ${LOCAL_OUTPUT_DIR})" ]; then
    # Use rsync for better progress and robustness
    rsync -av "${LOCAL_OUTPUT_DIR}/" "${FINAL_DESTINATION}/"
    echo "✅ Results successfully copied to ${FINAL_DESTINATION}"
else
    echo "⚠️ WARNING: Local output directory '${LOCAL_OUTPUT_DIR}' is empty or not found. Nothing to copy."
fi

echo "Job finished."
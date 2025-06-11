#!/bin/bash

#SBATCH --job-name="apptainer-deepspeed"
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=15
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-msc-cs

# --- Paths and Environment ---
PROJECT_DIR="/scratch/ktanahashi/LLM-Streamline"
CONTAINER_PATH="${PROJECT_DIR}/my_deepspeed_env.sif"

# Set Hugging Face cache
export HF_HOME="/scratch/ktanahashi/huggingface_cache"
# Only set offline mode if you're sure all data is cached
# export HF_HUB_OFFLINE=1

# --- Load Modules ---
echo "Loading modules..."
module load 2024r1
module load cuda/12.1
module load apptainer

echo "CUDA_HOME on host: ${CUDA_HOME}"


# --- Job Execution ---
echo "Job started on $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Container path: ${CONTAINER_PATH}"

# Navigate to project directory
cd "${PROJECT_DIR}" || { echo "ERROR: Failed to cd to ${PROJECT_DIR}."; exit 1; }

# Check if container exists
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container file not found at ${CONTAINER_PATH}"
    exit 1
fi

echo "Container found. Starting execution..."

# --- Run Code Inside the Apptainer Container ---
# The key is to prepend the host's CUDA library paths to the container's LD_LIBRARY_PATH
# Apptainer's --nv flag automatically binds the necessary NVIDIA drivers and CUDA libraries.
# By setting APPTAINERENV_LD_LIBRARY_PATH, we ensure the container's dynamic loader
# can find these host-provided libraries.

apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}":"/app" \
    --bind /scratch \
    --pwd /app \
    --env "APPTAINERENV_LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    "${CONTAINER_PATH}" \
    /opt/venv/bin/accelerate launch mseloss_entry.py --with_tracking

echo "Job finished."
#!/bin/bash

# --- Slurm Resource Request (from your script for DelftBlue) ---
#SBATCH --job-name="apptainer-deepspeed"
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-msc-cs

# --- Paths and Environment ---
PROJECT_DIR="/scratch/ktanahashi/LLM-Streamline"
CONTAINER_PATH="${PROJECT_DIR}/my_deepspeed_env.sif" # Assuming container is in the project dir on scratch

# Set Hugging Face cache to offline mode, pointing to your pre-downloaded files
export HF_HOME="/scratch/ktanahashi/huggingface_cache"
export HF_HUB_OFFLINE=1

# --- Load Modules ---
echo "Loading modules..."
# Load the general environment module for DelftBlue
module load 2024r1
# Load CUDA for the host to correctly interface with the GPU
module load cuda/12.1
# Load the Apptainer module itself
module load apptainer

# --- Job Execution ---
echo "Job started on $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Container path: ${CONTAINER_PATH}"
echo "Hugging Face cache (HF_HOME): ${HF_HOME}"
echo "Running in OFFLINE mode: ${HF_HUB_OFFLINE}"

# Navigate to project directory
cd "${PROJECT_DIR}" || { echo "ERROR: Failed to cd to ${PROJECT_DIR}."; exit 1; }

# --- Run Code Inside the Apptainer Container ---
# This command executes the training script inside the container's environment.
# Note that we are no longer sourcing a venv.
apptainer exec \
    --nv \
    --bind "${PROJECT_DIR}":"/app" \
    --bind /scratch \
    --pwd /app \
    "${CONTAINER_PATH}" \
    accelerate launch mseloss_entry.py --with_tracking

echo "Job finished."
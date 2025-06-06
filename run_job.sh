#!/bin/bash
# File: run_job_test.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=deepspeed-test-run
#SBATCH --partition=general
#SBATCH --time=03:00:00              # Request 1 hour for a test run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=128G                    # Request 64 GB of memory
#SBATCH --output=slurm_test_output_%j.log

# --- GPU Request ---
# This line actively requests one NVIDIA L40 GPU.
#SBATCH --gres=gpu:l40:1

# --- Job Execution ---
echo "Job started on $(hostname)"
echo "Running in directory: $(pwd)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# --- Define Paths ---
# Define path to your container image (assuming it's in your home directory)
CONTAINER_PATH="$HOME/my_deepspeed_env.sif"

# --- NEW: Set Caches to the Node's Temporary /tmp Directory ---
# This creates a unique temp dir for your job to avoid conflicts
JOB_TMP_DIR="/tmp/${SLURM_JOB_ID}"
export HF_HOME="${JOB_TMP_DIR}/huggingface_cache"
mkdir -p "${HF_HOME}"
echo "Using temporary cache for this job: ${HF_HOME}"

# --- Run Code Inside Apptainer ---
# Note the `--bind /tmp` to make the cache visible inside the container
apptainer exec \
    --nv \
    --bind $(pwd) \
    --bind /tmp \
    "$CONTAINER_PATH" \
    accelerate launch mseloss_entry.py --with_tracking

echo "Job finished. Temporary data in ${JOB_TMP_DIR} will be deleted."
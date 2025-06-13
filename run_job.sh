#!/bin/bash
# File: run_job_test.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=deepspeed-test-run
#SBATCH --partition=general
#SBATCH --time=03:00:00              # Request 3 hours for the run
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=512G                   # Request 256 GB of memory
#SBATCH --output=slurm_test_output_%j.log

# --- GPU Request ---
# This line actively requests one NVIDIA L40 GPU.
#SBATCH --gres=gpu:l40:1

# --- Load Required Modules ---
# FIX: Load the base module environment and a specific version of CUDA.
# The error "module(s) are unknown: 'cuda'" means we must specify a version.
echo "Loading modules..."
module use /opt/insy/modulefiles
module load cuda/12.1
module load apptainer

# --- Job Execution ---
echo "Job started on $(hostname)"
echo "Running in directory: $(pwd)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Host CUDA_HOME is: ${CUDA_HOME}"

# --- Define Paths ---
# Define path to your container image (assuming it's in your home directory)
CONTAINER_PATH="$HOME/my_deepspeed_env.sif"

# Check if container exists
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container file not found at ${CONTAINER_PATH}"
    exit 1
fi

# --- Set Caches to the Node's Temporary /tmp Directory ---
# This creates a unique temp dir for your job to avoid conflicts
JOB_TMP_DIR="/tmp/${SLURM_JOB_ID}"
export HF_HOME="${JOB_TMP_DIR}/huggingface_cache"
mkdir -p "${HF_HOME}"
echo "Using temporary cache for this job: ${HF_HOME}"

# --- Run Code Inside Apptainer ---
# This command should now work because `module load cuda/12.1` will
# correctly set the ${CUDA_HOME} variable needed for the LD_LIBRARY_PATH.
echo "Starting container execution..."
apptainer exec \
    --nv \
    --bind "$(pwd)":/app \
    --bind /tmp \
    --pwd /app \
    --env "APPTAINERENV_LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}" \
    "$CONTAINER_PATH" \
    accelerate launch mseloss_entry.py --with_tracking

echo "Job finished."

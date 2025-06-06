#!/bin/bash
# File: run_job.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=deepspeed-training
#SBATCH --partition=general
#SBATCH --time=03:00:00              # Request 5 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=64G                    # Request 64 GB of memory
#SBATCH --output=slurm_output_%j.log

# --- GPU Request ---
# This line actively requests one NVIDIA A100 80GB GPU.
#SBATCH --gres=gpu:l40:1

# --- Job Execution ---
echo "Job started on $(hostname)"
echo "Running in directory: $(pwd)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"

# Define path to your container image (assuming it's in your home directory)
CONTAINER_PATH="$HOME/my_deepspeed_env.sif"

# This command runs your code. Let's break it down:
# `apptainer exec`: Execute a command in the container.
# `--nv`:           Enable NVIDIA GPU access.
# `--bind $(pwd)`:  Mount the current directory (your repo) into the container.
#                   It will appear at the same path inside.
# `$CONTAINER_PATH`: The environment to run in.
# `accelerate launch ...`: Your actual command.
apptainer exec \
    --nv \
    --bind $(pwd) \
    "$CONTAINER_PATH" \
    accelerate launch mseloss_entry.py --with_tracking

echo "Job finished."
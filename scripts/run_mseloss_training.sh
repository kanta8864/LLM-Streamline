#!/bin/bash

#SBATCH --job-name="without_specific_python_module" # Changed job name for clarity
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-msc-cs
#SBATCH --output=without_specific_python_module_%j.out # Optional: specific output file
#SBATCH --error=without_specific_python_module_%j.err  # Optional: specific error file

PROJECT_DIR="/scratch/ktanahashi/LLM-Streamline"
VENV_PATH="${PROJECT_DIR}/venv"

# Load necessary modules
echo "Loading modules..."
module load 2024r1 || { echo "ERROR: Failed to load module 2024r1. Exiting."; exit 1; }
echo "Module 2024r1 loaded."

module load cuda/12.1 || { echo "ERROR: Failed to load module cuda/12.1. Exiting."; exit 1; }
echo "Module cuda/12.1 loaded."

# Section for loading python/3.10.12 is now commented out for this test
# echo "Attempting to load python/3.10.12..."
# module load python/3.10.12
# if [ $? -ne 0 ]; then
#     echo "ERROR: Failed to load module python/3.10.12."
#     echo "Listing available python modules for debugging (if any):"
#     module avail python # This will list modules containing "python"
#     echo "Please check the output above and with your HPC support."
#     exit 1
# fi
# echo "Module python/3.10.12 loaded."
echo "INFO: Skipping explicit load of system python/3.10.12 module for this test."

# Set Hugging Face cache
export HF_HOME="/scratch/ktanahashi/huggingface_cache"
export HF_HUB_OFFLINE=1

# Activate your virtual environment
echo "Activating virtual environment from: ${VENV_PATH}/bin/activate"
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "ERROR: Virtual environment activation script not found at ${VENV_PATH}/bin/activate"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

echo "Verifying Python version from virtual environment..."
which python3
python3 --version
echo "PYTHONPATH: $PYTHONPATH"
echo "PATH: $PATH"

cd "${PROJECT_DIR}" || { echo "ERROR: Failed to cd to ${PROJECT_DIR}. Exiting."; exit 1; } # Added check for cd


# Start training
TRAINING_SCRIPT="mseloss_entry.py"
echo "Starting training script: ${TRAINING_SCRIPT} in $(pwd)" # Added current directory
if [ ! -f "${TRAINING_SCRIPT}" ]; then
    echo "ERROR: Training script ${TRAINING_SCRIPT} not found in $(pwd)"
    exit 1
fi
srun --unbuffered python3 "${TRAINING_SCRIPT}"

echo "Job finished."
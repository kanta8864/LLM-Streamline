#!/bin/bash

#SBATCH --job-name="zeri" # Updated job name

#SBATCH --time=2:00:00 # Reduced time as no pip install in job

#SBATCH --ntasks=1

#SBATCH --partition=gpu

#SBATCH --gpus-per-task=1

#SBATCH --cpus-per-task=6

#SBATCH --mem-per-cpu=5000M

#SBATCH --account=education-eemcs-msc-cs

PROJECT_DIR="/scratch/ktanahashi/LLM-Streamline" # Your project directory
VENV_NAME="venv" # Using fixed venv name like friend's example

# No trap command for cleanup, to match friend's script more closely.
# If this venv needs to be fresh each time, manual removal or a job-specific name would be needed.

# Change to project directory first
cd "${PROJECT_DIR}" || { echo "ERROR: Failed to cd to ${PROJECT_DIR}. Exiting."; exit 1; }
echo "Changed directory to $(pwd)"

# Load necessary modules (following friend's example)
echo "Loading modules..."
module load 2023r1 || { echo "ERROR: Failed to load module 2023r1. Exiting."; exit 1; }
echo "Module 2023r1 loaded."

module load cuda/12.1 || { echo "ERROR: Failed to load module cuda/12.1. Exiting."; exit 1; }
echo "Module cuda/12.1 loaded."

module load python/3.8.12 || { echo "ERROR: Failed to load module python/3.8.12. Exiting."; exit 1; }
echo "Module python/3.8.12 loaded."

echo "System Python details after all module loads:"
which python3
python3 --version

# Create a new virtual environment in the current directory (PROJECT_DIR)
echo ""
echo "Creating new virtual environment at: ./${VENV_NAME}"
python3 -m venv "./${VENV_NAME}" || { echo "ERROR: Failed to create venv. Exiting."; exit 1; }

# Upgrade pip within the venv (using relative path after cd)
echo ""
echo "Upgrading pip in venv..."
"./${VENV_NAME}/bin/python" -m pip install --upgrade pip || { echo "ERROR: Failed to upgrade pip. Exiting."; exit 1; }

# Install dependencies from requirements.txt into the venv (using relative path)
# This assumes requirements.txt is in PROJECT_DIR (current directory)
echo ""
echo "Installing dependencies from requirements.txt into venv..."
if [ ! -f "requirements.txt" ]; then # Check in current directory
    echo "ERROR: requirements.txt not found in $(pwd)"
    exit 1
fi
"./${VENV_NAME}/bin/pip" install -r "requirements.txt" || { echo "ERROR: Failed to install requirements.txt. Exiting."; exit 1; }

# Install ultralytics into the venv (using relative path)
echo ""
echo "Installing ultralytics into venv..."
"./${VENV_NAME}/bin/pip" install ultralytics || { echo "ERROR: Failed to install ultralytics. Exiting."; exit 1; }

echo ""
echo "Verifying Python version from virtual environment..."
"./${VENV_NAME}/bin/python" --version

# Set Hugging Face cache (if your script uses it)
# export HF_HOME="/scratch/ktanahashi/huggingface_cache"
# export HF_HUB_OFFLINE=1 # Friend's script doesn't show this, assuming network for pip

# Already in PROJECT_DIR

# Start training
TRAINING_SCRIPT="mseloss_entry.py" # Make sure this is your correct script name
echo ""
echo "Starting training script: ${TRAINING_SCRIPT} in $(pwd)"
if [ ! -f "${TRAINING_SCRIPT}" ]; then
    echo "ERROR: Training script ${TRAINING_SCRIPT} not found in $(pwd)"
    exit 1
fi
# Run the python script using the venv's python interpreter (relative path)
# Friend's script redirects to job.log: > job.log
# Keeping SLURM output files for now. If you want to redirect like your friend,
# you can add "> job.log" to the end of the srun command below.
srun --unbuffered "./${VENV_NAME}/bin/python" "${TRAINING_SCRIPT}"

echo ""
echo "Job finished."

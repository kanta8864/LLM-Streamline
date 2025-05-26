#!/bin/bash

#SBATCH --job-name="mse_training"
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-cpu=8000M
#SBATCH --account=education-eemcs-msc-cs

# Load necessary modules
module load 2024r1
module load cuda/12.1
module load python

# todo: you have to change this path to your own cache directory
export HF_HOME="/scratch/ktanahashi/huggingface_cache"

# Activate your virtual environment (created beforehand)
source ~/myenv/bin/activate

echo "Verifying Python version..."
which python3
python3 --version

# Start training
echo "Starting training script..."
srun --unbuffered ~/myenv/bin/python3 mseloss_entry.py


echo "Job finished."
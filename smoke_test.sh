#!/bin/bash
# File: run_job_smoke_test.sh

# --- Slurm Resource Request ---
#SBATCH --job-name=smoke-test
#SBATCH --partition=general
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=smoke_test_output_%j.log
#SBATCH --gres=gpu:l40:1

set -e

# --- Load Required Modules ---
echo "Loading modules..."
module use /opt/insy/modulefiles
module load cuda/12.1

# --- Debug: Show module environment ---
echo "=== DEBUG: Environment after module load ==="
echo "CUDA_HOME: ${CUDA_HOME:-'not set'}"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-'not set'}"

# --- Job Execution & Path Definitions ---
echo "Job started on $(hostname)"
CONTAINER_PATH="/tudelft.net/staff-umbrella/llmstreamline88/containers/my_env_v4.sif"
if [ ! -f "${CONTAINER_PATH}" ]; then
    echo "ERROR: Container file not found at ${CONTAINER_PATH}"
    exit 1
fi

# --- Set Caches to the Node's Local, Fast /tmp Directory ---
JOB_TMP_DIR="/tmp/${USER}/${SLURM_JOB_ID}"
mkdir -p "${JOB_TMP_DIR}"
export HF_HOME="${JOB_TMP_DIR}/huggingface"
export HF_DATASETS_CACHE="${JOB_TMP_DIR}/datasets"
export TRANSFORMERS_CACHE="${JOB_TMP_DIR}/transformers"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
echo "Using temporary cache for this job: ${JOB_TMP_DIR}"

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up temporary directory: ${JOB_TMP_DIR}"
    rm -rf "${JOB_TMP_DIR}"
}
trap cleanup EXIT

# --- Debug: Check CUDA installation on host ---
echo "=== DEBUG: Checking CUDA on host ==="
CUDA_PATHS="/usr/local/cuda /opt/cuda /usr/local/cuda-12.1 /opt/cuda-12.1 /opt/insy/cuda/12.1"
FOUND_CUDA=""

for cuda_path in $CUDA_PATHS; do
    echo "Checking: ${cuda_path}/bin/nvcc"
    if [ -f "${cuda_path}/bin/nvcc" ]; then
        FOUND_CUDA="${cuda_path}"
        echo "✅ Found CUDA at: ${FOUND_CUDA}"
        echo "NVCC version:"
        "${FOUND_CUDA}/bin/nvcc" --version || echo "Could not get nvcc version"
        break
    else
        echo "❌ Not found: ${cuda_path}/bin/nvcc"
    fi
done

if [ -z "$FOUND_CUDA" ]; then
    echo "Warning: Could not find CUDA installation with nvcc on host"
    # Fallback to module path
    FOUND_CUDA="/opt/insy/cuda/12.1"
    echo "Using fallback CUDA path: ${FOUND_CUDA}"
fi

# --- Debug: Check what's available in container ---
echo "=== DEBUG: Checking container CUDA availability ==="
echo "Testing container CUDA paths..."
apptainer exec --nv "$CONTAINER_PATH" bash -c "
echo 'Container CUDA check:'
for path in /usr/local/cuda /opt/cuda /usr/local/cuda-12.1 /opt/cuda-12.1 /opt/insy/cuda/12.1; do
    if [ -f \"\$path/bin/nvcc\" ]; then
        echo \"✅ Found in container: \$path/bin/nvcc\"
    else
        echo \"❌ Not in container: \$path/bin/nvcc\"
    fi
done
echo 'Searching for any nvcc in container:'
find /usr /opt -name 'nvcc' 2>/dev/null | head -5 || echo 'No nvcc found with find command'
"

# --- Run Code Inside Apptainer ---
echo "=== Starting container execution for smoke test ==="
echo "Using CUDA_HOME: ${FOUND_CUDA}"
echo "Binding CUDA path: ${FOUND_CUDA}:${FOUND_CUDA}"

apptainer exec \
    --nv \
    --bind "$(pwd)":/app \
    --bind "${JOB_TMP_DIR}":"${JOB_TMP_DIR}" \
    --bind "${FOUND_CUDA}":"${FOUND_CUDA}" \
    --pwd /app \
    --env "HF_HOME=${HF_HOME}" \
    --env "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
    --env "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}" \
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
    --env "DS_BUILD_OPS=0" \
    --env "CUDA_HOME=${FOUND_CUDA}" \
    --env "CUDA_PATH=${FOUND_CUDA}" \
    "$CONTAINER_PATH" \
    bash -c "
        echo '=== DEBUG: Inside container environment ==='
        echo 'CUDA_HOME: ${CUDA_HOME:-not set}'
        echo 'CUDA_PATH: ${CUDA_PATH:-not set}'
        echo 'DS_BUILD_OPS: ${DS_BUILD_OPS:-not set}'
        echo 'Checking if nvcc is accessible:'
        if [ -f '${FOUND_CUDA}/bin/nvcc' ]; then
            echo '✅ nvcc found at: ${FOUND_CUDA}/bin/nvcc'
            '${FOUND_CUDA}/bin/nvcc' --version | head -3
        else
            echo '❌ nvcc not found at: ${FOUND_CUDA}/bin/nvcc'
        fi
        echo '=== Running Python smoke test ==='
        /opt/venv/bin/python3 smoke_test.py
    "

echo "Job finished."
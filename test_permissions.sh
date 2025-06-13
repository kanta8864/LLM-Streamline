#!/bin/bash
#SBATCH --job-name=permission-test
#SBATCH --time=00:05:00
#SBATCH --partition=general
#SBATCH --output=permission_test_%j.log
#SBATCH --gres=gpu:l40:1   # Request the same type of compute node

# The directory you want to write to
TARGET_DIR="/tudelft.net/staff-umbrella/llmstreamline88"
TEST_FILE="${TARGET_DIR}/test_from_compute_node.txt"

echo "--- Permission Test on a COMPUTE NODE ---"
echo "Job running on: $(hostname)"
echo "User: $(whoami)"
echo "Attempting to write to file: ${TEST_FILE}"

# Attempt to create an empty file. The 'timeout' prevents it from hanging.
timeout 30s touch "${TEST_FILE}"

# Check the exit code of the 'touch' command
if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: Successfully created the test file."
    ls -l "${TEST_FILE}"
    rm "${TEST_FILE}"    # Clean up
else
    echo "❌ FAILURE: Could not create the test file. This confirms a permission issue on the compute node."
fi

echo "--- Test Complete ---"
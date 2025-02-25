#!/bin/bash

REPO_PATH=$1
TEST_COMMAND=$2
OUTPUT_FILE=$3

# Run the test command with perf to measure energy consumption
echo "$TEST_COMMAND"
PERF_OUTPUT=$(perf stat -e power/energy-pkg/ -e power/energy-cores/ -e power/energy-gpu/ $TEST_COMMAND 2>&1)

# Extract energy values from perf output
ENERGY_PKG=$(echo "$PERF_OUTPUT" | grep "power/energy-pkg/" | awk '{print $1}')
ENERGY_CORES=$(echo "$PERF_OUTPUT" | grep "power/energy-cores/" | awk '{print $1}')
ENERGY_GPU=$(echo "$PERF_OUTPUT" | grep "power/energy-gpu/" | awk '{print $1}')
echo "ENERGY_PKG: $ENERGY_PKG"
echo "ENERGY_CORES: $ENERGY_CORES"
echo "ENERGY_GPU: $ENERGY_GPU"

# Append results to file
echo "$(git -C $REPO_PATH rev-parse HEAD),$ENERGY_PKG,$ENERGY_CORES,$ENERGY_GPU" >> "$OUTPUT_FILE"

exit 0

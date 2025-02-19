#!/bin/bash

REPO_PATH=$1
TEST_COMMAND=$2
OUTPUT_FILE=$3

echo "Running energy test at commit: $(git -C $REPO_PATH rev-parse HEAD)"

# Get initial energy reading
START_ENERGY=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)

# Run the test command
eval "$TEST_COMMAND"
TEST_EXIT_CODE=$?

# Get final energy reading
END_ENERGY=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)

# Compute energy consumption in microjoules
ENERGY_USED=$((END_ENERGY - START_ENERGY))

# Append results to file
echo "$(git -C $REPO_PATH rev-parse HEAD),$ENERGY_USED,$TEST_EXIT_CODE" >> "$OUTPUT_FILE"

exit 0

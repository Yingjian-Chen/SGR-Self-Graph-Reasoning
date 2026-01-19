#!/bin/bash

echo "=== Start reasoningLLM API generation ==="
echo "Start time: $(date)"

mkdir -p logs
LOG_FILE="logs/api_generation-$(date +%Y%m%d-%H%M%S).log"

echo "Running API script... Logs -> $LOG_FILE"

python data/graph_data_build.py > "$LOG_FILE" 2>&1

echo "Finished API generation at: $(date)"
echo "Logs saved to: $LOG_FILE"


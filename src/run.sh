#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a unique job ID (using timestamp)
JOB_ID=$(date +%Y%m%d%H%M%S)
LOG_FILE="logs/${JOB_ID}_output.txt"

echo "Starting job with ID: $JOB_ID"
echo "Logs will be saved to: $LOG_FILE"

# Run the command in the background
nohup python -u -m scripts.averitec.evaluate --procedure_variant infact > "$LOG_FILE" 2>&1 &

echo "Job started in background."

#!/bin/bash
# Shell script to run TensorBoard for monitoring training
# TensorBoard will automatically detect and display all experiment runs

# TensorBoard log directory (parent directory containing all runs)
LOG_DIR="./logs/tensorboard"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: TensorBoard log directory not found at $LOG_DIR"
    echo "Please run training first using: python train.py"
    exit 1
fi

# Count number of runs
NUM_RUNS=$(find "$LOG_DIR" -maxdepth 1 -type d -name "run_*" | wc -l)

# Display info
echo "=================================================="
echo "Starting TensorBoard"
echo "=================================================="
echo "Log directory: $LOG_DIR"
echo "Detected runs: $NUM_RUNS"
echo ""
echo "TensorBoard will be available at:"
echo "  http://localhost:6006"
echo ""
echo "You can compare all experiments in the UI!"
echo "Press Ctrl+C to stop TensorBoard"
echo "=================================================="
echo ""

# Run TensorBoard
tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006

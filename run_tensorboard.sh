#!/bin/bash
# Shell script to run TensorBoard for monitoring training

# TensorBoard log directory
LOG_DIR="./logs/tensorboard"

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: TensorBoard log directory not found at $LOG_DIR"
    echo "Please run training first using: python train.py"
    exit 1
fi

# Display info
echo "=================================================="
echo "Starting TensorBoard"
echo "=================================================="
echo "Log directory: $LOG_DIR"
echo ""
echo "TensorBoard will be available at:"
echo "  http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo "=================================================="
echo ""

# Run TensorBoard
tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006

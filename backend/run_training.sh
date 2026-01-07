#!/bin/bash
# Script to train the medical misinformation detection model

cd "$(dirname "$0")"

echo "=========================================="
echo "Training Medical Misinformation Model"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: venv not found. Make sure dependencies are installed."
fi

echo ""
echo "Starting training..."
echo ""

# Run training script
python train_model.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="


#!/usr/bin/env bash
# build.sh - Render build script

set -o errexit  # exit on error

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models
mkdir -p data
mkdir -p logs

# Generate data and train model if not exists
if [ ! -f "models/loan_model.pkl" ]; then
    echo "Training model..."
    python data/generate_data.py
    python models/loan_model.py
fi

echo "Build completed successfully!"
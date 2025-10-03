#!/usr/bin/env bash
# build.sh - Render build script with error handling

set -o errexit  # exit on error

echo "Python version:"
python --version

echo "Pip version:"
pip --version

# Upgrade pip to latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install minimal dependencies first
echo "Installing core dependencies..."
pip install --no-cache-dir --only-binary=all -r requirements_render.txt

# Test dependencies
echo "Testing dependencies..."
python test_dependencies.py

# Create necessary directories
echo "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p logs

# Generate data and train model if not exists
if [ ! -f "models/loan_model.pkl" ]; then
    echo "Generating training data..."
    python data/generate_data.py
    echo "Training model..."
    python models/loan_model.py
else
    echo "Model already exists, skipping training..."
fi

echo "Build completed successfully!"
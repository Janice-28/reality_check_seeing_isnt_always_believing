echo '#!/bin/bash
# Setup script for DeepFakeAudioDetection

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda env create -f conda_env.yml
    echo "Environment created. Activate with: conda activate deepfakeaudiodetection"
else
    echo "Conda not found. Installing with pip..."
    pip install -r requirements.txt
fi

# Create directories
mkdir -p model
mkdir -p static
mkdir -p templates

echo "Setup complete!"' > DeepFakeAudioDetection/setup.sh

# Make executable
chmod +x DeepFakeAudioDetection/setup.sh
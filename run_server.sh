#!/bin/bash
# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepfakeimagedetect

# Run the server
cd /Users/janicemascarenhas/Documents/RealityCheck/reality_check/DeepFakeImageDetection
python app.py

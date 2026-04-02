#!/bin/bash
# Change to the video detection directory
cd "/Users/janicemascarenhas/Documents/RealityCheck/reality_check/DeepFakeVideoDetection"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepfakevideodetection

# Run the server
python run_server.py

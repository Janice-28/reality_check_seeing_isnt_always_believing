#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Activate conda environment
# Adjust this path to your conda installation
source ~/opt/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh

# Activate the environment and run the server
conda activate deepfakevideodetection
python run_server.py
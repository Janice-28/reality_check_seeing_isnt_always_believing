#!/bin/bash

# Start Reality Check application
echo "Starting Reality Check application..."

# Kill any existing processes
echo "Cleaning up existing processes..."
python cleanup.py

# Start main app
echo "Starting main application..."
python main_app.py
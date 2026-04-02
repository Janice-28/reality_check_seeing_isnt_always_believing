
import os
import sys
import platform

# Force use of CPU for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable Numba JIT if on M1 Mac
if platform.system() == "Darwin" and platform.processor() == "arm":
    os.environ["NUMBA_DISABLE_JIT"] = "1"

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import app from the current directory
try:
    from app import app
    
    # Print diagnostic info
    print("Audio Detection Server starting...")
    print(f"Python version: {sys.version}")
    
    # Try importing key dependencies
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy not available")
    
    try:
        import librosa
        print(f"Librosa version: {librosa.__version__}")
    except ImportError:
        print("Librosa not available")
        
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow not available")
        
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=8003, debug=False, threaded=True)

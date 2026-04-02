# tf_patch.py - Patch for TensorFlow on Apple Silicon
import platform
import os

# Only apply patches on M1/M2 Mac
if platform.system() == "Darwin" and platform.processor() == "arm":
    # Disable Metal plugin (can cause issues)
    os.environ["TF_DISABLE_METAL_PLUGIN"] = "1"
    
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Disable OneDNN optimizations
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    # Limit TF to use only one thread
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    
    print("Applied TensorFlow patches for Apple Silicon")
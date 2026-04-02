import h5py
import numpy as np
import json
import tensorflow as tf
import os

model_path = "/Users/janicemascarenhas/Documents/RealityCheck/reality_check/DeepFakeAudioDetection/model/audio_classifier.h5"

if not os.path.exists(model_path):
    print(f"Error: File not found at '{model_path}'")
    print("Please update the model_path variable with the correct path to your .h5 file")
    # List common directories to help locate the file
    print("\nSearching for .h5 files in current directory and subdirectories...")
    for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
        for file in files:
            if file.endswith('.h5'):
                print(f"Found .h5 file: {os.path.join(root, file)}")
    exit(1)

print(f"Analyzing model file: {model_path}")

# Option 1: List structure using h5py
print("\nModel Structure:")
try:
    with h5py.File(model_path, 'r') as f:
        # Print all groups
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            else:
                print(f"Group: {name}")
        
        f.visititems(print_structure)
except Exception as e:
    print(f"Error reading file with h5py: {e}")

# Option 2: Load and summarize with TensorFlow
try:
    print("\nLoading model with TensorFlow...")
    model = tf.keras.models.load_model(model_path)
    print("\nModel Summary:")
    model.summary()
    
    # Get layer information
    print("\nLayer Details:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}, Type: {layer.__class__.__name__}")
        print(f"  Input Shape: {layer.input_shape}")
        print(f"  Output Shape: {layer.output_shape}")
        print(f"  Parameters: {layer.count_params()}")
        print("")
    
    # Get model configuration
    print("\nModel Configuration:")
    config = model.get_config()
    print(json.dumps(str(config), indent=2))
    
except Exception as e:
    print(f"Error loading model with TensorFlow: {e}")
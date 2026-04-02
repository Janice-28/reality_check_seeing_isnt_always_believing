# Save as test_tf.py
import tensorflow as tf
import numpy as np
import os

# Set environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

print(f"TensorFlow version: {tf.__version__}")

# Try to create a simple model
try:
    print("Creating test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("Model created successfully!")
    
    # Try inference
    print("Testing inference...")
    dummy_input = np.random.random((1, 5)).astype(np.float32)
    result = model.predict(dummy_input, verbose=0)
    print(f"Inference result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
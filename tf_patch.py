    # tf_patch.py
import tensorflow as tf
import os
   
   # Set TensorFlow options for M1 Mac
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
   
   # Check if tensor attribute exists, if not add it
if not hasattr(tf, 'tensor'):
       tf.tensor = tf.convert_to_tensor
       print("Applied TensorFlow patch: added missing 'tensor' attribute")
   
   # Fix for potential memory issues on M1 Mac
try:
       gpus = tf.config.experimental.list_physical_devices('GPU')
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
       print(f"Set memory growth for {len(gpus)} GPUs")
except Exception as e:
       print(f"GPU configuration error: {e}")
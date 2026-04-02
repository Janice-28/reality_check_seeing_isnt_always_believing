import tensorflow as tf
import os
import traceback
import numpy as np

# Define constants (must match those used during training)
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def build_model():
    """Build the model architecture to match the saved weights"""
    # Input layers
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES), name='frame_features_input')
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool", name='mask_input')
    
    # GRU layers with batch normalization
    x = tf.keras.layers.GRU(64, return_sequences=True)(frame_features_input, mask=mask_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GRU(32, return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GRU(16)(x)
    
    # Dense layers with dropout
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    # Create model
    model = tf.keras.Model([frame_features_input, mask_input], output)
    
    return model

def load_model_weights(model_path):
    """Load model weights from H5 file into a freshly built model"""
    try:
        # Build model with the right architecture
        model = build_model()
        
        # Try to load weights only (not the full model)
        model.load_weights(model_path, by_name=True)
        
        # Compile model
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["accuracy"]
        )
        
        print(f"Successfully loaded model weights from {model_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model weights: {e}")
        traceback.print_exc()
        return None

def build_feature_extractor():
    """Build the InceptionV3 feature extractor"""
    try:
        feature_extractor = tf.keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input

        inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return tf.keras.Model(inputs, outputs, name="feature_extractor")
    except Exception as e:
        print(f"Error building feature extractor: {e}")
        traceback.print_exc()
        return None

def test_model(model, feature_extractor):
    """Test the model with dummy data to ensure it works"""
    try:
        # Create dummy data
        frame_features = np.random.random((1, MAX_SEQ_LENGTH, NUM_FEATURES)).astype(np.float32)
        frame_mask = np.ones((1, MAX_SEQ_LENGTH), dtype=bool)
        
        # Test prediction
        prediction = model.predict([frame_features, frame_mask], verbose=0)
        
        print("Model test successful!")
        print(f"Test prediction shape: {prediction.shape}, value: {prediction[0][0]}")
        return True
    
    except Exception as e:
        print(f"Error testing model: {e}")
        traceback.print_exc()
        return False

# Test on known videos to determine if prediction needs to be inverted
def determine_prediction_inversion(model, feature_extractor):
    """Test on known videos to determine if prediction needs to be inverted"""
    import cv2
    import glob
    
    def load_video(path, max_frames=20):
        """Load video frames"""
        cap = cv2.VideoCapture(path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            y, x = frame.shape[0:2]
            min_dim = min(y, x)
            start_x = (x // 2) - (min_dim // 2)
            start_y = (y // 2) - (min_dim // 2)
            frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
            
        cap.release()
        
        # If we don't have enough frames, loop the video
        if 0 < len(frames) < max_frames:
            frames = frames * (max_frames // len(frames) + 1)
            frames = frames[:max_frames]
            
        return np.array(frames) if frames else None
    
    def predict_on_video(video_path):
        """Make prediction on a video"""
        # Extract frames from video
        frames = load_video(video_path)
        if frames is None:
            return None
        
        # Extract features from frames
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        
        for i, frame in enumerate(frames):
            frame_features[0, i, :] = feature_extractor.predict(frame[np.newaxis], verbose=0)
            frame_mask[0, i] = 1  # Mark as valid
        
        # Make prediction
        prediction = model.predict([frame_features, frame_mask], verbose=0)[0, 0]
        return float(prediction)
    
    try:
        # Find test videos
        real_videos = glob.glob("dataset/test_videos/*real*.mp4") + \
                      glob.glob("dataset/train_sample_videos/*real*.mp4")
        fake_videos = glob.glob("dataset/test_videos/*fake*.mp4") + \
                      glob.glob("dataset/train_sample_videos/*fake*.mp4")
        
        if not real_videos or not fake_videos:
            print("Insufficient test videos found - using default inversion")
            return True
        
        # Test on a sample of videos
        real_scores = []
        for video_path in real_videos[:2]:  # Test first 2 videos
            score = predict_on_video(video_path)
            if score is not None:
                real_scores.append(score)
                print(f"Real video {video_path}: {score}")
        
        fake_scores = []
        for video_path in fake_videos[:2]:  # Test first 2 videos
            score = predict_on_video(video_path)
            if score is not None:
                fake_scores.append(score)
                print(f"Fake video {video_path}: {score}")
        
        # Determine if inversion is needed
        if real_scores and fake_scores:
            avg_real = sum(real_scores) / len(real_scores)
            avg_fake = sum(fake_scores) / len(fake_scores)
            
            print(f"Average real score: {avg_real}")
            print(f"Average fake score: {avg_fake}")
            
            # If real videos have higher scores than fake videos, inversion is needed
            needs_inversion = avg_real > avg_fake
            print(f"Prediction inversion needed: {needs_inversion}")
            return needs_inversion
        
        return True  # Default to inversion
    
    except Exception as e:
        print(f"Error determining prediction inversion: {e}")
        return True  # Default to inversion

# Usage example
if __name__ == "__main__":
    model_path = "./model/deepfake_video_model.h5"
    
    # Load model
    model = load_model_weights(model_path)
    
    if model is not None:
        # Build feature extractor
        feature_extractor = build_feature_extractor()
        
        if feature_extractor is not None:
            # Test the model
            test_model(model, feature_extractor)
            
            # Determine if prediction inversion is needed
            invert_prediction = determine_prediction_inversion(model, feature_extractor)
            print(f"Should invert predictions: {invert_prediction}")
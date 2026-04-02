import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from audio_deepfake_detection import load_and_process_audio, build_model

# Define directories
REAL_DIR = 'datasets/real_audio'
FAKE_DIR = 'datasets/fake_audio'
MODEL_SAVE_PATH = 'model/audio_classifier.h5'

def load_dataset(real_dir, fake_dir):
    """Load and process audio files from real and fake directories"""
    X = []
    y = []
    
    # Process real audio files
    for filename in os.listdir(real_dir):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(real_dir, filename)
            mel_spec = load_and_process_audio(file_path)
            if mel_spec is not None:
                X.append(mel_spec)
                y.append(1)  # 1 for real
    
    # Process fake audio files
    for filename in os.listdir(fake_dir):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(fake_dir, filename)
            mel_spec = load_and_process_audio(file_path)
            if mel_spec is not None:
                X.append(mel_spec)
                y.append(0)  # 0 for fake
    
    return np.array(X), np.array(y)

def train_model():
    """Train the audio deepfake detection model"""
    # Load dataset
    X, y = load_dataset(REAL_DIR, FAKE_DIR)
    print(f"Dataset loaded: {len(X)} samples")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get input shape
    input_shape = X_train[0].shape
    print(f"Input shape: {input_shape}")
    
    # Build model
    model = build_model(input_shape)
    model.summary()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, 
            save_best_only=True, 
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate on validation set
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    train_model()
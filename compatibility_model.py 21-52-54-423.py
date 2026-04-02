"""
Create a simpler model compatible with the original structure
for fallback purposes.
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# Constants
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109
NUM_CLASSES = 2

def extract_basic_features(audio_path, sr=SAMPLE_RATE, duration=DURATION):
    try:
        # Load audio file
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure consistent shape
        if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]
        
        return mel_spectrogram
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return np.zeros((N_MELS, MAX_TIME_STEPS))

def build_basic_model(input_shape):
    model_input = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    model_output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Define paths
    DATASET_PATH = "LA/ASVspoof2019_LA_train/flac"
    LABEL_FILE_PATH = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    
    # Load labels
    labels = {}
    with open(LABEL_FILE_PATH, 'r') as label_file:
        lines = label_file.readlines()
        
    for line in lines:
        parts = line.strip().split()
        file_name = parts[1]
        label = 1 if parts[-1] == "bonafide" else 0
        labels[file_name] = label
    
    # Prepare data
    X = []
    y = []
    
    # Limit to a smaller subset for quicker training of the fallback model
    count = 0
    max_samples = 5000  # Limit to 5000 samples for faster training
    
    for file_name, label in labels.items():
        if count >= max_samples:
            break
            
        file_path = os.path.join(DATASET_PATH, file_name + ".flac")
        if not os.path.exists(file_path):
            continue
        
        # Extract features
        features = extract_basic_features(file_path)
        X.append(features)
        y.append(label)
        count += 1
        
        if count % 100 == 0:
            print(f"Processed {count} files")
    
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension for CNN
    X = np.expand_dims(X, axis=-1)
    
    # Convert labels to one-hot encoding
    y_encoded = to_categorical(y, NUM_CLASSES)
    
    # Split data
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y_encoded[:split_index], y_encoded[split_index:]
    
    # Build model
    input_shape = (N_MELS, MAX_TIME_STEPS, 1)
    model = build_basic_model(input_shape)
    model.summary()
    
    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val))
    
    # Save model
    model.save("audio_classifier.h5")
    print("Fallback model saved as 'audio_classifier.h5'")

if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
N_MELS = 128
MAX_TIME_STEPS = 469


def build_deepfake_detector(input_shape=None):
    """
    Build CNN model for audio deepfake detection
    
    Parameters:
    -----------
    input_shape : tuple, optional
        Input shape for the model (default: (N_MELS, MAX_TIME_STEPS, 1))
        
    Returns:
    --------
    tensorflow.keras.Model
        Compiled model
    """
    if input_shape is None:
        input_shape = (N_MELS, MAX_TIME_STEPS, 1)
    
    # Create a sequential model
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Final layers
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model


def build_ensemble_model():
    """
    Build an ensemble of models for more robust detection
    
    Returns:
    --------
    tuple
        (ensemble_model, individual_models)
    """
    # Create individual models with different architectures
    input_shape = (N_MELS, MAX_TIME_STEPS, 1)
    
    # Model 1: Standard CNN
    model1 = build_deepfake_detector(input_shape)
    
    # Model 2: Deeper CNN with residual connections
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x_shortcut = x
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Upsample shortcut to match x
    x_shortcut = Conv2D(64, (1, 1), strides=(2, 2))(x_shortcut)
    x = tf.keras.layers.add([x, x_shortcut])
    
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model2 = Model(inputs=inputs, outputs=outputs)
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Model 3: Lightweight model focused on spectral features
    model3 = Sequential([
        Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((4, 4)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((4, 4)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model3.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create ensemble model
    ensemble_input = Input(shape=input_shape)
    
    # Get predictions from each model
    pred1 = model1(ensemble_input)
    pred2 = model2(ensemble_input)
    pred3 = model3(ensemble_input)
    
    # Average predictions
    ensemble_output = tf.keras.layers.Average()([pred1, pred2, pred3])
    
    ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble_model, [model1, model2, model3]


def load_model(model_path):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
        
    Returns:
    --------
    tensorflow.keras.Model or None
        Loaded model or None if loading fails
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def save_model(model, save_path):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Model to save
    save_path : str
        Path to save model
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    model = build_deepfake_detector()
    model.summary()
    
    print("\nEnsemble model:")
    ensemble_model, _ = build_ensemble_model()
    ensemble_model.summary()
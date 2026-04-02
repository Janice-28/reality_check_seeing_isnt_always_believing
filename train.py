
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import argparse
import logging
import time
from tqdm import tqdm

# Import local modules
from audio_processing import load_audio, extract_mel_spectrogram, split_audio_into_chunks
from model import build_deepfake_detector, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 30
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_classifier.h5')


def load_dataset(real_dir, fake_dir, max_samples=None):
    """
    Load dataset from directories
    
    Parameters:
    -----------
    real_dir : str
        Directory containing real audio files
    fake_dir : str
        Directory containing fake audio files
    max_samples : int, optional
        Maximum number of samples to load from each class
        
    Returns:
    --------
    tuple
        (X, y) - features and labels
    """
    X = []
    y = []
    
    # Process real audio files
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav') or f.endswith('.mp3')]
    if max_samples:
        real_files = real_files[:max_samples]
    
    logger.info(f"Processing {len(real_files)} real audio files...")
    for filename in tqdm(real_files):
        file_path = os.path.join(real_dir, filename)
        
        # Load audio
        audio, sr = load_audio(file_path)
        if audio is None:
            continue
        
        # Split into chunks
        chunks = split_audio_into_chunks(audio, sr)
        
        # Process each chunk
        for chunk in chunks:
            # Extract features
            features = extract_mel_spectrogram(chunk, sr)
            X.append(features)
            y.append(1)  # 1 for real
    
    # Process fake audio files
    fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav') or f.endswith('.mp3')]
    if max_samples:
        fake_files = fake_files[:max_samples]
    
    logger.info(f"Processing {len(fake_files)} fake audio files...")
    for filename in tqdm(fake_files):
        file_path = os.path.join(fake_dir, filename)
        
        # Load audio
        audio, sr = load_audio(file_path)
        if audio is None:
            continue
        
        # Split into chunks
        chunks = split_audio_into_chunks(audio, sr)
        
        # Process each chunk
        for chunk in chunks:
            # Extract features
            features = extract_mel_spectrogram(chunk, sr)
            X.append(features)
            y.append(0)  # 0 for fake
    
    return np.array(X), np.array(y)


def train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train the deepfake detection model
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
        
    Returns:
    --------
    tuple
        (model, history)
    """
    # Get input shape from the data
    input_shape = X_train[0].shape
    logger.info(f"Input shape: {input_shape}")
    
    # Build model
    model = build_deepfake_detector(input_shape)
    model.summary(print_fn=logger.info)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train model
    logger.info(f"Starting model training with {epochs} epochs, batch size {batch_size}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Evaluate model
    logger.info("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
    logger.info(f"Classification report:\n{report}")
    
    # Plot results
    plot_evaluation_results(y_test, y_pred, y_pred_prob, history)
    
    return {
        'accuracy': test_accuracy,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }


def plot_evaluation_results(y_test, y_pred, y_pred_prob, history=None):
    """
    Plot evaluation results
    
    Parameters:
    -----------
    y_test : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    y_pred_prob : numpy.ndarray
        Prediction probabilities
    history : tensorflow.keras.callbacks.History, optional
        Training history
        
    Returns:
    --------
    None
    """
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Fake', 'Real']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Plot training history if available
    if history is not None:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(2, 2, 4)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation_results.png')
    plt.show()


def main():
    """Main function to run training and evaluation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model")
    parser.add_argument("--real-dir", type=str, default="datasets/real_audio", help="Directory containing real audio files")
    parser.add_argument("--fake-dir", type=str, default="datasets/fake_audio", help="Directory containing fake audio files")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples per class")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to save model")
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    X, y = load_dataset(args.real_dir, args.fake_dir, args.max_samples)
    logger.info(f"Dataset loaded: {len(X)} samples ({np.sum(y)} real, {len(y) - np.sum(y)} fake)")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, args.epochs, args.batch_size)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, args.model_path)
    logger.info(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
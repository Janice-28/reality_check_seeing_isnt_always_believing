# ensemble_prediction.py
import numpy as np
import tensorflow as tf
from scipy.special import softmax

def ensemble_prediction(model, inputs, threshold=0.5):
    """
    Make a prediction using ensemble of augmented inputs
    
    Args:
        model: The trained model
        inputs: List of input features from augmented audio
        threshold: Classification threshold
    
    Returns:
        is_fake: Boolean indicating if audio is fake
        confidence: Confidence score
        all_predictions: Individual predictions for each augmentation
    """
    if not inputs:
        return True, 0.5, []
    
    # Make predictions for each augmented version
    all_predictions = []
    all_confidences = []
    
    for input_features in inputs:
        # Add batch dimension
        input_batch = np.expand_dims(input_features, axis=0)
        
        # Get prediction
        prediction = model.predict(input_batch, verbose=0)[0]
        all_predictions.append(prediction)
        
        # Get confidence
        confidence = np.max(prediction)
        all_confidences.append(confidence)
    
    # Average predictions
    avg_prediction = np.mean(all_predictions, axis=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(avg_prediction)
    confidence = avg_prediction[predicted_class]
    
    # Determine if fake (assuming class 0 is fake, class 1 is real)
    is_fake = predicted_class == 0
    
    # For display purposes, also calculate:
    # 1. Agreement rate (how many augmentations agree with final prediction)
    agreement_rate = sum(1 for p in all_predictions if np.argmax(p) == predicted_class) / len(all_predictions)
    
    # 2. Confidence stability (std dev of confidence scores)
    confidence_stability = 1.0 - np.std([p[predicted_class] for p in all_predictions])
    
    return is_fake, confidence, {
        'predictions': all_predictions,
        'agreement_rate': agreement_rate,
        'confidence_stability': confidence_stability
    }
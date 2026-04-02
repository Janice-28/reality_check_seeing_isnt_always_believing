from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import tempfile
import numpy as np
import time
import warnings
import logging
import traceback
import json
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Increase maximum file size limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_classifier.h5')
DETECTION_THRESHOLD = 0.4  # Lower threshold to catch more fakes

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# M1 Mac-specific configuration
if platform.system() == "Darwin" and platform.processor() == "arm":
    os.environ["NUMBA_DISABLE_JIT"] = "1"  # Disable Numba JIT
    os.environ["TF_DISABLE_METAL_PLUGIN"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Try to import the patch
    try:
        import tf_patch
    except ImportError:
        logger.warning("tf_patch.py not found. TensorFlow compatibility may be affected.")

# Import dependencies directly instead of lazy loading
try:
    import librosa
    import soundfile as sf
    from audio_processing import (
        load_audio, extract_mel_spectrogram, analyze_acoustic_features, 
        get_deepfake_indicators, simple_audio_analysis, detect_phase_inconsistency
    )
    DEPENDENCIES_AVAILABLE = True
    logger.info("Audio processing dependencies loaded successfully")
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    logger.error(f"Failed to import audio processing dependencies: {e}")

# Try to import tensorflow only if needed for model
try:
    import tensorflow as tf
    # Configure TensorFlow to use memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info(f"Set memory growth for {len(physical_devices)} GPUs")
    TF_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - model predictions will be placeholders")
except Exception as e:
    TF_AVAILABLE = True
    logger.warning(f"TensorFlow configuration error: {e}")

# Global variables
model = None

# Load model if TensorFlow is available
if TF_AVAILABLE:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Test prediction with random input
        dummy_input = np.random.random((1, 128, 109, 1)).astype(np.float32)
        test_pred = model.predict(dummy_input, verbose=0)
        logger.info(f"Test prediction: {test_pred}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        logger.info("Server will start without model - predictions will return placeholder results")

def load_model_with_retry(model_path, max_attempts=3):
    """Load model with retry logic"""
    for attempt in range(max_attempts):
        try:
            model = tf.keras.models.load_model(model_path)
            # Test with dummy input to verify model works
            dummy_input = np.random.random((1, 128, 109, 1)).astype(np.float32)
            _ = model.predict(dummy_input, verbose=0)
            logger.info(f"Model loaded successfully on attempt {attempt+1}")
            return model
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)  # Wait before retry
    
    logger.error("All model loading attempts failed")
    return None

def get_confidence_rating(confidence):
    """Convert numerical confidence to a rating"""
    if confidence > 0.9:
        return "Very High"
    elif confidence > 0.75:
        return "High"
    elif confidence > 0.6:
        return "Moderate"
    elif confidence > 0.5:
        return "Low"
    else:
        return "Very Low"

def combined_prediction(audio_data, sr, mel_spec=None):
    """Combine model prediction with acoustic analysis for more robust detection"""
    # Get acoustic features
    acoustic_features = analyze_acoustic_features(audio_data, sr)
    
    # Get deepfake indicators
    indicators = get_deepfake_indicators(acoustic_features)
    
    # Calculate acoustic score
    if indicators:
        acoustic_score = sum(ind['score'] for ind in indicators) / len(indicators)
    else:
        acoustic_score = 0.3  # Default mild suspicion
    
    # Get model prediction if available
    model_score = 0.5  # Neutral default
    model_confidence = 0.0
    
    if model is not None and TF_AVAILABLE and mel_spec is not None:
        try:
            # Add batch dimension if needed
            if len(mel_spec.shape) == 3:
                mel_spec = np.expand_dims(mel_spec, axis=0)
                
            # Get prediction
            prediction = model.predict(mel_spec, verbose=0)
            
            # Get fake probability (index 1 in softmax output)
            model_score = float(prediction[0, 1])
            model_confidence = max(prediction[0, 0], prediction[0, 1])
            
            logger.info(f"Raw model score (fake probability): {model_score:.4f}")
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
    
    # Check for phase inconsistency
    try:
        phase_inconsistent = detect_phase_inconsistency(audio_data, sr)
        if phase_inconsistent:
            acoustic_score = max(acoustic_score, 0.7)
            indicators.append({
                'name': 'phase_inconsistency',
                'description': 'Unnatural phase relationships detected in audio',
                'score': 0.8
            })
    except Exception as e:
        logger.warning(f"Phase inconsistency detection failed: {e}")
    
    # Calculate weighted final score
    # Give more weight to acoustic analysis if model confidence is low
    if model_confidence > 0.8:
        final_score = (model_score * 0.7) + (acoustic_score * 0.3)
    else:
        final_score = (model_score * 0.4) + (acoustic_score * 0.6)
    
    # Check for strong indicators that should override model
    strong_indicators = [ind for ind in indicators if ind['score'] > 0.8]
    if strong_indicators and final_score < 0.5:
        final_score = 0.7  # Override with high fake probability
    
    logger.info(f"Combined score: {final_score:.4f} (model: {model_score:.4f}, acoustic: {acoustic_score:.4f})")
    
    # Determine prediction with lower threshold
    is_fake = final_score > DETECTION_THRESHOLD
    confidence = max(0.6, final_score if is_fake else (1.0 - final_score))
    
    return "FAKE" if is_fake else "REAL", confidence, indicators

@app.route('/')
def home():
    """Render home page"""
    return render_template('audio.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "dependencies_available": DEPENDENCIES_AVAILABLE,
        "detection_threshold": DETECTION_THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    start_time = time.time()
    
    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        return jsonify({
            "error": "Dependencies not available",
            "message": "The audio processing dependencies are not available."
        }), 500
    
    # Check if audio file was provided
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio = request.files['audio']
    
    # Validate file
    filename = audio.filename
    if not filename:
        return jsonify({'error': 'Invalid filename'}), 400
    
    logger.info(f"Processing file: {filename}")
    
    # Save audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    audio.save(temp_file.name)
    temp_file.close()
    
    try:
        # Process the audio file
        audio_data, sr = load_audio(temp_file.name)
        
        if audio_data is None:
            return jsonify({
                "error": "Failed to process audio file",
                "message": f"Error: {sr}. Please try with a different audio file or format."
            }), 422
        
        # Extract mel spectrogram
        mel_spec = extract_mel_spectrogram(audio_data, sr)
        
        # Get combined prediction
        model_prediction, confidence, indicators = combined_prediction(audio_data, sr, mel_spec)
        
        # Get acoustic features
        acoustic_features = analyze_acoustic_features(audio_data, sr)
        
        # Get confidence rating
        confidence_rating = get_confidence_rating(confidence)
        
        # Prepare response
        response = {
            "prediction": model_prediction,
            "confidence": float(confidence),
            "confidence_rating": confidence_rating,
            "indicators_found": len(indicators),
            "key_indicators": indicators,
            "acoustic_features": {k: v for k, v in acoustic_features.items() if not isinstance(v, (list, np.ndarray))},
            "processing_time": time.time() - start_time
        }
        
        logger.info(f"Prediction: {response['prediction']}, Confidence: {confidence:.4f}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "An error occurred during processing",
            "details": str(e),
            "message": "Please try with a different audio file or format."
        }), 500
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass

@app.route('/debug', methods=['POST'])
def debug_audio():
    """Debug endpoint for analyzing audio files"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio = request.files['audio']
    filename = audio.filename
    
    # Save audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
    audio.save(temp_file.name)
    temp_file.close()
    
    try:
        # Load audio
        audio_data, sr = load_audio(temp_file.name)
        
        if audio_data is None:
            return jsonify({
                "error": "Failed to process audio file",
                "message": f"Error: {sr}."
            }), 422
        
        # Extract features
        mel_spec = extract_mel_spectrogram(audio_data, sr)
        acoustic_features = analyze_acoustic_features(audio_data, sr)
        indicators = get_deepfake_indicators(acoustic_features)
        
        # Get raw model prediction
        model_prediction = None
        if model is not None and mel_spec is not None:
            try:
                mel_batch = np.expand_dims(mel_spec, axis=0)
                prediction = model.predict(mel_batch, verbose=0)
                fake_prob = prediction[0, 1]
                real_prob = prediction[0, 0]
                model_prediction = {
                    "real_probability": float(real_prob),
                    "fake_probability": float(fake_prob)
                }
            except Exception as e:
                model_prediction = {"error": str(e)}
        
        # Run combined prediction
        final_prediction, confidence, _ = combined_prediction(audio_data, sr, mel_spec)
        
        # Try phase inconsistency detection
        phase_inconsistent = False
        try:
            phase_inconsistent = detect_phase_inconsistency(audio_data, sr)
        except Exception as e:
            logger.warning(f"Phase detection failed: {e}")
        
        # Prepare detailed response
        response = {
            "filename": filename,
            "audio_stats": {
                "duration": len(audio_data) / sr,
                "sample_rate": sr,
                "mean_amplitude": float(np.mean(np.abs(audio_data))),
                "std_amplitude": float(np.std(audio_data))
            },
            "acoustic_features": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                for k, v in acoustic_features.items()},
            "deepfake_indicators": indicators,
            "phase_inconsistency_detected": phase_inconsistent,
            "model_prediction": model_prediction,
            "final_prediction": final_prediction,
            "confidence": float(confidence)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": "An error occurred during processing",
            "details": str(e)
        }), 500
    finally:
        # Clean up
        try:
            os.unlink(temp_file.name)
        except:
            pass

@app.route('/simple_test', methods=['POST'])
def simple_test():
    """Simple test endpoint for basic audio processing"""
    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        return jsonify({
            'success': False,
            'error': "Dependencies not available"
        }), 500
    
    # Check if audio file was provided
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio = request.files['audio']
    
    # Save audio to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1])
    audio.save(temp_file.name)
    temp_file.close()
    
    try:
        # Try to load the file
        audio_data, sr = load_audio(temp_file.name)
        
        if audio_data is None:
            return jsonify({
                'success': False,
                'error': sr  # Error message
            }), 400
        
        # Get basic audio information
        duration = len(audio_data) / sr
        
        return jsonify({
            'success': True,
            'message': 'Audio loaded successfully',
            'details': {
                'samples': len(audio_data),
                'samplerate': sr,
                'duration': duration,
                'channels': 1  # We convert to mono
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500
    finally:
        # Clean up
        try:
            os.unlink(temp_file.name)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003, debug=False, threaded=False)
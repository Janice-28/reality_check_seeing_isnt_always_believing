from flask import Flask, request, jsonify, render_template
import os
import torch
import numpy as np
from PIL import Image
import cv2
import logging
import time
import hashlib
from werkzeug.utils import secure_filename
from transformers import ViTForImageClassification, ViTImageProcessor
import platform
import sys
import os
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

import site
print(f"Site packages: {site.getsitepackages()}")
   
# M1 Mac-specific configuration
if platform.system() == "Darwin" and platform.processor() == "arm":
       os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
       
       # Import the patch
       try:
           import tf_patch
       except ImportError:
           print("Warning: tf_patch.py not found. TensorFlow compatibility may be affected.")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Constants
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit

# Initialize the ViT model
try:
    # Try to load directly from HuggingFace since we can see the model structure in your files
    model = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    processor = ViTImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    
    # Set the model to evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model loaded successfully from HuggingFace on {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_ela(img_path, quality=90):
    """Performs Error Level Analysis to detect image manipulation"""
    try:
        # Save original image at high quality
        temp_orig_path = os.path.join(UPLOAD_FOLDER, "temp_orig.jpg")
        temp_saved_path = os.path.join(UPLOAD_FOLDER, "temp_saved.jpg")
        
        # Open the image
        img = Image.open(img_path).convert('RGB')
        
        # Save at high quality
        img.save(temp_orig_path, 'JPEG', quality=100)
        
        # Save at a lower quality
        img.save(temp_saved_path, 'JPEG', quality=quality)
        
        # Open both images
        orig_img = cv2.imread(temp_orig_path)
        saved_img = cv2.imread(temp_saved_path)
        
        # Calculate the difference
        diff = cv2.absdiff(orig_img, saved_img)
        
        # Calculate ELA metrics
        ela_mean = np.mean(diff) / 255.0
        ela_std = np.std(diff) / 255.0
        
        # Clean up
        os.remove(temp_orig_path)
        os.remove(temp_saved_path)
        
        # Higher values indicate more compression artifacts (potential manipulation)
        ela_score = min(ela_std / (ela_mean + 1e-5), 1.0)
        
        return ela_score
    except Exception as e:
        logger.error(f"Error in ELA analysis: {e}")
        return 0.0

def analyze_noise(img_path):
    """Analyzes image noise patterns to detect inconsistencies"""
    try:
        # Open the image
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Extract noise by subtracting denoised image
        noise = cv2.absdiff(gray, denoised)
        
        # Calculate noise statistics
        noise_mean = float(np.mean(noise) / 255.0)
        noise_std = float(np.std(noise) / 255.0)
        
        # Calculate local noise variance across the image
        block_size = 8
        h, w = noise.shape
        local_vars = []
        
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = noise[i:i+block_size, j:j+block_size]
                local_vars.append(np.var(block))
        
        # Variance of local variances - higher means inconsistent noise (potential deepfake)
        if local_vars:
            noise_consistency = float(np.var(local_vars) / (np.mean(local_vars) + 1e-5))
        else:
            noise_consistency = 0.0
            
        # Normalize score
        noise_score = min(noise_consistency, 1.0)
        
        return noise_score
    except Exception as e:
        logger.error(f"Error in noise analysis: {e}")
        return 0.0

def detect_faces(img_path):
    """Simple face detection to count faces in image"""
    try:
        # Load the pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Read the image
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces)
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return 0
# Add this to your app.py to ensure proper error handling and dependency verification

def verify_dependencies():
    """Verify critical dependencies are properly installed"""
    try:
        import numpy as np
        import librosa
        import numba
        
        # Log versions for debugging
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Librosa version: {librosa.__version__}")
        logger.info(f"Numba version: {numba.__version__}")
        
        # Test basic librosa functionality with Numba acceleration
        test_audio = np.random.random(16000)  # 1 second of random audio
        _ = librosa.feature.melspectrogram(y=test_audio, sr=16000)
        
        logger.info("Audio processing dependencies verified successfully")
        return True
    except ImportError as e:
        logger.error(f"Dependency verification failed - import error: {e}")
        return False
    except Exception as e:
        logger.error(f"Dependency verification failed - runtime error: {e}")
        return False

# Call this during application initialization
if not verify_dependencies():
    logger.warning("Running with limited functionality due to dependency issues")
    
def predict_image(img_path):
    """Run prediction on an image using our ViT model"""
    try:
        if model is None or processor is None:
            return {
                "error": "Model not loaded",
                "prediction": "ERROR",
                "confidence": 0.0
            }
            
        # Load and preprocess the image
        image = Image.open(img_path).convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Process image for ViT model
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get prediction
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Extract prediction and confidence
        fake_prob = probabilities[0][1].item()
        real_prob = probabilities[0][0].item()
        
        # Run forensic analyses
        ela_score = perform_ela(img_path)
        noise_score = analyze_noise(img_path)
        
        # Count faces in the image
        face_count = detect_faces(img_path)
        has_face = face_count > 0
        
        # Calculate face analysis score - if faces are detected, this is more likely a real portrait
        # Lower score means more likely to be real
        face_score = 0.5  # Default neutral score
        if has_face and face_count == 1:
            # For portrait images, be more suspicious
            weights = [0.7, 0.0, 0.1, 0.1, 0.1] 
            face_score = 0.3  # More likely real
            # Multiple faces don't change the score much
        else:
            face_score = 0.45
        
        # Calculate a combined forensic score
        forensic_score = (ela_score * 0.5 + noise_score * 0.5)
        
        # Combine all scores with weights - CONSISTENT WITH ensemble_model.py
        # Primary model, secondary (none here), face analysis, noise, ela
        weights = [0.5, 0.0, 0.2, 0.15, 0.15]
        
        # If no face detected, reduce weight of face analysis
        if not has_face:
            weights = [0.6, 0.0, 0.0, 0.2, 0.2]
            
        # Compute ensemble score
        scores = [fake_prob, 0.0, face_score, noise_score, ela_score]
        ensemble_score = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        
        # Calculate image naturalness as inverse of forensic score
        image_naturalness = 1.0 - forensic_score
        
        # Determine final prediction with adjusted threshold
        prediction = "FAKE" if ensemble_score >= 0.55 else "REAL"  # Increased threshold
        confidence = ensemble_score if prediction == "FAKE" else (1 - ensemble_score)
        
        # Add some logging for debugging
        logger.info(f"Scores: primary={fake_prob:.3f}, face={face_score:.3f}, noise={noise_score:.3f}, ela={ela_score:.3f}")
        logger.info(f"Ensemble score: {ensemble_score:.3f}, Final prediction: {prediction} with confidence {confidence:.3f}")
        
        # Create confidence rating
        if confidence > 0.9:
            confidence_rating = "Very High"
        elif confidence > 0.75:
            confidence_rating = "High"
        elif confidence > 0.6:
            confidence_rating = "Moderate"
        else:
            confidence_rating = "Low"
        
        # Prepare key indicators
        key_indicators = []
        
        if ela_score > 0.6:
            key_indicators.append({
                "description": "Error level analysis indicates possible manipulation"
            })
            
        if noise_score > 0.6:
            key_indicators.append({
                "description": "Inconsistent noise patterns detected across the image"
            })
            
        if fake_prob > 0.8:
            key_indicators.append({
                "description": "Strong AI model indicators of manipulation"
            })
        
        # Result dictionary
        result = {
            "prediction": prediction,
            "confidence": float(confidence),
            "confidence_rating": confidence_rating,
            "analysis": {
                "image_naturalness": float(image_naturalness),
                "forensic_score": float(forensic_score),
                "primary_model_score": float(fake_prob),
                "face_analysis_score": float(face_score),
                "noise_analysis_score": float(noise_score),
                "ela_score": float(ela_score),
                "ensemble_score": float(ensemble_score),
                "image_dimensions": {
                    "width": width,
                    "height": height
                },
                "faces_detected": face_count
            }
        }
        
        # Add key indicators if any
        if key_indicators:
            result["key_indicators"] = key_indicators
            
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            "error": str(e),
            "prediction": "ERROR",
            "confidence": 0.0
        }

@app.route('/')
def home():
    return render_template('image.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG, PNG, JPEG, BMP, TIFF or WEBP image.'}), 400
    
    # Size validation
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large (maximum 10MB)'}), 400
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Size validation
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large (maximum 10MB)'}), 400
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{int(time.time())}_{filename}")
    file.save(temp_path)
    
    try:
        # Run prediction
        result = predict_image(temp_path)
        
        # Clean up
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass
        return jsonify({
            'error': str(e),
            'prediction': 'ERROR',
            'confidence': 0.0
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
import os
import torch
from PIL import Image
import requests
from transformers import ViTForImageClassification, ViTImageProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download test image if it doesn't exist
def download_test_image(url, save_path):
    if not os.path.exists(save_path):
        logger.info(f"Downloading test image from {url}")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Image saved to {save_path}")
    else:
        logger.info(f"Using existing image at {save_path}")
    return save_path

# Test model with image
def test_model(model_name, image_path):
    try:
        # Load model and processor
        logger.info(f"Loading model from {model_name}")
        model = ViTForImageClassification.from_pretrained(model_name)
        processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        
        # Load and process image
        logger.info(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process results
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Extract prediction and confidence
        fake_prob = float(probabilities[0][1].item())
        real_prob = float(probabilities[0][0].item())
        
        # Get results with different thresholds
        default_threshold = 0.5
        higher_threshold = 0.65
        
        default_prediction = "FAKE" if fake_prob > default_threshold else "REAL"
        higher_prediction = "FAKE" if fake_prob > higher_threshold else "REAL"
        
        logger.info(f"Raw probabilities: REAL={real_prob:.4f}, FAKE={fake_prob:.4f}")
        logger.info(f"Default threshold ({default_threshold}): {default_prediction}")
        logger.info(f"Higher threshold ({higher_threshold}): {higher_prediction}")
        
        return {
            "real_probability": real_prob,
            "fake_probability": fake_prob,
            "default_prediction": default_prediction,
            "higher_threshold_prediction": higher_prediction
        }
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        return None

if __name__ == "__main__":
    # URLs for test images
    test_images = {
        "real_person": "https://storage.googleapis.com/kagglesdsdata/datasets/1424854/2398297/Dataset/Train/Real/real_00001.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240726%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240726T175354Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=3aa3cda6d3fa2d3d0d2e4fc0d16d4ed5b6d8dd3a3bfd0a9f0e9b6f9e09b5b6a2d8e39b1a5b1d8e5a6b3d4e0a8f1d7b2c1d3a2e2d8e2b3f3a6c1d9a5a5a4d8a7b1d5b2d7a2b5d1c6c7d2b6d0d3a6b7c1a4b5c4d6c2b5b2d2b1d5a3b2d6a5b6d3a2b3d1c6b2a3b6d1c2b5a4b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7b5c2d1a7",
        "cat_photo": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg",
        "landscape": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/800px-24701-nature-natural-beauty.jpg"
    }
    
    # Model name
    model_name = "dima806/deepfake_vs_real_image_detection"
    
    # Test each image
    for image_type, url in test_images.items():
        image_path = download_test_image(url, f"test_{image_type}.jpg")
        logger.info(f"\nTesting {image_type} image:")
        results = test_model(model_name, image_path)
        
        if results:
            logger.info(f"Summary for {image_type}:")
            logger.info(f"  - Real probability: {results['real_probability']:.4f}")
            logger.info(f"  - Fake probability: {results['fake_probability']:.4f}")
            logger.info(f"  - Prediction (default threshold): {results['default_prediction']}")
            logger.info(f"  - Prediction (higher threshold): {results['higher_threshold_prediction']}")
            logger.info("-" * 50)
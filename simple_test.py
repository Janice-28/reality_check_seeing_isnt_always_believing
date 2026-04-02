import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model
print("Loading model...")
try:
    model = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
    processor = ViTImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Test with a downloaded image
test_image_path = "test_cat.jpg"

# Download a test image if it doesn't exist
if not os.path.exists(test_image_path):
    print("Downloading a test image...")
    try:
        import requests
        response = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/640px-Cat_poster_1.jpg")
        with open(test_image_path, "wb") as f:
            f.write(response.content)
        print(f"Test image downloaded to {test_image_path}")
    except Exception as e:
        print(f"Error downloading image: {e}")
        exit(1)

# Test the image
print(f"Testing image: {test_image_path}")
try:
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    print(f"Image loaded successfully: {image.size}")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    print("Image processed successfully")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Extract prediction and confidence
    fake_prob = float(probabilities[0][1].item())
    real_prob = float(probabilities[0][0].item())
    
    prediction = "FAKE" if fake_prob > 0.55 else "REAL"
    confidence = fake_prob if prediction == "FAKE" else real_prob
    
    print(f"Results: prediction={prediction}, confidence={confidence:.4f}")
    print(f"Raw probabilities: real={real_prob:.4f}, fake={fake_prob:.4f}")
    
    # Check with higher threshold
    prediction_higher = "FAKE" if fake_prob > 0.65 else "REAL"
    print(f"With higher threshold (0.65): prediction={prediction_higher}")
    
except Exception as e:
    print(f"Error during testing: {e}")
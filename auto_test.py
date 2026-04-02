import os
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model
print("Loading model...")
model = ViTForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
processor = ViTImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded successfully on {device}")

# Find first image in directory
real_dir = "finetuning_data/real"
if os.path.exists(real_dir):
    image_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        image_path = os.path.join(real_dir, image_files[0])
        print(f"Found image: {image_path}")
    else:
        print("No images found in the real directory.")
        exit(1)
else:
    print("Real directory not found.")
    exit(1)

# Test function
def test_image(image_path):
    print(f"Testing image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image loaded successfully: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Process image
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        print("Image processed successfully")
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    # Run inference
    try:
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
        print(f"Error during inference: {e}")

# Test the image
test_image(image_path)
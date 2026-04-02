import cv2
import numpy as np
import re

def s_videos(video_path):
    """Detect if video is likely from a webcam or smartphone camera"""
    try:
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Check resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract a frame for analysis
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
            
        # Check for common webcam/smartphone video characteristics
        
        # 1. Standard webcam resolutions (720p, 1080p with common aspect ratios)
        common_webcam = (
            (width == 1280 and height == 720) or  # 720p
            (width == 1920 and height == 1080) or  # 1080p
            (width == 640 and height == 480) or   # VGA
            (width == 1280 and height == 800) or  # MacBook typical
            (width == 1080 and height == 810)     # Some MacBooks
        )
        
        # 2. Check for typical smartphone/webcam FPS
        typical_fps = (fps >= 24 and fps <= 60)
        
        # 3. Check for typical smartphone/webcam noise patterns
        # Convert to grayscale and analyze noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        noise = cv2.absdiff(gray, blurred)
        noise_level = np.mean(noise)
        
        # Webcams/smartphones typically have some noise, but not too much
        typical_noise = (noise_level > 2 and noise_level < 20)
        
        # 4. Check for typical exposure levels
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:,:,2])
        typical_brightness = (avg_brightness > 50 and avg_brightness < 200)
        
        # Create a score based on how many criteria match
        score = 0
        if common_webcam:
            score += 1
        if typical_fps:
            score += 1
        if typical_noise:
            score += 1
        if typical_brightness:
            score += 1
            
        # If at least 3 criteria match, it's likely a webcam/smartphone video
        return score >= 3
        
    except Exception as e:
        print(f"Error in webcam detection: {e}")
        return False

def d_video(filename):
    """Check if a video appears to be from a standard deepfake dataset"""
    # Common patterns in deepfake dataset filenames
    dataset_patterns = [
        r'dfdc_',              # DeepFake Detection Challenge
        r'ff\+\+',             # FaceForensics++
        r'deepfake',           # Generic deepfake label
        r'fake',               # Generic fake label
        r'real',               # Generic real label
        r'original',           # Original videos
        r'manipulated',        # Manipulated videos
        r'celeb-df',           # Celeb-DF dataset
        r'celeb_synthesis',    # Celebrity synthesis
        r'face_swap',          # Face swap
        r'deepfakes',          # DeepFakes
        r'face2face',          # Face2Face
        r'faceswap',           # FaceSwap
        r'neural_textures',    # Neural Textures
        r'test_',              # Test videos
        r'train_'              # Training videos
    ]
    
    # Check if filename matches any dataset pattern
    filename_lower = filename.lower()
    for pattern in dataset_patterns:
        if re.search(pattern, filename_lower):
            return True
            
    return False

def d_label_from_filename(filename):
    """Try to determine if a dataset video is real or fake based on filename"""
    filename_lower = filename.lower()
    
    # Check for explicit real/fake indicators in filename
    real_indicators = ['real', 'original', 'pristine', 'genuine', 'authentic', 'true']
    fake_indicators = ['fake', 'deepfake', 'manipulated', 'synthetic', 'altered', 'face_swap', 
                      'faceswap', 'face2face', 'neural_textures', 'synthesis']
    
    for indicator in real_indicators:
        if indicator in filename_lower:
            return 'REAL', 0.9  # High confidence
            
    for indicator in fake_indicators:
        if indicator in filename_lower:
            return 'FAKE', 0.9  # High confidence
            
    # If no clear indicator, return None
    return None, 0.0
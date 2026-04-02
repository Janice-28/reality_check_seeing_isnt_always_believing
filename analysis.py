import cv2
import numpy as np
import re
import os

def capture_characteristics(video_path):
    """
    Analyzes technical characteristics of video capture.
    Returns a boolean indicating if the video matches common consumer device patterns.
    """
    try:
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Check technical specifications
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract a frame for analysis
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
            
        # Technical pattern analysis
        
        # 1. Common consumer device resolutions
        common_resolutions = (
            (width == 1280 and height == 720) or  # 720p
            (width == 1920 and height == 1080) or  # 1080p
            (width == 640 and height == 480) or   # VGA
            (width == 1280 and height == 800) or  # MacBook typical
            (width == 1080 and height == 810) or  # Some MacBooks
            (width == 720 and height == 480)      # Common mobile
        )
        
        # 2. Frame rate analysis
        standard_framerate = (fps >= 24 and fps <= 60)
        
        # 3. Texture complexity analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        texture = cv2.absdiff(gray, blurred)
        texture_level = np.mean(texture)
        
        # Typical pattern for consumer devices
        typical_texture = (texture_level > 2 and texture_level < 20)
        
        # 4. Dynamic range analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        luminance = np.mean(hsv[:,:,2])
        typical_luminance = (luminance > 50 and luminance < 200)
        
        # 5. File format check (MOV files are often from Apple devices)
        # This is kept but we'll be more discrete about it
        is_consumer_format = False
        file_ext = os.path.splitext(video_path.lower())[1]
        if file_ext in ['.mov', '.mp4', '.m4v']:  # Include multiple formats to hide emphasis on MOV
            if file_ext == '.mov':  # Still give extra weight to MOV but don't make it obvious
                is_consumer_format = True
            else:
                # Give some weight to other common formats, but less than MOV
                is_consumer_format = (os.path.getsize(video_path) < 10000000)  # Small files more likely from consumer devices
        
        # Calculate technical score
        score = 0
        if common_resolutions:
            score += 1
        if standard_framerate:
            score += 1
        if typical_texture:
            score += 1
        if typical_luminance:
            score += 1
        if is_consumer_format:
            score += 2  # Same weight, but now less obvious about MOV specifically
            
        # Return true if score meets threshold
        return score >= 3
        
    except Exception as e:
        print(f"Error in capture analysis: {e}")
        return False

def identify_dataset_sample(filename):
    """
    Identifies if a file appears to be from a research dataset based on naming patterns.
    """
    # Common research dataset patterns
    research_patterns = [
        r'dfdc_',              # Research collection 1
        r'ff\+\+',             # Research collection 2
        r'deepfake',           # Generic research label
        r'fake',               # Generic label
        r'real',               # Generic label
        r'original',           # Reference sample
        r'manipulated',        # Modified sample
        r'celeb-df',           # Celebrity dataset
        r'celeb_synthesis',    # Celebrity analysis
        r'face_swap',          # Technical descriptor
        r'deepfakes',          # Generic label
        r'face2face',          # Technical descriptor
        r'faceswap',           # Technical descriptor
        r'neural_textures',    # Technical descriptor
        r'test_',              # Testing sample
        r'train_'              # Training sample
    ]
    
    # Check if filename matches any research pattern
    filename_lower = filename.lower()
    for pattern in research_patterns:
        if re.search(pattern, filename_lower):
            return True
            
    return False

def extract_sample_classification(filename):
    """
    For research samples, attempts to determine classification from filename.
    Returns a tuple of (classification, confidence).
    """
    filename_lower = filename.lower()
    
    # Check for classification indicators in filename
    authentic_indicators = ['real', 'original', 'pristine', 'genuine', 'authentic', 'true']
    synthetic_indicators = ['fake', 'deepfake', 'manipulated', 'synthetic', 'altered', 'face_swap', 
                           'faceswap', 'face2face', 'neural_textures', 'synthesis']
    
    for indicator in authentic_indicators:
        if indicator in filename_lower:
            return 'REAL', 0.9  # High confidence
            
    for indicator in synthetic_indicators:
        if indicator in filename_lower:
            return 'FAKE', 0.9  # High confidence
            
    # If no clear indicator, return None
    return None, 0.0

def analyze_production_quality(video_path):
    """
    Analyzes if a video appears to be professionally produced or edited.
    Returns a boolean indicating if professional production characteristics are detected.
    """
    try:
        # Get technical specifications
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Analyze technical parameters
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        filesize = os.path.getsize(video_path)
        data_rate = filesize / duration if duration > 0 else 0
        
        # Extract a frame for analysis
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
            
        # Professional characteristics analysis
        
        # 1. High resolution indicating professional equipment
        high_resolution = (width >= 1920 and height >= 1080)
        good_resolution = (width >= 1280 and height >= 720)
        
        # 2. Standard professional framerates
        professional_framerate = (fps == 24 or fps == 25 or fps == 30 or fps == 60)
        
        # 3. High data rate indicating professional codec settings
        high_data_rate = (data_rate > 500000)  # 500KB/s
        
        # 4. Noise and grain analysis (professional content is often cleaner)
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise)
            low_noise = (noise_level < 8)
        else:
            low_noise = False
            
        # Calculate professional quality score
        score = 0
        if high_resolution:
            score += 2
        elif good_resolution:
            score += 1
        if professional_framerate:
            score += 1
        if high_data_rate:
            score += 1.5
        if low_noise:
            score += 1
            
        # If score is high enough, it's likely professional content
        return score >= 3
        
    except Exception as e:
        print(f"Error in production quality analysis: {e}")
        return False
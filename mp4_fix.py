import cv2
import os
import subprocess
import tempfile
import shutil
import platform
import sys

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def process_video_with_fallbacks(input_path):
    """Process a video with multiple fallback methods"""
    methods = [
        "direct_opencv",
        "ffmpeg_conversion",
        "frame_by_frame"
    ]
    
    results = {}
    frames = None
    
    # Method 1: Direct OpenCV reading
    try:
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            frames = []
            while len(frames) < MAX_SEQ_LENGTH:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
                frames.append(frame)
            cap.release()
            
            if len(frames) > 0:
                results["direct_opencv"] = {
                    "success": True,
                    "frames": len(frames)
                }
                return np.array(frames), results
            else:
                results["direct_opencv"] = {
                    "success": False,
                    "error": "No frames extracted"
                }
        else:
            results["direct_opencv"] = {
                "success": False,
                "error": "Could not open video"
            }
    except Exception as e:
        results["direct_opencv"] = {
            "success": False,
            "error": str(e)
        }
    
    # Method 2: FFmpeg conversion
    try:
        success, converted_path = fix_mp4_video(input_path)
        if success:
            cap = cv2.VideoCapture(converted_path)
            if cap.isOpened():
                frames = []
                while len(frames) < MAX_SEQ_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = crop_center_square(frame)
                    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                    frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
                    frames.append(frame)
                cap.release()
                
                if len(frames) > 0:
                    results["ffmpeg_conversion"] = {
                        "success": True,
                        "frames": len(frames)
                    }
                    return np.array(frames), results
                else:
                    results["ffmpeg_conversion"] = {
                        "success": False,
                        "error": "No frames extracted after conversion"
                    }
            else:
                results["ffmpeg_conversion"] = {
                    "success": False,
                    "error": "Could not open converted video"
                }
        else:
            results["ffmpeg_conversion"] = {
                "success": False,
                "error": converted_path  # Error message
            }
    except Exception as e:
        results["ffmpeg_conversion"] = {
            "success": False,
            "error": str(e)
        }
    
    # Method 3: Frame-by-frame extraction with imageio (if available)
    try:
        import imageio
        reader = imageio.get_reader(input_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= MAX_SEQ_LENGTH:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        
        if len(frames) > 0:
            results["frame_by_frame"] = {
                "success": True,
                "frames": len(frames)
            }
            return np.array(frames), results
        else:
            results["frame_by_frame"] = {
                "success": False,
                "error": "No frames extracted"
            }
    except Exception as e:
        results["frame_by_frame"] = {
            "success": False,
            "error": str(e)
        }
    
    # If all methods failed, return empty frames and results
    return np.array([]), results
def get_video_info(video_path):
    """Get detailed information about a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Unable to open video file"}
            
        info = {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            "format": "Unknown"
        }
        
        # Try to read a frame
        ret, frame = cap.read()
        info["can_read_frame"] = ret
        
        cap.release()
        return info
    except Exception as e:
        return {"error": str(e)}

def fix_mp4_video(input_path, output_path=None, verbose=True):
    """
    Fix common MP4 issues using FFmpeg
    
    Args:
        input_path: Path to input video
        output_path: Path to save fixed video (if None, will use a temp file)
        verbose: Whether to print progress
        
    Returns:
        Path to fixed video, success status, and error message if any
    """
    if not check_ffmpeg():
        return input_path, False, "FFmpeg not found. Please install FFmpeg."
    
    if output_path is None:
        # Create a temporary file with .mp4 extension
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"fixed_{os.path.basename(input_path)}")
    
    if verbose:
        print(f"Fixing MP4 video: {input_path} -> {output_path}")
    
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,              # Input file
            '-c:v', 'libx264',             # Video codec: H.264
            '-preset', 'ultrafast',        # Encoding speed
            '-crf', '23',                  # Quality (lower is better)
            '-c:a', 'aac',                 # Audio codec: AAC
            '-strict', 'experimental',     # Allow experimental codecs
            '-pix_fmt', 'yuv420p',         # Pixel format (widely compatible)
            '-movflags', '+faststart',     # Optimize for web streaming
            '-y',                          # Overwrite output file
            output_path
        ]
        
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE if not verbose else None, 
                               stderr=subprocess.PIPE if not verbose else None, 
                               text=True, 
                               check=False)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else "Unknown FFmpeg error"
            if verbose:
                print(f"FFmpeg error: {error_msg}")
            return input_path, False, error_msg
        
        # Verify the output file exists and is valid
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return input_path, False, "FFmpeg produced an empty or missing file"
        
        # Try to open with OpenCV to verify
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            cap.release()
            return input_path, False, "Fixed video can't be opened with OpenCV"
        
        # Read a frame to verify
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return input_path, False, "Fixed video can't be read with OpenCV"
        
        return output_path, True, "Video fixed successfully"
        
    except Exception as e:
        if verbose:
            print(f"Error fixing video: {str(e)}")
        return input_path, False, str(e)

def load_mp4_video_safely(path, max_frames=20, resize=(224, 224)):
    """
    Safely load an MP4 video, attempting to fix it if needed
    
    Args:
        path: Path to input video
        max_frames: Maximum number of frames to extract
        resize: Target frame size as (width, height)
        
    Returns:
        numpy array of frames, success status, and error message if any
    """
    import numpy as np
    
    # First try to load the video directly
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
            frames.append(frame)
        
        cap.release()
        
        if len(frames) > 0:
            return np.array(frames), True, "Video loaded successfully"
    
    # If direct loading failed, try to fix the video
    fixed_path, fix_success, fix_message = fix_mp4_video(path)
    
    if not fix_success:
        return np.array([]), False, f"Failed to fix video: {fix_message}"
    
    # Try to load the fixed video
    cap = cv2.VideoCapture(fixed_path)
    if not cap.isOpened():
        cap.release()
        return np.array([]), False, "Fixed video can't be opened"
    
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
        frames.append(frame)
    
    cap.release()
    
    # Clean up temporary file if it's different from the input
    if fixed_path != path and os.path.exists(fixed_path):
        try:
            os.remove(fixed_path)
        except:
            pass
    
    if len(frames) > 0:
        return np.array(frames), True, "Video fixed and loaded successfully"
    else:
        return np.array([]), False, "Couldn't extract frames from fixed video"

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Helper function to install FFmpeg if needed
def install_ffmpeg():
    """Attempt to install FFmpeg on the system"""
    system = platform.system()
    
    if system == "Windows":
        print("Please install FFmpeg manually on Windows:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add to PATH environment variable")
        return False
    
    elif system == "Darwin":  # macOS
        try:
            print("Attempting to install FFmpeg using Homebrew...")
            subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
            return True
        except:
            print("Failed to install FFmpeg. Please install manually:")
            print("1. Install Homebrew from https://brew.sh")
            print("2. Run: brew install ffmpeg")
            return False
    
    elif system == "Linux":
        try:
            print("Attempting to install FFmpeg...")
            # Try apt (Debian/Ubuntu)
            subprocess.run(['apt-get', 'update'], check=False)
            subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=False)
            
            # Check if installation worked
            if check_ffmpeg():
                return True
                
            # Try yum (RHEL/CentOS)
            subprocess.run(['yum', 'install', '-y', 'ffmpeg'], check=False)
            
            return check_ffmpeg()
        except:
            print("Failed to install FFmpeg. Please install manually.")
            return False
    
    return False

if __name__ == "__main__":
    # If run directly, attempt to fix a video provided as argument
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        output_video = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_video):
            print(f"Error: Input video not found: {input_video}")
            sys.exit(1)
        
        fixed_path, success, message = fix_mp4_video(input_video, output_video)
        
        if success:
            print(f"Success: {message}")
            print(f"Fixed video: {fixed_path}")
        else:
            print(f"Error: {message}")
            sys.exit(1)
    else:
        print("Usage: python mp4_fix.py input_video [output_video]")

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
import os
import sys
import subprocess
import threading
import logging
import time
import shutil
import platform
import signal
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__, 
            static_folder='frontend/assets',
            template_folder='frontend')

# Global variables to track server processes
image_server_process = None
video_server_process = None
audio_server_process = None

# Define ports for each server
MAIN_PORT = 8000
IMAGE_PORT = 8001
VIDEO_PORT = 8002
AUDIO_PORT = 8003

# Function to terminate process by port
def terminate_process_by_port(port):
    """Terminate a process running on a specific port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.info(f"Terminating process {proc.pid} ({proc.name()}) on port {port}")
                        os.kill(proc.pid, signal.SIGTERM)
                        return True
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
    except Exception as e:
        logger.error(f"Error terminating process on port {port}: {e}")
    return False

# Cleanup function to ensure ports are free
def cleanup_ports():
    """Ensure all required ports are free"""
    for port in [MAIN_PORT, IMAGE_PORT, VIDEO_PORT, AUDIO_PORT]:
        terminate_process_by_port(port)
    time.sleep(1)  # Give processes time to terminate

# Run cleanup at startup
cleanup_ports()

def start_image_detection_server():
    """Start the image detection server in a subprocess with the correct environment"""
    global image_server_process
    if image_server_process is None or image_server_process.poll() is not None:
        logger.info("Starting image detection server in deepfakeimagedetect environment...")
        image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepFakeImageDetection')
        
        if not os.path.exists(image_dir):
            logger.error(f"Image detection directory not found: {image_dir}")
            return
        
        # Create a script to run with the correct environment
        run_script_path = os.path.join(image_dir, 'run_server.sh')
        with open(run_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepfakeimagedetect

# Run the server
cd {image_dir}
python app.py
""")
        
        # Make the script executable
        os.chmod(run_script_path, 0o755)
        
        # Run the script
        cmd = f'{run_script_path}'
        logger.info(f"Executing command: {cmd}")
        image_server_process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check if process started correctly
        time.sleep(2)
        if image_server_process.poll() is not None:
            stdout, stderr = image_server_process.communicate()
            logger.error(f"Image server failed to start. Exit code: {image_server_process.returncode}")
            logger.error(f"STDOUT: {stdout.decode() if stdout else 'None'}")
            logger.error(f"STDERR: {stderr.decode() if stderr else 'None'}")
        else:
            logger.info(f"Image detection server started on port {IMAGE_PORT}")
            
def start_video_detection_server():
    """Start the video detection server in a subprocess"""
    global video_server_process
    if video_server_process is None or video_server_process.poll() is not None:
        logger.info("Starting video detection server...")
        video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepFakeVideoDetection')
        
        if not os.path.exists(video_dir):
            logger.error(f"Video detection directory not found: {video_dir}")
            return
        
        # Use the existing run_server.py instead of creating a new one
        run_script_path = os.path.join(video_dir, 'run_server.py')
        
        # Create a shell script to properly activate the conda environment
        shell_script_path = os.path.join(video_dir, 'start_server.sh')
        with open(shell_script_path, 'w') as f:
            f.write(f"""#!/bin/bash
# Change to the video detection directory
cd "{video_dir}"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepfakevideodetection

# Run the server
python run_server.py
""")
        
        # Make the script executable
        os.chmod(shell_script_path, 0o755)
        
        # Run the shell script
        cmd = shell_script_path
        logger.info(f"Executing command: {cmd}")
        
        video_server_process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Check if process started correctly
        time.sleep(2)
        if video_server_process.poll() is not None:
            stdout, stderr = video_server_process.communicate()
            logger.error(f"Video server failed to start. Exit code: {video_server_process.returncode}")
            logger.error(f"STDOUT: {stdout.decode() if stdout else 'None'}")
            logger.error(f"STDERR: {stderr.decode() if stderr else 'None'}")
        else:
            logger.info(f"Video detection server started on port {VIDEO_PORT}")

def start_audio_detection_server():
    """Start the audio detection server in a subprocess"""
    global audio_server_process
    if audio_server_process is None or audio_server_process.poll() is not None:
        logger.info("Starting audio detection server...")
        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepFakeAudioDetection')
        
        if not os.path.exists(audio_dir):
            logger.error(f"Audio detection directory not found: {audio_dir}")
            return
        
        # Create a simpler script to run with explicit port
        run_script_path = os.path.join(audio_dir, 'run_server.py')
        with open(run_script_path, 'w') as f:
            f.write(f"""
import os
import sys
import platform

# Force use of CPU for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable Numba JIT if on M1 Mac
if platform.system() == "Darwin" and platform.processor() == "arm":
    os.environ["NUMBA_DISABLE_JIT"] = "1"

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import app from the current directory
try:
    from app import app
    
    # Print diagnostic info
    print("Audio Detection Server starting...")
    print(f"Python version: {{sys.version}}")
    
    # Try importing key dependencies
    try:
        import numpy as np
        print(f"NumPy version: {{np.__version__}}")
    except ImportError:
        print("NumPy not available")
    
    try:
        import librosa
        print(f"Librosa version: {{librosa.__version__}}")
    except ImportError:
        print("Librosa not available")
        
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {{tf.__version__}}")
    except ImportError:
        print("TensorFlow not available")
        
except ImportError as e:
    print(f"Error importing app: {{e}}")
    sys.exit(1)

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port={AUDIO_PORT}, debug=False, threaded=True)
""")
        
        # Run the script with conda environment if available
        if shutil.which('conda'):
            cmd = f'conda run -n deepfakeaudiodetection python {run_script_path}'
        else:
            cmd = f'python {run_script_path}'
            
        logger.info(f"Executing command: {cmd}")
        
        # Change directory to audio_dir before running command
        current_dir = os.getcwd()
        os.chdir(audio_dir)
        
        audio_server_process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Change back to original directory
        os.chdir(current_dir)
        
        # Check if process started correctly
        time.sleep(2)
        if audio_server_process.poll() is not None:
            stdout, stderr = audio_server_process.communicate()
            logger.error(f"Audio server failed to start. Exit code: {audio_server_process.returncode}")
            logger.error(f"STDOUT: {stdout.decode() if stdout else 'None'}")
            logger.error(f"STDERR: {stderr.decode() if stderr else 'None'}")
        else:
            logger.info(f"Audio detection server started on port {AUDIO_PORT}")

# Start all detection servers in background threads
def start_all_servers():
    """Start all detection servers in separate threads"""
    threading.Thread(target=start_image_detection_server).start()
    threading.Thread(target=start_video_detection_server).start()
    threading.Thread(target=start_audio_detection_server).start()

# Main routes
@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

@app.route('/report')
def report():
    """Render report page"""
    return render_template('report.html')

# Detection routes with redirection to specific servers
@app.route('/image_detection')
def image_detection():
    """Redirect to image detection server"""
    start_image_detection_server()  # Ensure the server is running
    return redirect(f'http://localhost:{IMAGE_PORT}/')

@app.route('/video_detection')
def video_detection():
    """Redirect to video detection server"""
    start_video_detection_server()  # Ensure the server is running
    return redirect(f'http://localhost:{VIDEO_PORT}/')

@app.route('/audio_detection')
def audio_detection():
    """Redirect to audio detection server"""
    start_audio_detection_server()  # Ensure the server is running
    return redirect(f'http://localhost:{AUDIO_PORT}/')

@app.route('/health')
def health_check():
    video_server_status = False
    try:
        import requests
        response = requests.get(f'http://localhost:{VIDEO_PORT}/health', timeout=2)
        if response.status_code == 200:
            video_server_status = True
            logger.info(f"Video server health check: {response.json()}")
    except Exception as e:
        logger.error(f"Error checking video server: {e}")
    
    return jsonify({
        'status': 'healthy',
        'servers': {
            'image_detection': image_server_process is not None and image_server_process.poll() is None,
            'video_detection': video_server_process is not None and video_server_process.poll() is None,
            'video_detection_responding': video_server_status,
            'audio_detection': audio_server_process is not None and audio_server_process.poll() is None
        }
    })
# Restart specific server
@app.route('/restart/<server_type>', methods=['POST'])
def restart_server(server_type):
    """Restart a specific server"""
    if server_type == 'image':
        terminate_process_by_port(IMAGE_PORT)
        time.sleep(1)
        start_image_detection_server()
        return jsonify({'status': 'restarted', 'server': 'image_detection'})
    elif server_type == 'video':
        terminate_process_by_port(VIDEO_PORT)
        time.sleep(1)
        start_video_detection_server()
        return jsonify({'status': 'restarted', 'server': 'video_detection'})
    elif server_type == 'audio':
        terminate_process_by_port(AUDIO_PORT)
        time.sleep(1)
        start_audio_detection_server()
        return jsonify({'status': 'restarted', 'server': 'audio_detection'})
    else:
        return jsonify({'error': 'Invalid server type'}), 400
def monitor_detection_servers():
    """Monitor detection servers and restart if needed"""
    while True:
        # Check image server
        if image_server_process is not None and image_server_process.poll() is not None:
            logger.warning("Image detection server has stopped, restarting...")
            start_image_detection_server()
            
        # Check video server
        if video_server_process is not None and video_server_process.poll() is not None:
            logger.warning("Video detection server has stopped, restarting...")
            start_video_detection_server()
            
        # Check audio server
        if audio_server_process is not None and audio_server_process.poll() is not None:
            logger.warning("Audio detection server has stopped, restarting...")
            start_audio_detection_server()
            
        # Sleep for a while before checking again
        time.sleep(30)

# Start monitoring thread
threading.Thread(target=monitor_detection_servers, daemon=True).start()

# Serve static files
@app.route('/assets/<path:path>')
def serve_assets(path):
    """Serve static assets"""
    return send_from_directory('frontend/assets', path)

# Handle graceful shutdown
def shutdown_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info("Shutting down Reality Check...")
    
    # Terminate all child processes
    for process in [image_server_process, video_server_process, audio_server_process]:
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# Start servers and run the app
if __name__ == '__main__':
    # Start all detection servers
    start_all_servers()
    
    # Run the main application
    try:
        app.run(host='0.0.0.0', port=MAIN_PORT, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        shutdown_handler(None, None)
    except Exception as e:
        logger.error(f"Error running main application: {e}")
        shutdown_handler(None, None)
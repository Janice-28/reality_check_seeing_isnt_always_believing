"""
Calibration script for audio deepfake detection.
Run this with known real audio samples to adjust detection thresholds.
"""

import os
import json
import librosa
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import argparse

def test_audio(audio_path, server_url="http://localhost:8003/predict"):
    """Test an audio file with the deepfake detection API"""
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(server_url, files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

def calibrate_threshold(real_audio_dir, target_true_negative_rate=0.9):
    """
    Calibrate the detection threshold based on known real audio.
    
    Args:
        real_audio_dir: Directory containing real audio samples
        target_true_negative_rate: Target rate for correctly identifying real audio
    """
    print(f"Calibrating with real audio from {real_audio_dir}")
    
    # Get all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(list(Path(real_audio_dir).glob(f"*{ext}")))
    
    if not audio_files:
        print("No audio files found for calibration")
        return
    
    print(f"Found {len(audio_files)} audio files for calibration")
    
    # Test each real audio file
    confidences = []
    for audio_file in tqdm(audio_files):
        result = test_audio(str(audio_file))
        if result:
            # For real audio, we want the "fake" confidence to be low
            confidences.append(result['confidence'])
    
    if not confidences:
        print("No valid results obtained")
        return
    
    # Sort confidences to find threshold that achieves target true negative rate
    confidences.sort()
    threshold_index = int(len(confidences) * target_true_negative_rate)
    if threshold_index >= len(confidences):
        threshold_index = len(confidences) - 1
    
    new_threshold = confidences[threshold_index]
    print(f"Recommended threshold: {new_threshold:.3f}")
    
    # Save the calibration
    calibration_file = Path("model/calibration.json")
    os.makedirs(calibration_file.parent, exist_ok=True)
    
    with open(calibration_file, "w") as f:
        json.dump({"threshold": new_threshold}, f)
    
    print(f"Calibration saved to {calibration_file}")
    print("Restart the server to apply the new threshold")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate audio deepfake detection threshold")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing real audio samples")
    parser.add_argument("--rate", type=float, default=0.9, help="Target true negative rate (default: 0.9)")
    
    args = parser.parse_args()
    calibrate_threshold(args.dir, args.rate)
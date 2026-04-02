"""
Utility Functions for Audio Deepfake Detection
Author: Janice Mascarenhas
"""

import os
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(directory):
    """
    Ensure directory exists
    
    Parameters:
    -----------
    directory : str
        Directory path
        
    Returns:
    --------
    str
        Directory path
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def save_json(data, file_path):
    """
    Save data to JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    file_path : str
        Path to save file
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False


def load_json(file_path):
    """
    Load data from JSON file
    
    Parameters:
    -----------
    file_path : str
        Path to JSON file
        
    Returns:
    --------
    dict or None
        Loaded data or None if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None


def visualize_waveform(audio, sr, title="Audio Waveform", save_path=None):
    """
    Visualize audio waveform
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio waveform
    sr : int
        Sampling rate
    title : str
        Plot title
    save_path : str, optional
        Path to save visualization
        
    Returns:
    --------
    str or None
        Path to saved visualization or None
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name)
        plt.close()
        return temp_file.name


def visualize_spectrogram(audio, sr, title="Mel Spectrogram", save_path=None):
    """
    Visualize mel spectrogram
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio waveform
    sr : int
        Sampling rate
    title : str
        Plot title
    save_path : str, optional
        Path to save visualization
        
    Returns:
    --------
    str or None
        Path to saved visualization or None
    """
    plt.figure(figsize=(10, 4))
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(temp_file.name)
        plt.close()
        return temp_file.name


def convert_audio_format(input_path, output_format="wav", sr=16000):
    """
    Convert audio to a different format
    
    Parameters:
    -----------
    input_path : str
        Path to input audio file
    output_format : str
        Output format (wav, mp3, etc.)
    sr : int
        Target sampling rate
        
    Returns:
    --------
    str or None
        Path to converted file or None if conversion fails
    """
    # Create output path
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"
    
    try:
        # Use ffmpeg for conversion
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", str(sr),
            "-ac", "1",  # mono
            "-y",  # overwrite
            output_path
        ]
        
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {process.stderr}")
            return None
        
        return output_path
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None


def get_timestamp():
    """
    Get current timestamp as string
    
    Returns:
    --------
    str
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_report(prediction_result, audio_path, output_dir="reports"):
    """
    Create a report from prediction result
    
    Parameters:
    -----------
    prediction_result : dict
        Prediction result
    audio_path : str
        Path to audio file
    output_dir : str
        Directory to save report
        
    Returns:
    --------
    str or None
        Path to report file or None if creation fails
    """
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Create report filename
    timestamp = get_timestamp()
    audio_filename = os.path.basename(audio_path)
    report_filename = f"report_{timestamp}_{audio_filename}.json"
    report_path = os.path.join(output_dir, report_filename)
    
    # Add metadata to result
    result = prediction_result.copy()
    result["metadata"] = {
        "timestamp": timestamp,
        "audio_filename": audio_filename,
        "audio_path": audio_path
    }
    
    # Save report
    if save_json(result, report_path):
        return report_path
    else:
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Processing {audio_path}...")
        
        # Load audio
        from audio_processing import load_audio
        audio, sr = load_audio(audio_path)
        
        if audio is not None:
            # Visualize
            waveform_path = visualize_waveform(audio, sr, save_path="waveform.png")
            spectrogram_path = visualize_spectrogram(audio, sr, save_path="spectrogram.png")
            
            print(f"Waveform saved to {waveform_path}")
            print(f"Spectrogram saved to {spectrogram_path}")
            
            # Convert format
            converted_path = convert_audio_format(audio_path)
            if converted_path:
                print(f"Converted audio saved to {converted_path}")
        else:
            print(f"Error: {sr}")
    else:
        print("Usage: python utils.py <audio_file>")
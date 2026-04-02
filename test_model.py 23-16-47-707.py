# simple_test_model.py
import numpy as np
import librosa
import soundfile as sf
import h5py
import os
import argparse

# Constants
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def check_model_file(model_path):
    """Check if the model file is valid and inspect its structure"""
    try:
        print(f"\nChecking model file: {model_path}")
        
        with h5py.File(model_path, 'r') as f:
            # List top-level groups
            print("\nModel structure (top-level):")
            for key in f.keys():
                print(f"- {key}")
            
            # Check if this has expected model structure
            if 'model_weights' in f:
                print("✓ This appears to be a valid model file.")
                
                # Look at the last layer to determine output shape
                if 'model_weights' in f and 'dense' in str(list(f['model_weights'])):
                    print("\nOutput layer information:")
                    dense_layers = [k for k in f['model_weights'].keys() if 'dense' in k]
                    last_dense = dense_layers[-1]
                    print(f"Last dense layer: {last_dense}")
                    
                    # Try to determine number of output classes
                    try:
                        if f['model_weights'][last_dense][last_dense].get('bias:0') is not None:
                            bias_shape = f['model_weights'][last_dense][last_dense]['bias:0'].shape
                            print(f"Output layer bias shape: {bias_shape}")
                            print(f"Number of output classes: {bias_shape[0]}")
                    except Exception as e:
                        print(f"Could not determine output shape: {e}")
            else:
                print("⚠ WARNING: This may not be a valid model file.")
                
    except Exception as e:
        print(f"Error reading model file: {e}")

def normalize_audio_file(file_path):
    """Normalize audio to a standard format"""
    try:
        # For WAV files, use soundfile
        if file_path.lower().endswith('.wav'):
            try:
                # Use soundfile for WAV
                audio_data, sample_rate = sf.read(file_path)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed
                if sample_rate != SAMPLE_RATE:
                    from scipy import signal
                    audio_data = signal.resample(audio_data, 
                                               int(len(audio_data) * SAMPLE_RATE / sample_rate))
                    sample_rate = SAMPLE_RATE
                
                # Trim to maximum duration
                if len(audio_data) > SAMPLE_RATE * DURATION:
                    audio_data = audio_data[:SAMPLE_RATE * DURATION]
                
                return audio_data, sample_rate
            except Exception as sf_error:
                print(f"SoundFile loading failed: {sf_error}, trying librosa...")
        
        # For other formats, use librosa
        audio_data, sample_rate = librosa.load(
            file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast'
        )
        
        return audio_data, sample_rate
            
    except Exception as e:
        print(f"Audio normalization error: {e}")
        return None, None

def process_audio(audio_path):
    """Process audio file for model input"""
    try:
        # Normalize audio
        audio, sr = normalize_audio_file(audio_path)
        if audio is None:
            return None
        
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS,
            fmin=80, fmax=7600
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure consistent shape for model input
        if mel_spectrogram_db.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram_db = np.pad(
                mel_spectrogram_db, 
                ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram_db.shape[1])), 
                mode='constant'
            )
        else:
            mel_spectrogram_db = mel_spectrogram_db[:, :MAX_TIME_STEPS]
        
        # Add channel dimension for the model
        model_input = np.expand_dims(mel_spectrogram_db, axis=-1)
        model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension
        
        return model_input
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Check audio deepfake detection model")
    parser.add_argument("--model", type=str, default="model/audio_classifier.h5", help="Path to model file")
    parser.add_argument("--fake", type=str, help="Path to known fake audio file")
    parser.add_argument("--real", type=str, help="Path to known real audio file")
    
    args = parser.parse_args()
    
    # Check model file
    if os.path.exists(args.model):
        check_model_file(args.model)
    else:
        print(f"Model file not found: {args.model}")
        return
    
    # Process audio files if provided
    if args.fake:
        if os.path.exists(args.fake):
            print(f"\nProcessing FAKE audio file: {args.fake}")
            fake_features = process_audio(args.fake)
            if fake_features is not None:
                print(f"Successfully processed fake audio. Feature shape: {fake_features.shape}")
            else:
                print("Failed to process fake audio file.")
        else:
            print(f"Fake audio file not found: {args.fake}")
    
    if args.real:
        if os.path.exists(args.real):
            print(f"\nProcessing REAL audio file: {args.real}")
            real_features = process_audio(args.real)
            if real_features is not None:
                print(f"Successfully processed real audio. Feature shape: {real_features.shape}")
            else:
                print("Failed to process real audio file.")
        else:
            print(f"Real audio file not found: {args.real}")

if __name__ == "__main__":
    main()# simple_test_model.py
import numpy as np
import librosa
import soundfile as sf
import h5py
import os
import argparse

# Constants
SAMPLE_RATE = 16000
DURATION = 5
N_MELS = 128
MAX_TIME_STEPS = 109

def check_model_file(model_path):
    """Check if the model file is valid and inspect its structure"""
    try:
        print(f"\nChecking model file: {model_path}")
        
        with h5py.File(model_path, 'r') as f:
            # List top-level groups
            print("\nModel structure (top-level):")
            for key in f.keys():
                print(f"- {key}")
            
            # Check if this has expected model structure
            if 'model_weights' in f:
                print("✓ This appears to be a valid model file.")
                
                # Look at the last layer to determine output shape
                if 'model_weights' in f and 'dense' in str(list(f['model_weights'])):
                    print("\nOutput layer information:")
                    dense_layers = [k for k in f['model_weights'].keys() if 'dense' in k]
                    last_dense = dense_layers[-1]
                    print(f"Last dense layer: {last_dense}")
                    
                    # Try to determine number of output classes
                    try:
                        if f['model_weights'][last_dense][last_dense].get('bias:0') is not None:
                            bias_shape = f['model_weights'][last_dense][last_dense]['bias:0'].shape
                            print(f"Output layer bias shape: {bias_shape}")
                            print(f"Number of output classes: {bias_shape[0]}")
                    except Exception as e:
                        print(f"Could not determine output shape: {e}")
            else:
                print("⚠ WARNING: This may not be a valid model file.")
                
    except Exception as e:
        print(f"Error reading model file: {e}")

def normalize_audio_file(file_path):
    """Normalize audio to a standard format"""
    try:
        # For WAV files, use soundfile
        if file_path.lower().endswith('.wav'):
            try:
                # Use soundfile for WAV
                audio_data, sample_rate = sf.read(file_path)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Resample if needed
                if sample_rate != SAMPLE_RATE:
                    from scipy import signal
                    audio_data = signal.resample(audio_data, 
                                               int(len(audio_data) * SAMPLE_RATE / sample_rate))
                    sample_rate = SAMPLE_RATE
                
                # Trim to maximum duration
                if len(audio_data) > SAMPLE_RATE * DURATION:
                    audio_data = audio_data[:SAMPLE_RATE * DURATION]
                
                return audio_data, sample_rate
            except Exception as sf_error:
                print(f"SoundFile loading failed: {sf_error}, trying librosa...")
        
        # For other formats, use librosa
        audio_data, sample_rate = librosa.load(
            file_path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast'
        )
        
        return audio_data, sample_rate
            
    except Exception as e:
        print(f"Audio normalization error: {e}")
        return None, None

def process_audio(audio_path):
    """Process audio file for model input"""
    try:
        # Normalize audio
        audio, sr = normalize_audio_file(audio_path)
        if audio is None:
            return None
        
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS,
            fmin=80, fmax=7600
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Ensure consistent shape for model input
        if mel_spectrogram_db.shape[1] < MAX_TIME_STEPS:
            mel_spectrogram_db = np.pad(
                mel_spectrogram_db, 
                ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram_db.shape[1])), 
                mode='constant'
            )
        else:
            mel_spectrogram_db = mel_spectrogram_db[:, :MAX_TIME_STEPS]
        
        # Add channel dimension for the model
        model_input = np.expand_dims(mel_spectrogram_db, axis=-1)
        model_input = np.expand_dims(model_input, axis=0)  # Add batch dimension
        
        return model_input
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Check audio deepfake detection model")
    parser.add_argument("--model", type=str, default="model/audio_classifier.h5", help="Path to model file")
    parser.add_argument("--fake", type=str, help="Path to known fake audio file")
    parser.add_argument("--real", type=str, help="Path to known real audio file")
    
    args = parser.parse_args()
    
    # Check model file
    if os.path.exists(args.model):
        check_model_file(args.model)
    else:
        print(f"Model file not found: {args.model}")
        return
    
    # Process audio files if provided
    if args.fake:
        if os.path.exists(args.fake):
            print(f"\nProcessing FAKE audio file: {args.fake}")
            fake_features = process_audio(args.fake)
            if fake_features is not None:
                print(f"Successfully processed fake audio. Feature shape: {fake_features.shape}")
            else:
                print("Failed to process fake audio file.")
        else:
            print(f"Fake audio file not found: {args.fake}")
    
    if args.real:
        if os.path.exists(args.real):
            print(f"\nProcessing REAL audio file: {args.real}")
            real_features = process_audio(args.real)
            if real_features is not None:
                print(f"Successfully processed real audio. Feature shape: {real_features.shape}")
            else:
                print("Failed to process real audio file.")
        else:
            print(f"Real audio file not found: {args.real}")

if __name__ == "__main__":
    main()
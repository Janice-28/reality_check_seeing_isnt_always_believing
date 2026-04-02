import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import os
import warnings
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure for M1 Mac if needed
if platform.system() == "Darwin" and platform.processor() == "arm":
    os.environ["NUMBA_DISABLE_JIT"] = "1"  # Disable Numba JIT on M1 Mac

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 5  # seconds
N_MELS = 128
MAX_TIME_STEPS = 109  # Corrected to match your model's expected input

# Handle import errors gracefully
try:
    import librosa
    import numpy as np
    import soundfile as sf
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some audio processing dependencies are missing: {e}")
    DEPENDENCIES_AVAILABLE = False

def load_audio(file_path, sr=16000):
    """Load audio file with error handling"""
    if not DEPENDENCIES_AVAILABLE:
        return None, "Required dependencies not available"
    
    try:
        # Try loading with soundfile first (more thread-safe)
        try:
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if sample_rate != sr:
                try:
                    audio_data = librosa.resample(
                        audio_data, 
                        orig_sr=sample_rate, 
                        target_sr=sr,
                        res_type='kaiser_fast'  # Use a faster resampling method
                    )
                    sample_rate = sr
                except Exception as e:
                    logger.warning(f"Resampling failed: {e}")
            
            return audio_data, sample_rate
        except Exception as sf_error:
            logger.warning(f"SoundFile loading failed: {sf_error}, trying librosa...")
        
        # Try with librosa if soundfile fails
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True, res_type='kaiser_fast')
            return audio_data, sample_rate
        except Exception as librosa_error:
            logger.error(f"Librosa loading failed: {librosa_error}")
            return None, f"Failed to load audio: {str(librosa_error)}"
            
    except Exception as e:
        logger.error(f"Audio loading error: {e}")
        return None, f"Error loading audio: {str(e)}"

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Extract mel spectrogram from audio data"""
    try:
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=N_MELS,
            fmax=8000,  # Maximum frequency
            hop_length=512,
            win_length=1024
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Ensure consistent shape - using 109 time steps to match your model
        if mel_spec_norm.shape[1] < MAX_TIME_STEPS:
            # Pad if too short
            pad_width = MAX_TIME_STEPS - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_norm.shape[1] > MAX_TIME_STEPS:
            # Trim if too long
            mel_spec_norm = mel_spec_norm[:, :MAX_TIME_STEPS]
        
        # Add channel dimension for CNN input
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)
        
        return mel_spec_norm
    except Exception as e:
        logger.error(f"Error extracting mel spectrogram: {e}")
        return None

def detect_phase_inconsistency(audio, sr=SAMPLE_RATE):
    """Detect phase inconsistencies typical in synthetic audio"""
    try:
        # Calculate short-time Fourier transform
        stft = librosa.stft(audio)
        phase = np.angle(stft)
        
        # Calculate phase derivative over time
        phase_diff = np.diff(phase, axis=1)
        
        # Unwrap phase to avoid discontinuities
        phase_unwrapped = np.unwrap(phase_diff, axis=1)
        
        # Calculate standard deviation of phase derivative
        phase_std = np.std(phase_unwrapped)
        
        # Synthetic audio often has unnaturally consistent phase changes
        return phase_std < 0.8
    except Exception as e:
        logger.warning(f"Phase inconsistency detection failed: {e}")
        return False

def simple_audio_analysis(audio_data, sr=SAMPLE_RATE):
    """Simple audio analysis that doesn't rely on complex libraries"""
    features = {}
    
    try:
        # Calculate basic statistics
        features['duration'] = len(audio_data) / sr
        features['mean_amplitude'] = float(np.mean(np.abs(audio_data)))
        features['max_amplitude'] = float(np.max(np.abs(audio_data)))
        features['std_amplitude'] = float(np.std(audio_data))
        
        # Calculate energy
        features['energy'] = float(np.sum(audio_data**2) / len(audio_data))
        
        # Calculate zero-crossing rate (simple version)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
        features['zero_crossing_rate'] = float(zero_crossings / len(audio_data))
        
        return features
    
    except Exception as e:
        logger.error(f"Error in simple audio analysis: {e}")
        return {'error': str(e)}

def split_audio_into_chunks(audio, sr=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
    """Split audio into fixed-size chunks"""
    chunk_length = chunk_size * sr
    
    # Split audio into chunks
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        
        # Only keep chunks that are at least 1 second long
        if len(chunk) >= sr:
            # Pad if chunk is shorter than chunk_length
            if len(chunk) < chunk_length:
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')
            chunks.append(chunk)
    
    # If no valid chunks were found, just use the original audio
    if not chunks and len(audio) > 0:
        chunks = [audio]
    
    return chunks

def analyze_acoustic_features(audio, sr=SAMPLE_RATE):
    """Analyze acoustic features of audio data"""
    features = {}
    
    try:
        # Calculate basic statistics
        features['duration'] = len(audio) / sr
        features['mean_amplitude'] = float(np.mean(np.abs(audio)))
        features['max_amplitude'] = float(np.max(np.abs(audio)))
        features['std_amplitude'] = float(np.std(audio))
        
        # Only calculate advanced features if the audio is long enough
        if len(audio) > sr * 0.5:  # At least 0.5 seconds
            try:
                # Calculate pitch
                pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
                pitch_values = []
                
                # Extract valid pitches
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                # Calculate pitch statistics
                if pitch_values:
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                    features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
                    features['pitch_range'] = 0.0
            except Exception as e:
                logger.warning(f"Pitch calculation error: {e}")
            
            try:
                # Calculate MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                features['mfcc_mean'] = float(np.mean(mfccs))
                features['mfcc_std'] = float(np.std(mfccs))
                features['mfcc_stds_mean'] = float(np.mean(np.std(mfccs, axis=1)))
            except Exception as e:
                logger.warning(f"MFCC calculation error: {e}")
            
            try:
                # Calculate spectral contrast
                contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
                features['contrast_mean'] = float(np.mean(contrast))
                features['contrast_std'] = float(np.std(contrast))
            except Exception as e:
                logger.warning(f"Spectral contrast calculation error: {e}")
            
            try:
                # Calculate harmonic-percussive ratio
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_energy = np.sum(harmonic**2)
                percussive_energy = np.sum(percussive**2)
                if harmonic_energy + percussive_energy > 0:
                    features['hp_ratio'] = float(harmonic_energy / (harmonic_energy + percussive_energy))
                else:
                    features['hp_ratio'] = 0.5
            except Exception as e:
                logger.warning(f"Harmonic-percussive separation error: {e}")
            
            try:
                # Calculate spectral flatness
                flatness = librosa.feature.spectral_flatness(y=audio)
                features['flatness_mean'] = float(np.mean(flatness))
                features['flatness_std'] = float(np.std(flatness))
            except Exception as e:
                logger.warning(f"Spectral flatness calculation error: {e}")
                
            try:
                # Calculate spectral rolloff
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                features['rolloff_mean'] = float(np.mean(rolloff))
                features['rolloff_std'] = float(np.std(rolloff))
            except Exception as e:
                logger.warning(f"Spectral rolloff calculation error: {e}")
                
            try:
                # Calculate spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                features['bandwidth_mean'] = float(np.mean(bandwidth))
                features['bandwidth_std'] = float(np.std(bandwidth))
            except Exception as e:
                logger.warning(f"Spectral bandwidth calculation error: {e}")
                
            try:
                # Calculate zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)
                features['zcr_mean'] = float(np.mean(zcr))
                features['zcr_std'] = float(np.std(zcr))
            except Exception as e:
                logger.warning(f"Zero crossing rate calculation error: {e}")
                
            try:
                # Calculate tempo
                onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                features['tempo'] = float(tempo[0])
            except Exception as e:
                logger.warning(f"Tempo calculation error: {e}")
                
            try:
                # Calculate RMS energy
                rms = librosa.feature.rms(y=audio)
                features['rms_mean'] = float(np.mean(rms))
                features['rms_std'] = float(np.std(rms))
                features['rms_max'] = float(np.max(rms))
            except Exception as e:
                logger.warning(f"RMS calculation error: {e}")
                
            try:
                # Calculate envelope stability
                # A synthetic voice often has an unnaturally stable envelope
                hop_length = 512
                frames = 1 + len(audio) // hop_length
                env = np.abs(librosa.stft(audio, hop_length=hop_length)).mean(axis=0)
                
                if len(env) > 1:
                    # Calculate envelope variance
                    env_normalized = (env - np.min(env)) / (np.max(env) - np.min(env) + 1e-10)
                    env_diff = np.diff(env_normalized)
                    features['env_stability'] = float(np.std(env_diff))
                else:
                    features['env_stability'] = 0.0
            except Exception as e:
                logger.warning(f"Envelope stability calculation error: {e}")
                
        return features
    
    except Exception as e:
        logger.error(f"Error analyzing acoustic features: {e}")
        return features

def get_deepfake_indicators(features):
    """Identify potential deepfake indicators from acoustic features"""
    indicators = []
    
    # 1. Pitch stability - flag extremely stable pitch
    if features.get('pitch_std', 100) < 15:  # Increased sensitivity
        score = min(1.0, 1.0 - (features['pitch_std'] / 15))
        indicators.append({
            'name': 'pitch_stability',
            'description': 'Unnaturally stable pitch (common in synthetic voices)',
            'score': float(score)
        })
    
    # 2. MFCC stability - flag extremely stable MFCCs
    if features.get('mfcc_stds_mean', 1.0) < 0.5:  # More sensitive
        score = min(1.0, 1.0 - (features['mfcc_stds_mean'] / 0.5))
        indicators.append({
            'name': 'vocal_stability',
            'description': 'Unnaturally stable vocal characteristics',
            'score': float(score * 1.2)  # Increase weight
        })
    
    # 3. Spectral contrast - flag extremely smooth spectrum
    if features.get('contrast_std', 1.0) < 0.5:
        score = min(1.0, 1.0 - (features['contrast_std'] / 0.5))
        indicators.append({
            'name': 'spectral_smoothness',
            'description': 'Unnaturally smooth spectral features',
            'score': float(score)
        })
    
    # 4. Harmonic-percussive ratio - flag extremely harmonic audio
    hp_ratio = features.get('hp_ratio', 0.5)
    if hp_ratio > 0.8 or hp_ratio < 0.2:  # Detect both extremes
        score = min(1.0, max(abs(hp_ratio - 0.5) * 4, 0.6))
        indicators.append({
            'name': 'harmonic_imbalance',
            'description': 'Unusual harmonic/percussive balance (robotic voice indicator)',
            'score': float(score)
        })
    
    # 5. Check amplitude consistency
    mean_amp = features.get('mean_amplitude', 0)
    max_amp = features.get('max_amplitude', 1)
    if mean_amp > 0.01 and max_amp / mean_amp < 8:  # More sensitive
        score = min(1.0, 0.8)
        indicators.append({
            'name': 'amplitude_consistency',
            'description': 'Unnaturally consistent amplitude (common in synthetic voices)',
            'score': float(score)
        })
    
    # 6. Check spectral flatness
    flatness_mean = features.get('flatness_mean', 0.1)
    if flatness_mean > 0.3 or flatness_mean < 0.05:  # Detect both extremes
        score = min(1.0, 0.7 + abs(flatness_mean - 0.15) * 2)
        indicators.append({
            'name': 'spectral_flatness',
            'description': 'Unusual spectral flatness (common in synthetic audio)',
            'score': float(score)
        })
    
    # 7. Check envelope stability
    env_stability = features.get('env_stability', 0.1)
    if env_stability < 0.05:  # Unnaturally stable envelope
        score = min(1.0, 0.7 + (0.05 - env_stability) * 10)
        indicators.append({
            'name': 'envelope_stability',
            'description': 'Unnaturally stable audio envelope (common in synthetic voices)',
            'score': float(score)
        })
    
    # 8. Check RMS energy consistency
    rms_std = features.get('rms_std', 0.1)
    rms_mean = features.get('rms_mean', 0.1)
    if rms_mean > 0 and rms_std / rms_mean < 0.2:  # Unnaturally consistent energy
        score = min(1.0, 0.7 + (0.2 - rms_std/rms_mean) * 3)
        indicators.append({
            'name': 'energy_consistency',
            'description': 'Unnaturally consistent energy levels (common in synthetic audio)',
            'score': float(score)
        })
    
    # 9. Check for abnormal bandwidth
    bandwidth_mean = features.get('bandwidth_mean', 2000)
    if bandwidth_mean < 1000 or bandwidth_mean > 4000:
        score = 0.7
        indicators.append({
            'name': 'abnormal_bandwidth',
            'description': 'Unusual spectral bandwidth (common in synthetic voices)',
            'score': float(score)
        })
    
    # 10. Check zero crossing rate stability
    zcr_std = features.get('zcr_std', 0.1)
    zcr_mean = features.get('zcr_mean', 0.1)
    if zcr_mean > 0 and zcr_std / zcr_mean < 0.3:
        score = min(1.0, 0.7 + (0.3 - zcr_std/zcr_mean) * 2)
        indicators.append({
            'name': 'zcr_stability',
            'description': 'Unnaturally stable zero crossing rate (synthetic indicator)',
            'score': float(score)
        })
    
    return indicators

def process_audio_file(file_path):
    """Process audio file and extract features"""
    # Load audio
    audio, sr = load_audio(file_path)
    if audio is None:
        return None, sr  # sr contains error message
    
    # Extract features
    features = extract_mel_spectrogram(audio, sr)
    
    return features, audio
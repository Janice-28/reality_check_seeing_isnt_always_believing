# feature_extraction.py
import numpy as np
import librosa
import scipy.signal as signal

def extract_advanced_features(audio_path, sr=16000, duration=5, n_mels=128, max_time_steps=109):
    """
    Extract multiple advanced audio features for deepfake detection
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Ensure audio is long enough
        if len(audio) < sr * duration:
            padding = sr * duration - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        
        # 1. Mel Spectrogram (standard)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels,
            fmin=80, fmax=7600  # Focus on human voice range
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 2. MFCC (captures vocal tract information)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # 3. Spectral Contrast (useful for distinguishing voiced/unvoiced)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        # 4. Chroma Features (capture harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # 5. Spectral Rolloff (useful for voice characteristics)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # 6. Zero Crossing Rate (useful for detecting synthetic voice)
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        # 7. Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr
        )
        
        # 8. Delta and Delta-Delta MFCC (capture dynamics)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Resize all features to consistent dimensions
        features_list = [
            mel_spec_db,  # Primary feature
            mfcc,
            contrast,
            chroma,
            rolloff,
            zcr,
            tonnetz,
            mfcc_delta,
            mfcc_delta2
        ]
        
        # Prepare model input (mel spectrogram)
        if mel_spec_db.shape[1] < max_time_steps:
            mel_spec_db = np.pad(
                mel_spec_db, 
                ((0, 0), (0, max_time_steps - mel_spec_db.shape[1])), 
                mode='constant'
            )
        else:
            mel_spec_db = mel_spec_db[:, :max_time_steps]
        
        # Create feature dictionary for advanced analysis
        feature_dict = {
            'mel_spectrogram': mel_spec_db,
            'mfcc': mfcc,
            'spectral_contrast': contrast,
            'chroma': chroma,
            'rolloff': rolloff,
            'zcr': zcr,
            'tonnetz': tonnetz,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'audio': audio,
            'sr': sr
        }
        
        # Add channel dimension for CNN input
        model_input = np.expand_dims(mel_spec_db, axis=-1)
        
        return model_input, feature_dict
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return empty features
        return np.zeros((n_mels, max_time_steps, 1)), {}
# test_audio_detection.py
import requests
import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_audio(audio_path):
    """Test an audio file with the deepfake detection API"""
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post('http://localhost:8003/predict', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"File: {os.path.basename(audio_path)}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f} ({result['confidence_rating']})")
        
        if 'celebrity_voice' in result:
            print(f"Celebrity Voice Type: {result['celebrity_voice']['type']} ({result['celebrity_voice']['category']})")
        
        if 'key_indicators' in result:
            print("\nKey Indicators:")
            for indicator in result['key_indicators']:
                print(f"- {indicator['description']} (Score: {indicator['score']:.2f})")
        
        if 'ensemble_analysis' in result:
            print("\nEnsemble Analysis:")
            print(f"- Agreement Rate: {result['ensemble_analysis']['agreement_rate']:.2f}")
            print(f"- Confidence Stability: {result['ensemble_analysis']['confidence_stability']:.2f}")
        
        print("-" * 60)
        return result
    else:
        print(f"Error: {response.text}")
        return None

def analyze_audio_characteristics(audio_path):
    """Analyze audio characteristics to check for common deepfake traits"""
    import librosa
    import numpy as np
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate key characteristics
        results = {}
        
        # 1. Pitch statistics
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        
        # Extract valid pitches
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            results['pitch_mean'] = np.mean(pitch_values)
            results['pitch_std'] = np.std(pitch_values)
            results['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        
        # 2. Spectral characteristics
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        results['spectral_centroid_mean'] = np.mean(spectral_centroid)
        results['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # 3. Harmonics
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = np.sum(harmonic**2)
        percussive_energy = np.sum(percussive**2)
        
        if harmonic_energy > 0 and percussive_energy > 0:
            results['harmonic_percussive_ratio'] = harmonic_energy / (harmonic_energy + percussive_energy)
        
        # 4. MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        results['mfcc_means'] = np.mean(mfccs, axis=1).tolist()
        results['mfcc_stds'] = np.std(mfccs, axis=1).tolist()
        
        # 5. Check for unnatural stability (common in deepfakes)
        if results.get('pitch_std', 0) < 5.0:
            results['unnatural_pitch_stability'] = True
        
        if np.mean(results.get('mfcc_stds', [10])) < 1.0:
            results['unnatural_mfcc_stability'] = True
        
        # 6. Check for robotic voice indicators
        if results.get('harmonic_percussive_ratio', 0.5) > 0.95:
            results['robotic_voice_indicator'] = True
        
        return results
    
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return {}

def test_audio_with_analysis(audio_path):
    """Test audio with additional analysis beyond the API"""
    # First, test with the API
    api_result = test_audio(audio_path)
    
    # Then, perform additional analysis
    print("\nDetailed Audio Analysis:")
    print("=" * 60)
    
    characteristics = analyze_audio_characteristics(audio_path)
    
    if characteristics:
        # Print key characteristics
        if 'pitch_mean' in characteristics:
            print(f"Pitch Mean: {characteristics['pitch_mean']:.2f} Hz")
        if 'pitch_std' in characteristics:
            print(f"Pitch Stability: {characteristics['pitch_std']:.2f}")
        if 'harmonic_percussive_ratio' in characteristics:
            print(f"Harmonic-Percussive Ratio: {characteristics['harmonic_percussive_ratio']:.2f}")
        
        # Print deepfake indicators
        print("\nDeepfake Indicators:")
        if characteristics.get('unnatural_pitch_stability', False):
            print("- Unnaturally stable pitch (common in synthetic voices)")
        if characteristics.get('unnatural_mfcc_stability', False):
            print("- Unnaturally stable vocal characteristics")
        if characteristics.get('robotic_voice_indicator', False):
            print("- High harmonic content (robotic voice indicator)")
            
        # No indicators found
        if not any([
            characteristics.get('unnatural_pitch_stability', False),
            characteristics.get('unnatural_mfcc_stability', False),
            characteristics.get('robotic_voice_indicator', False)
        ]):
            print("- No strong deepfake indicators found in the audio characteristics")
    
    print("=" * 60)
    
    return api_result, characteristics

def visualize_audio_analysis(audio_path, output_path=None):
    """Create visualizations of audio analysis for deepfake detection"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Create figure with multiple plots
        plt.figure(figsize=(12, 10))
        
        # 1. Waveform
        plt.subplot(3, 2, 1)
        plt.title('Waveform')
        librosa.display.waveshow(audio, sr=sr)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # 2. Mel Spectrogram
        plt.subplot(3, 2, 2)
        plt.title('Mel Spectrogram')
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        
        # 3. Pitch Track
        plt.subplot(3, 2, 3)
        plt.title('Pitch Track')
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        times = []
        
        # Extract valid pitches
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                times.append(librosa.frames_to_time(t, sr=sr))
        
        if pitch_values:
            plt.scatter(times, pitch_values, alpha=0.5, s=1)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
        
        # 4. Spectral Contrast
        plt.subplot(3, 2, 4)
        plt.title('Spectral Contrast')
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        librosa.display.specshow(contrast, x_axis='time')
        plt.colorbar()
        
        # 5. MFCC
        plt.subplot(3, 2, 5)
        plt.title('MFCCs')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        
        # 6. Harmonic-Percussive Separation
        plt.subplot(3, 2, 6)
        plt.title('Harmonic Component')
        harmonic, percussive = librosa.effects.hpss(audio)
        librosa.display.waveshow(harmonic, sr=sr, alpha=0.5, label='Harmonic')
        librosa.display.waveshow(percussive, sr=sr, color='r', alpha=0.5, label='Percussive')
        plt.legend()
        
        plt.tight_layout()
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error in visualization: {e}")

def batch_test_directory(directory):
    """Test all audio files in a directory and compute accuracy metrics"""
    results = []
    files_tested = 0
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Test each file
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(root, file)
                
                # Determine expected label from filename
                expected_fake = 'fake' in file.lower() or 'deepfake' in file.lower()
                
                # Test the file
                result = test_audio(file_path)
                files_tested += 1
                
                if result:
                    # Determine if prediction was correct
                    predicted_fake = result['prediction'] == 'FAKE'
                    correct = predicted_fake == expected_fake
                    
                    # Save result
                    results.append({
                        'file': file,
                        'expected': 'FAKE' if expected_fake else 'REAL',
                        'predicted': result['prediction'],
                        'confidence': result['confidence'],
                        'correct': correct
                    })
    
    # Calculate accuracy metrics
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        # Calculate precision and recall for FAKE class
        true_positives = sum(1 for r in results if r['predicted'] == 'FAKE' and r['expected'] == 'FAKE')
        false_positives = sum(1 for r in results if r['predicted'] == 'FAKE' and r['expected'] == 'REAL')
        false_negatives = sum(1 for r in results if r['predicted'] == 'REAL' and r['expected'] == 'FAKE')
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n" + "="*60)
        print(f"Testing Results: {files_tested} files tested")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
        print("="*60)
        
        # Save results to CSV
        with open(results_dir / "test_results.csv", "w") as f:
            f.write("File,Expected,Predicted,Confidence,Correct\n")
            for r in results:
                f.write(f"{r['file']},{r['expected']},{r['predicted']},{r['confidence']:.2f},{r['correct']}\n")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [i for i, r in enumerate(results) if r['expected'] == 'REAL'], 
            [r['confidence'] for r in results if r['expected'] == 'REAL'],
            c='green', label='Real', alpha=0.7
        )
        plt.scatter(
            [i for i, r in enumerate(results) if r['expected'] == 'FAKE'], 
            [r['confidence'] for r in results if r['expected'] == 'FAKE'][::-1],
            c='red', label='Fake', alpha=0.7
        )
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.ylabel('Confidence')
        plt.xlabel('Sample Index')
        plt.ylim(0, 1)
        plt.legend()
        plt.title('Deepfake Detection Results')
        plt.savefig(results_dir / "results_plot.png")
        
        return accuracy, precision, recall, f1_score
    
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test audio deepfake detection')
    parser.add_argument('--dir', type=str, help='Directory of audio files to test')
    parser.add_argument('--file', type=str, help='Single audio file to test')
    parser.add_argument('--analysis', action='store_true', help='Perform detailed analysis')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    if args.file:
        if args.analysis:
            test_audio_with_analysis(args.file)
        else:
            test_audio(args.file)
            
        if args.visualize:
            output_path = f"{os.path.splitext(args.file)[0]}_analysis.png"
            visualize_audio_analysis(args.file, output_path)
            
    elif args.dir:
        batch_test_directory(args.dir)
    else:
        print("Please provide either --file or --dir argument")
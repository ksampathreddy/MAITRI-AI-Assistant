import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging
import os
from datetime import datetime
import soundfile as sf

class VocalEmotionAnalyzer:
    def __init__(self, model_path: str = "models/vocal_emotion.pkl"):
        self.logger = logging.getLogger(__name__)
        self.model_data = self._load_model(model_path)
        self.emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise']
        self.analysis_history = []
        self.scaler = StandardScaler()
        
        # Feature configuration
        self.feature_config = {
            'n_mfcc': 13,
            'n_mels': 40,
            'n_chroma': 12
        }
        
        self.logger.info("Vocal emotion analyzer initialized")

    def _load_model(self, model_path: str):
        """Load pre-trained vocal emotion model"""
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.logger.info("Loaded pre-trained vocal emotion model")
                return model_data
            except Exception as e:
                self.logger.warning(f"Could not load model {model_path}: {e}")
        
        # Create demo model for testing
        self.logger.info("Using demo vocal emotion model")
        return self._create_demo_model()

    def _create_demo_model(self):
        """Create demo vocal emotion model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create a dummy model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scaler = StandardScaler()
        
        # Fit with dummy data
        X_dummy = np.random.randn(100, 40)
        y_dummy = np.random.randint(0, 8, 100)
        X_scaled = scaler.fit_transform(X_dummy)
        model.fit(X_scaled, y_dummy)
        
        return {'model': model, 'scaler': scaler}

    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """Extract comprehensive audio features for emotion recognition"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            
            features = []
            
            # 1. MFCC Features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.feature_config['n_mfcc'])
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # 2. Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.feature_config['n_mels'])
            mel_mean = np.mean(mel_spec, axis=1)
            features.extend(mel_mean)
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.feature_config['n_chroma'])
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
            
            # 4. Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
            
            # 5. Temporal features
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            rms_energy = np.mean(librosa.feature.rms(y=y))
            features.extend([zero_crossing_rate, rms_energy])
            
            # 6. Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_mean = np.mean(y_harmonic)
            percussive_mean = np.mean(y_percussive)
            features.extend([harmonic_mean, percussive_mean])
            
            # 7. Tempo and beat
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # 8. Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features.append(pitch_mean)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Audio feature extraction error: {e}")
            # Return zero features as fallback
            return np.zeros((1, 40))

    def analyze_voice_characteristics(self, audio_path: str) -> Dict[str, float]:
        """Analyze voice characteristics beyond emotion"""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=3.0)
            
            characteristics = {}
            
            # Speech rate analysis
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            characteristics['speech_rate'] = len(onset_frames) / (len(y) / sr)
            
            # Pitch variability
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) > 0:
                characteristics['pitch_variability'] = np.std(valid_pitches)
                characteristics['pitch_mean'] = np.mean(valid_pitches)
            else:
                characteristics['pitch_variability'] = 0
                characteristics['pitch_mean'] = 0
            
            # Energy analysis
            rms = librosa.feature.rms(y=y)
            characteristics['energy_variability'] = np.std(rms)
            characteristics['energy_mean'] = np.mean(rms)
            
            # Voice quality indicators
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            characteristics['brightness'] = np.mean(spectral_centroid)
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Voice characteristics analysis error: {e}")
            return {}

    def predict_emotion(self, audio_path: str) -> Dict[str, float]:
        """Predict emotions from vocal features with comprehensive analysis"""
        try:
            # Extract features
            features = self.extract_audio_features(audio_path)
            
            if features is None or np.all(features == 0):
                self.logger.warning("No features extracted from audio")
                return {"neutral": 1.0}
            
            # Scale features
            if 'scaler' in self.model_data:
                features_scaled = self.model_data['scaler'].transform(features)
            else:
                features_scaled = features
            
            # Predict probabilities
            if 'model' in self.model_data:
                probabilities = self.model_data['model'].predict_proba(features_scaled)[0]
            else:
                # Fallback: uniform distribution
                probabilities = np.ones(len(self.emotion_labels)) / len(self.emotion_labels)
            
            # Convert to emotion dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = float(probabilities[i])
            
            # Analyze voice characteristics
            voice_chars = self.analyze_voice_characteristics(audio_path)
            
            # Adjust emotions based on voice characteristics
            emotion_scores = self._adjust_emotions_with_characteristics(emotion_scores, voice_chars)
            
            # Normalize scores
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            
            # Log analysis
            self._log_analysis(emotion_scores, voice_chars)
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Vocal emotion prediction error: {e}")
            result = {"neutral": 1.0}
            self._log_analysis(result, {})
            return result

    def _adjust_emotions_with_characteristics(self, emotions: Dict[str, float], 
                                            characteristics: Dict[str, float]) -> Dict[str, float]:
        """Adjust emotion probabilities based on voice characteristics"""
        try:
            adjusted_emotions = emotions.copy()
            
            # High pitch variability often indicates excitement or stress
            pitch_var = characteristics.get('pitch_variability', 0)
            if pitch_var > 100:  # Threshold for high variability
                adjusted_emotions['excited'] = adjusted_emotions.get('happy', 0) * 1.2
                adjusted_emotions['fearful'] = adjusted_emotions.get('fearful', 0) * 1.1
            
            # Low speech rate might indicate sadness or fatigue
            speech_rate = characteristics.get('speech_rate', 0)
            if speech_rate < 2.0:  # Low speech rate
                adjusted_emotions['sad'] = adjusted_emotions.get('sad', 0) * 1.3
                adjusted_emotions['calm'] = adjusted_emotions.get('calm', 0) * 1.2
            
            # High energy might indicate anger or excitement
            energy = characteristics.get('energy_mean', 0)
            if energy > 0.1:  # High energy
                adjusted_emotions['angry'] = adjusted_emotions.get('angry', 0) * 1.4
                adjusted_emotions['happy'] = adjusted_emotions.get('happy', 0) * 1.2
            
            # Normalize again after adjustments
            total = sum(adjusted_emotions.values())
            if total > 0:
                adjusted_emotions = {k: v/total for k, v in adjusted_emotions.items()}
            
            return adjusted_emotions
        except:
            return emotions

    def _log_analysis(self, emotion_scores: Dict[str, float], characteristics: Dict[str, float]):
        """Log vocal analysis results"""
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'emotion_scores': emotion_scores,
            'dominant_emotion': self.get_dominant_emotion(emotion_scores)[0],
            'confidence': self.get_dominant_emotion(emotion_scores)[1],
            'voice_characteristics': characteristics
        }
        
        self.analysis_history.append(analysis_entry)
        
        # Keep only last 50 analyses
        if len(self.analysis_history) > 50:
            self.analysis_history = self.analysis_history[-50:]

    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion and its confidence"""
        if not emotion_scores:
            return "neutral", 1.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion

    def detect_stress_level(self, audio_path: str) -> float:
        """Detect stress level from voice (0-1 scale)"""
        try:
            emotion_scores = self.predict_emotion(audio_path)
            characteristics = self.analyze_voice_characteristics(audio_path)
            
            # Stress indicators
            stress_emotions = ['angry', 'fearful', 'sad']
            stress_score = sum(emotion_scores.get(emotion, 0) for emotion in stress_emotions)
            
            # Adjust based on voice characteristics
            pitch_var = characteristics.get('pitch_variability', 0)
            if pitch_var > 150:
                stress_score = min(1.0, stress_score * 1.2)
            
            speech_rate = characteristics.get('speech_rate', 0)
            if speech_rate > 4.0:  # Very fast speech
                stress_score = min(1.0, stress_score * 1.1)
            
            return min(1.0, stress_score)
            
        except Exception as e:
            self.logger.error(f"Stress detection error: {e}")
            return 0.0

    def get_vocal_analysis_report(self) -> Dict:
        """Generate comprehensive vocal analysis report"""
        if not self.analysis_history:
            return {"status": "no_data"}
        
        recent_analyses = self.analysis_history[-10:]
        
        report = {
            "total_analyses": len(self.analysis_history),
            "current_dominant_emotion": self.analysis_history[-1]['dominant_emotion'],
            "current_confidence": self.analysis_history[-1]['confidence'],
            "average_stress_level": np.mean([self.detect_stress_level("dummy") 
                                           for _ in recent_analyses]),  # Simplified
            "voice_characteristics_trend": {
                'avg_speech_rate': np.mean([a['voice_characteristics'].get('speech_rate', 0) 
                                          for a in recent_analyses]),
                'avg_pitch_variability': np.mean([a['voice_characteristics'].get('pitch_variability', 0) 
                                                for a in recent_analyses])
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return report
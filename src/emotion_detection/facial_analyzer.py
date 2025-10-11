import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List
import logging
import os
from datetime import datetime

class FacialEmotionAnalyzer:
    def __init__(self, model_path: str = "models/facial_emotion.h5"):
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.analysis_history = []
        
        # Initialize face detection from DNN for better accuracy
        self.net = self._initialize_dnn_face_detector()
        
        self.logger.info("Facial emotion analyzer initialized")

    def _initialize_dnn_face_detector(self):
        """Initialize DNN-based face detector for better accuracy"""
        try:
            # Load pre-trained DNN model
            prototxt_path = "models/deploy.prototxt"
            model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.logger.info("DNN face detector initialized")
                return net
            else:
                self.logger.warning("DNN model files not found, using Haar cascade")
                return None
        except Exception as e:
            self.logger.warning(f"Could not initialize DNN detector: {e}")
            return None

    def _load_model(self, model_path: str):
        """Load pre-trained facial emotion recognition model"""
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                self.logger.info("Loaded pre-trained facial emotion model")
                return model
            except Exception as e:
                self.logger.warning(f"Could not load model {model_path}: {e}")
        
        # Create a demo model for testing
        self.logger.info("Using demo facial emotion model")
        return self._create_demo_model()

    def _create_demo_model(self):
        """Create a demo model that returns simulated emotions"""
        class DemoModel:
            def __init__(self):
                self.emotion_weights = {
                    'neutral': 0.3, 'happy': 0.2, 'sad': 0.15, 
                    'angry': 0.1, 'fear': 0.1, 'surprise': 0.1, 'disgust': 0.05
                }
                
            def predict(self, x):
                batch_size = x.shape[0]
                # Generate realistic emotion probabilities based on face presence
                base_probs = np.array([self.emotion_weights[label] for label in 
                                     ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']])
                
                # Add some randomness
                noise = np.random.normal(0, 0.1, 7)
                probs = base_probs + noise
                probs = np.clip(probs, 0.01, 0.99)
                probs = probs / probs.sum()
                
                return probs.reshape(1, -1)
        
        return DemoModel()

    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN for better accuracy"""
        if self.net is None:
            return []
            
        try:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    faces.append((startX, startY, endX - startX, endY - startY))
            
            return faces
        except Exception as e:
            self.logger.error(f"DNN face detection error: {e}")
            return []

    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using best available method"""
        # Try DNN first, fallback to Haar
        faces = self.detect_faces_dnn(frame)
        if not faces:
            faces = self.detect_faces_haar(frame)
        return faces

    def extract_facial_features(self, face_roi: np.ndarray) -> Dict[str, float]:
        """Extract additional facial features for analysis"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            features = {}
            
            # Eye detection for fatigue analysis
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
            features['eyes_detected'] = len(eyes)
            
            # Mouth detection for expression analysis
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            mouths = mouth_cascade.detectMultiScale(gray, 1.8, 20)
            features['mouth_detected'] = len(mouths) > 0
            
            # Basic facial metrics
            height, width = gray.shape
            features['face_symmetry'] = self._calculate_symmetry(gray)
            features['face_brightness'] = np.mean(gray)
            
            return features
        except Exception as e:
            self.logger.error(f"Facial feature extraction error: {e}")
            return {}

    def _calculate_symmetry(self, face_image: np.ndarray) -> float:
        """Calculate face symmetry score"""
        try:
            height, width = face_image.shape
            mid = width // 2
            
            # Split face into left and right halves
            left_half = face_image[:, :mid]
            right_half = face_image[:, mid:]
            
            # Flip right half for comparison
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize if dimensions don't match
            if left_half.shape != right_half_flipped.shape:
                right_half_flipped = cv2.resize(right_half_flipped, 
                                              (left_half.shape[1], left_half.shape[0]))
            
            # Calculate symmetry score
            diff = cv2.absdiff(left_half, right_half_flipped)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            return max(0.0, min(1.0, symmetry_score))
        except:
            return 0.5

    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face region for emotion prediction"""
        try:
            # Convert to grayscale
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
            
            # Resize to 48x48 for FER2013 compatibility
            resized = cv2.resize(gray, (48, 48))
            
            # Normalize pixel values
            normalized = resized.astype('float32') / 255.0
            
            # Reshape for model input
            reshaped = normalized.reshape(1, 48, 48, 1)
            
            return reshaped
        except Exception as e:
            self.logger.error(f"Face preprocessing error: {e}")
            # Return zero array as fallback
            return np.zeros((1, 48, 48, 1))

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze frame and return emotion probabilities with facial features"""
        try:
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                # No face detected
                result = {"neutral": 1.0}
                self._log_analysis(result, False)
                return result
            
            # Use the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract facial features
            facial_features = self.extract_facial_features(face_roi)
            
            # Preprocess for emotion prediction
            processed_face = self.preprocess_face(face_roi)
            
            # Predict emotions
            predictions = self.model.predict(processed_face, verbose=0)
            
            # Convert to emotion dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotion_labels):
                emotion_scores[emotion] = float(predictions[0][i])
            
            # Normalize scores
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v/total for k, v in emotion_scores.items()}
            
            # Adjust based on facial features
            emotion_scores = self._adjust_emotions_with_features(emotion_scores, facial_features)
            
            # Log analysis
            self._log_analysis(emotion_scores, True, facial_features)
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Facial analysis error: {e}")
            result = {"neutral": 1.0}
            self._log_analysis(result, False)
            return result

    def _adjust_emotions_with_features(self, emotions: Dict[str, float], 
                                     features: Dict[str, float]) -> Dict[str, float]:
        """Adjust emotion probabilities based on facial features"""
        try:
            adjusted_emotions = emotions.copy()
            
            # Adjust based on mouth detection (smile)
            if features.get('mouth_detected', False):
                adjusted_emotions['happy'] = min(1.0, adjusted_emotions.get('happy', 0) * 1.3)
                adjusted_emotions['sad'] = max(0.0, adjusted_emotions.get('sad', 0) * 0.7)
            
            # Adjust based on eyes (fatigue)
            if features.get('eyes_detected', 0) == 0:  # Eyes closed or not detected
                adjusted_emotions['neutral'] = min(1.0, adjusted_emotions.get('neutral', 0) * 1.2)
            
            # Normalize again after adjustments
            total = sum(adjusted_emotions.values())
            if total > 0:
                adjusted_emotions = {k: v/total for k, v in adjusted_emotions.items()}
            
            return adjusted_emotions
        except:
            return emotions

    def _log_analysis(self, emotion_scores: Dict[str, float], face_detected: bool, 
                     features: Dict = None):
        """Log analysis results for monitoring"""
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'face_detected': face_detected,
            'emotion_scores': emotion_scores,
            'dominant_emotion': self.get_dominant_emotion(emotion_scores)[0],
            'confidence': self.get_dominant_emotion(emotion_scores)[1],
            'facial_features': features or {}
        }
        
        self.analysis_history.append(analysis_entry)
        
        # Keep only last 100 analyses
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion and its confidence"""
        if not emotion_scores:
            return "neutral", 1.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion

    def get_emotional_trend(self, window_size: int = 10) -> Dict[str, float]:
        """Get emotional trend over recent analyses"""
        if len(self.analysis_history) < window_size:
            return {}
        
        recent_analyses = self.analysis_history[-window_size:]
        
        # Calculate average emotion scores
        trend = {}
        for emotion in self.emotion_labels:
            scores = [analysis['emotion_scores'].get(emotion, 0) 
                     for analysis in recent_analyses 
                     if analysis['face_detected']]
            if scores:
                trend[emotion] = np.mean(scores)
        
        return trend

    def get_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        if not self.analysis_history:
            return {"status": "no_data"}
        
        recent_with_faces = [a for a in self.analysis_history[-20:] if a['face_detected']]
        
        if not recent_with_faces:
            return {"status": "no_recent_faces"}
        
        report = {
            "total_analyses": len(self.analysis_history),
            "recent_face_detection_rate": len(recent_with_faces) / 20,
            "current_dominant_emotion": self.analysis_history[-1]['dominant_emotion'],
            "current_confidence": self.analysis_history[-1]['confidence'],
            "emotional_trend": self.get_emotional_trend(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
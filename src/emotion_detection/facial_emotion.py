import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List
import logging

class FacialEmotionDetector:
    def __init__(self, model_path: str):
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model(model_path)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def _load_model(self, model_path: str):
        """Load pre-trained emotion recognition model"""
        try:
            model = tf.keras.models.load_model(model_path)
            self.logger.info("Facial emotion model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Fallback to a simple model
            return self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model for demo purposes"""
        # In practice, you would use a pre-trained model like FER2013
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(7, activation='softmax')
        ])
        return model
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face ROI for emotion prediction"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)
        return reshaped
    
    def predict_emotion(self, frame: np.ndarray) -> Dict[str, float]:
        """Predict emotions from facial features"""
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return {"neutral": 1.0}
        
        # Use the first face detected
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        processed_face = self.preprocess_face(face_roi)
        predictions = self.model.predict(processed_face, verbose=0)
        
        emotion_scores = {}
        for i, emotion in enumerate(self.emotion_labels):
            emotion_scores[emotion] = float(predictions[0][i])
        
        return emotion_scores
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion and its confidence"""
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion
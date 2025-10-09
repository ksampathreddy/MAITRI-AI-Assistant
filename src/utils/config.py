import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EmotionConfig:
    FER_MODEL_PATH: str = "models/emotion_models/fer_model.h5"
    VOCAL_MODEL_PATH: str = "models/emotion_models/vocal_model.pkl"
    EMOTION_LABELS: List[str] = None
    
    def __post_init__(self):
        self.EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

@dataclass
class ConversationConfig:
    RESPONSE_TEMPLATES: Dict[str, List[str]] = None
    INTERVENTION_THRESHOLD: float = 0.7
    
    def __post_init__(self):
        self.RESPONSE_TEMPLATES = {
            "stress": [
                "I notice you seem stressed. Would you like to try a quick breathing exercise?",
                "It's normal to feel stressed in challenging environments. Let's focus on one task at a time.",
                "How about taking a short break? I can guide you through a relaxation technique."
            ],
            "fatigue": [
                "You seem tired. Remember to maintain your sleep schedule for optimal performance.",
                "Fatigue can affect decision-making. Consider taking a rest period.",
                "Let me adjust the lighting to help reduce eye strain and fatigue."
            ],
            "anxiety": [
                "I sense some anxiety. Would you like to talk about what's concerning you?",
                "Anxiety is common in isolated environments. Let's practice some grounding techniques.",
                "Remember your training and take deep, slow breaths with me."
            ]
        }

@dataclass
class SystemConfig:
    OFFLINE_MODE: bool = True
    DATA_PATH: str = "data/"
    MODEL_PATH: str = "models/"
    LOG_LEVEL: str = "INFO"
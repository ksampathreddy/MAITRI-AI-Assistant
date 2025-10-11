import os
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class MAITRIConfig:
    # System settings
    OFFLINE_MODE: bool = True
    LOG_LEVEL: str = "INFO"
    DATA_PATH: str = "data/"
    MODEL_PATH: str = "models/"
    
    # Camera settings
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    
    # Audio settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    WAKE_WORD: str = "maitri"
    
    # Emotion detection thresholds
    EMOTION_THRESHOLD: float = 0.7
    INTERVENTION_THRESHOLD: float = 0.8
    
    # Model paths
    FACIAL_MODEL_PATH: str = "models/facial_emotion.h5"
    VOCAL_MODEL_PATH: str = "models/vocal_emotion.pkl"
    POSTURE_MODEL_PATH: str = "models/posture_detector.pkl"
    
    # Response settings
    MAX_RESPONSE_LENGTH: int = 200
    RESPONSE_TEMPERATURE: float = 0.7
    
    # Psychological intervention templates
    INTERVENTION_TEMPLATES: Dict[str, List[str]] = None
    
    def __post_init__(self):
        self.INTERVENTION_TEMPLATES = {
            "stress": [
                "I notice signs of stress. Let's practice a quick breathing exercise together.",
                "Stress is common in high-pressure environments. Remember your training and take deep breaths.",
                "How about we break this down into smaller, manageable steps?"
            ],
            "fatigue": [
                "I'm detecting fatigue. Consider taking a scheduled rest period.",
                "Fatigue can impact performance. Let me adjust the environmental controls for better comfort.",
                "Remember the importance of sleep hygiene in space environments."
            ],
            "isolation": [
                "Feelings of isolation are normal. Remember your team is just a call away.",
                "You're doing important work for humanity. Your efforts are valued.",
                "Would you like to schedule a video call with your support network?"
            ],
            "physical_discomfort": [
                "I notice physical discomfort. Let me guide you through some microgravity exercises.",
                "Proper ergonomics are crucial. Would you like posture adjustment suggestions?",
                "Consider using the onboard exercise equipment to relieve physical tension."
            ]
        }
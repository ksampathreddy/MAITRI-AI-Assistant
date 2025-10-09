from typing import Dict, Tuple, List
import numpy as np
import logging

class MultimodalEmotionFusion:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.facial_weight = 0.6
        self.vocal_weight = 0.4
    
    def fuse_emotions(self, facial_emotions: Dict[str, float], 
                     vocal_emotions: Dict[str, float]) -> Dict[str, float]:
        """Fuse facial and vocal emotion predictions"""
        
        # Map emotions to common labels
        common_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
        
        fused_emotions = {}
        
        for emotion in common_labels:
            facial_score = facial_emotions.get(emotion, 0.0)
            vocal_score = vocal_emotions.get(emotion, 0.0)
            
            # Weighted fusion
            fused_score = (self.facial_weight * facial_score + 
                          self.vocal_weight * vocal_score)
            
            fused_emotions[emotion] = fused_score
        
        # Normalize scores
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: v/total for k, v in fused_emotions.items()}
        
        return fused_emotions
    
    def detect_critical_state(self, fused_emotions: Dict[str, float], 
                            threshold: float = 0.7) -> Tuple[bool, str]:
        """Detect if emotional state requires intervention"""
        
        high_risk_emotions = ['angry', 'fear', 'sad']
        
        for emotion in high_risk_emotions:
            if fused_emotions.get(emotion, 0) > threshold:
                return True, emotion
        
        return False, ""
    
    def get_emotional_summary(self, fused_emotions: Dict[str, float]) -> Dict:
        """Generate emotional state summary"""
        dominant_emotion = max(fused_emotions.items(), key=lambda x: x[1])
        
        summary = {
            "dominant_emotion": dominant_emotion[0],
            "confidence": dominant_emotion[1],
            "all_emotions": fused_emotions,
            "wellbeing_score": self._calculate_wellbeing_score(fused_emotions),
            "requires_intervention": False,
            "risk_level": "low"
        }
        
        # Check for critical state
        is_critical, critical_emotion = self.detect_critical_state(fused_emotions)
        if is_critical:
            summary["requires_intervention"] = True
            summary["risk_level"] = "high" if critical_emotion in ['angry', 'fear'] else "medium"
            summary["critical_emotion"] = critical_emotion
        
        return summary
    
    def _calculate_wellbeing_score(self, emotions: Dict[str, float]) -> float:
        """Calculate overall wellbeing score (0-100)"""
        positive_emotions = emotions.get('happy', 0) + emotions.get('neutral', 0) * 0.5
        negative_emotions = (emotions.get('angry', 0) + emotions.get('fear', 0) + 
                           emotions.get('sad', 0) + emotions.get('disgust', 0))
        
        wellbeing = max(0, min(100, (positive_emotions - negative_emotions + 1) * 50))
        return wellbeing
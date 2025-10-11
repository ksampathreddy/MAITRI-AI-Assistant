from typing import Dict, Tuple, List
import numpy as np
import logging
from datetime import datetime
from enum import Enum

class FusionMethod(Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    DECISION_LEVEL = "decision_level"
    FEATURE_LEVEL = "feature_level"

class MultimodalFusion:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Fusion weights (can be adjusted based on modality reliability)
        self.weights = {
            'facial': 0.6,
            'vocal': 0.4
        }
        
        # Emotion mapping between modalities
        self.emotion_mapping = {
            'facial': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
            'vocal': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprise'],
            'common': ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
        }
        
        # Fusion history for trend analysis
        self.fusion_history = []
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        self.logger.info("Multimodal fusion engine initialized")

    def map_emotions_to_common_space(self, facial_emotions: Dict[str, float], 
                                   vocal_emotions: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Map facial and vocal emotions to common emotion space"""
        common_facial = {}
        common_vocal = {}
        
        # Map facial emotions
        for common_emotion in self.emotion_mapping['common']:
            if common_emotion in facial_emotions:
                common_facial[common_emotion] = facial_emotions[common_emotion]
            else:
                # Find closest mapping
                if common_emotion == 'fear':
                    common_facial[common_emotion] = facial_emotions.get('fear', 0)
                else:
                    common_facial[common_emotion] = 0.0
        
        # Map vocal emotions
        vocal_mapping = {
            'neutral': 'neutral',
            'calm': 'neutral',
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'fear',
            'disgust': 'angry',  # Approximation
            'surprise': 'surprise'
        }
        
        for vocal_emotion, common_emotion in vocal_mapping.items():
            if common_emotion not in common_vocal:
                common_vocal[common_emotion] = 0.0
            common_vocal[common_emotion] += vocal_emotions.get(vocal_emotion, 0)
        
        # Normalize vocal emotions
        total_vocal = sum(common_vocal.values())
        if total_vocal > 0:
            common_vocal = {k: v/total_vocal for k, v in common_vocal.items()}
        
        return common_facial, common_vocal

    def weighted_average_fusion(self, facial_emotions: Dict[str, float], 
                              vocal_emotions: Dict[str, float]) -> Dict[str, float]:
        """Fuse emotions using weighted average method"""
        # Map to common emotion space
        common_facial, common_vocal = self.map_emotions_to_common_space(facial_emotions, vocal_emotions)
        
        fused_emotions = {}
        
        for emotion in self.emotion_mapping['common']:
            facial_score = common_facial.get(emotion, 0.0)
            vocal_score = common_vocal.get(emotion, 0.0)
            
            # Apply modality weights
            fused_score = (self.weights['facial'] * facial_score + 
                         self.weights['vocal'] * vocal_score)
            
            fused_emotions[emotion] = fused_score
        
        # Normalize
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: v/total for k, v in fused_emotions.items()}
        
        return fused_emotions

    def decision_level_fusion(self, facial_emotions: Dict[str, float], 
                            vocal_emotions: Dict[str, float]) -> Dict[str, float]:
        """Fuse emotions at decision level using confidence scores"""
        # Get dominant emotions and confidences
        facial_dominant, facial_confidence = self._get_dominant_emotion(facial_emotions)
        vocal_dominant, vocal_confidence = self._get_dominant_emotion(vocal_emotions)
        
        # Map to common space
        common_facial, common_vocal = self.map_emotions_to_common_space(facial_emotions, vocal_emotions)
        
        # Calculate modality reliability
        facial_reliability = self._calculate_modality_reliability(facial_confidence, facial_emotions)
        vocal_reliability = self._calculate_modality_reliability(vocal_confidence, vocal_emotions)
        
        # Normalize reliabilities for weights
        total_reliability = facial_reliability + vocal_reliability
        if total_reliability > 0:
            facial_weight = facial_reliability / total_reliability
            vocal_weight = vocal_reliability / total_reliability
        else:
            facial_weight = vocal_weight = 0.5
        
        # Fuse based on reliability-weighted combination
        fused_emotions = {}
        for emotion in self.emotion_mapping['common']:
            fused_score = (facial_weight * common_facial.get(emotion, 0) + 
                         vocal_weight * common_vocal.get(emotion, 0))
            fused_emotions[emotion] = fused_score
        
        # Normalize
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: v/total for k, v in fused_emotions.items()}
        
        return fused_emotions

    def _get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get dominant emotion and its confidence"""
        if not emotion_scores:
            return "neutral", 1.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion

    def _calculate_modality_reliability(self, confidence: float, emotion_scores: Dict[str, float]) -> float:
        """Calculate modality reliability based on confidence and emotion distribution"""
        if not emotion_scores:
            return 0.0
        
        # Base reliability on confidence
        reliability = confidence
        
        # Adjust based on emotion distribution (more uniform distribution = less reliable)
        entropy = self._calculate_entropy(emotion_scores)
        max_entropy = np.log(len(emotion_scores))
        
        if max_entropy > 0:
            # High entropy (uniform distribution) reduces reliability
            reliability *= (1 - (entropy / max_entropy) * 0.5)
        
        return max(0.0, min(1.0, reliability))

    def _calculate_entropy(self, emotion_scores: Dict[str, float]) -> float:
        """Calculate entropy of emotion distribution"""
        scores = np.array(list(emotion_scores.values()))
        scores = scores[scores > 0]  # Remove zero probabilities
        if len(scores) == 0:
            return 0.0
        return -np.sum(scores * np.log(scores))

    def fuse_emotions(self, facial_emotions: Dict[str, float], 
                     vocal_emotions: Dict[str, float], 
                     method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE) -> Dict[str, float]:
        """Fuse facial and vocal emotions using specified method"""
        try:
            if method == FusionMethod.WEIGHTED_AVERAGE:
                fused_emotions = self.weighted_average_fusion(facial_emotions, vocal_emotions)
            elif method == FusionMethod.DECISION_LEVEL:
                fused_emotions = self.decision_level_fusion(facial_emotions, vocal_emotions)
            else:
                fused_emotions = self.weighted_average_fusion(facial_emotions, vocal_emotions)
            
            # Log fusion result
            self._log_fusion(facial_emotions, vocal_emotions, fused_emotions, method)
            
            return fused_emotions
            
        except Exception as e:
            self.logger.error(f"Emotion fusion error: {e}")
            # Fallback: return facial emotions or neutral
            if facial_emotions:
                return facial_emotions
            return {"neutral": 1.0}

    def get_emotional_summary(self, facial_emotions: Dict[str, float], 
                            vocal_emotions: Dict[str, float],
                            physical_state: Dict = None) -> Dict:
        """Generate comprehensive emotional state summary"""
        # Fuse emotions
        fused_emotions = self.fuse_emotions(facial_emotions, vocal_emotions)
        
        # Get dominant emotion
        dominant_emotion, confidence = self._get_dominant_emotion(fused_emotions)
        
        # Calculate wellbeing score
        wellbeing_score = self.calculate_emotional_wellbeing(fused_emotions)
        
        # Detect critical states
        requires_intervention, critical_emotion = self.detect_critical_state(fused_emotions, physical_state)
        
        # Determine risk level
        risk_level = self.determine_risk_level(fused_emotions, confidence, physical_state)
        
        summary = {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "wellbeing_score": wellbeing_score,
            "fused_emotions": fused_emotions,
            "requires_intervention": requires_intervention,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }
        
        if requires_intervention:
            summary["critical_emotion"] = critical_emotion
        
        return summary

    def calculate_emotional_wellbeing(self, emotions: Dict[str, float]) -> float:
        """Calculate emotional wellbeing score (0-100)"""
        # Positive emotions contribute positively
        positive_emotions = emotions.get('happy', 0) + emotions.get('neutral', 0) * 0.5
        
        # Negative emotions contribute negatively
        negative_emotions = (emotions.get('sad', 0) + emotions.get('angry', 0) + 
                           emotions.get('fear', 0))
        
        # Calculate score (0-100 scale)
        raw_score = (positive_emotions - negative_emotions + 1) * 50
        wellbeing_score = max(0, min(100, raw_score))
        
        return wellbeing_score

    def detect_critical_state(self, emotions: Dict[str, float], 
                            physical_state: Dict = None) -> Tuple[bool, str]:
        """Detect if emotional state requires intervention"""
        critical_threshold = 0.7
        high_risk_emotions = ['angry', 'fear', 'sad']
        
        # Check emotional criteria
        for emotion in high_risk_emotions:
            if emotions.get(emotion, 0) > critical_threshold:
                return True, emotion
        
        # Check physical criteria if available
        if physical_state:
            if (physical_state.get('fatigue') == 'high' and 
                emotions.get('sad', 0) > 0.5):
                return True, 'fatigue_depression'
        
        return False, ""

    def determine_risk_level(self, emotions: Dict[str, float], confidence: float,
                           physical_state: Dict = None) -> str:
        """Determine overall risk level"""
        high_risk_score = sum(emotions.get(emotion, 0) for emotion in ['angry', 'fear', 'sad'])
        
        if high_risk_score > 0.8 and confidence > self.confidence_thresholds['high']:
            return "high"
        elif high_risk_score > 0.6 and confidence > self.confidence_thresholds['medium']:
            return "medium"
        elif high_risk_score > 0.4:
            return "low"
        else:
            return "normal"

    def _log_fusion(self, facial_emotions: Dict, vocal_emotions: Dict, 
                   fused_emotions: Dict, method: FusionMethod):
        """Log fusion process and results"""
        fusion_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method.value,
            'facial_input': facial_emotions,
            'vocal_input': vocal_emotions,
            'fused_output': fused_emotions,
            'dominant_emotion': self._get_dominant_emotion(fused_emotions)[0],
            'confidence': self._get_dominant_emotion(fused_emotions)[1]
        }
        
        self.fusion_history.append(fusion_entry)
        
        # Keep only last 100 entries
        if len(self.fusion_history) > 100:
            self.fusion_history = self.fusion_history[-100:]

    def get_fusion_analysis_report(self) -> Dict:
        """Generate fusion analysis report"""
        if not self.fusion_history:
            return {"status": "no_data"}
        
        recent_fusions = self.fusion_history[-20:]
        
        report = {
            "total_fusions": len(self.fusion_history),
            "recent_dominant_emotions": [f['dominant_emotion'] for f in recent_fusions],
            "average_confidence": np.mean([f['confidence'] for f in recent_fusions]),
            "modality_agreement_rate": self._calculate_modality_agreement_rate(),
            "fusion_trend": self._calculate_fusion_trend(),
            "timestamp": datetime.now().isoformat()
        }
        
        return report

    def _calculate_modality_agreement_rate(self) -> float:
        """Calculate how often facial and vocal modalities agree"""
        if len(self.fusion_history) < 2:
            return 0.0
        
        agreements = 0
        total_comparisons = 0
        
        for fusion in self.fusion_history[-50:]:
            facial_dominant = self._get_dominant_emotion(fusion['facial_input'])[0]
            vocal_dominant = self._get_dominant_emotion(fusion['vocal_input'])[0]
            
            # Simple agreement check
            if (facial_dominant in ['happy', 'neutral'] and vocal_dominant in ['happy', 'calm', 'neutral']) or \
               (facial_dominant in ['sad', 'angry', 'fear'] and vocal_dominant in ['sad', 'angry', 'fearful']):
                agreements += 1
            
            total_comparisons += 1
        
        return agreements / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_fusion_trend(self) -> Dict[str, float]:
        """Calculate emotional trend from fusion history"""
        if len(self.fusion_history) < 5:
            return {}
        
        recent_fusions = self.fusion_history[-10:]
        
        trend = {}
        for emotion in self.emotion_mapping['common']:
            scores = [f['fused_output'].get(emotion, 0) for f in recent_fusions]
            if scores:
                trend[emotion] = np.mean(scores)
        
        return trend
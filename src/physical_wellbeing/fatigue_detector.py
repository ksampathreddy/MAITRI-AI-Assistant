import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import dlib
import math

class FatigueDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize dlib face detector and shape predictor
        try:
            self.detector = dlib.get_frontal_face_detector()
            # You'll need to download shape_predictor_68_face_landmarks.dat
            # from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        except:
            self.logger.warning("Dlib not available, using OpenCV fallback")
            self.detector = None
            self.predictor = None
        
        # Fatigue detection parameters
        self.fatigue_history = []
        self.eye_aspect_ratio_threshold = 0.25
        self.mouth_aspect_ratio_threshold = 0.6
        self.blink_threshold = 0.2
        
        # Fatigue indicators
        self.fatigue_indicators = {
            'eye_closures': 0,
            'yawns': 0,
            'head_movements': 0,
            'blink_rate': 0,
            'last_update': datetime.now()
        }
        
        # State tracking
        self.eye_state_history = []
        self.mouth_state_history = []
        self.head_pose_history = []
        
        self.logger.info("Fatigue detector initialized")

    def detect_fatigue(self, frame: np.ndarray, emotional_state: Dict = None) -> Dict[str, any]:
        """Detect fatigue indicators from frame and emotional state"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self._detect_faces(gray)
            
            if not faces:
                return self._get_default_fatigue_analysis(emotional_state)
            
            # Use the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Extract facial landmarks
            landmarks = self._extract_landmarks(gray, face)
            
            if landmarks is None:
                return self._get_default_fatigue_analysis(emotional_state)
            
            # Calculate fatigue indicators
            fatigue_metrics = self._calculate_fatigue_metrics(landmarks, frame.shape)
            
            # Incorporate emotional state if available
            if emotional_state:
                fatigue_metrics = self._adjust_with_emotional_state(fatigue_metrics, emotional_state)
            
            # Determine fatigue level
            fatigue_level = self._assess_fatigue_level(fatigue_metrics)
            
            # Generate recommendations
            recommendations = self._generate_fatigue_recommendations(fatigue_metrics, fatigue_level)
            
            analysis = {
                'fatigue_level': fatigue_level,
                'fatigue_score': fatigue_metrics['overall_fatigue_score'],
                'metrics': fatigue_metrics,
                'recommendations': recommendations,
                'face_detected': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update fatigue history
            self._update_fatigue_history(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Fatigue detection error: {e}")
            return self._get_default_fatigue_analysis(emotional_state)

    def _detect_faces(self, gray_frame: np.ndarray) -> List:
        """Detect faces using available methods"""
        if self.detector is not None:
            # Use dlib detector
            return self.detector(gray_frame, 1)
        else:
            # Fallback to OpenCV Haar cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
            # Convert to dlib rectangles
            return [dlib.rectangle(x, y, x + w, y + h) for (x, y, w, h) in faces]

    def _extract_landmarks(self, gray_frame: np.ndarray, face_rect) -> Optional[List]:
        """Extract facial landmarks"""
        if self.predictor is not None:
            return self.predictor(gray_frame, face_rect)
        else:
            # Simple fallback - return None to use default analysis
            return None

    def _calculate_fatigue_metrics(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Calculate various fatigue metrics from facial landmarks"""
        metrics = {}
        
        # 1. Eye Aspect Ratio (EAR) for blink detection
        metrics['eye_aspect_ratio'] = self._calculate_eye_aspect_ratio(landmarks)
        
        # 2. Mouth Aspect Ratio (MAR) for yawning detection
        metrics['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(landmarks)
        
        # 3. Head pose estimation (simplified)
        metrics['head_pose_variability'] = self._calculate_head_pose_variability(landmarks)
        
        # 4. Blink rate estimation
        metrics['blink_rate'] = self._estimate_blink_rate(metrics['eye_aspect_ratio'])
        
        # 5. PERCLOS (Percentage of Eye Closure)
        metrics['perclos'] = self._calculate_perclos(metrics['eye_aspect_ratio'])
        
        # Overall fatigue score (weighted combination)
        weights = {
            'eye_aspect_ratio': 0.25,
            'mouth_aspect_ratio': 0.20,
            'head_pose_variability': 0.15,
            'blink_rate': 0.20,
            'perclos': 0.20
        }
        
        # Normalize and combine metrics
        overall_score = 0
        for metric, weight in weights.items():
            normalized_value = self._normalize_fatigue_metric(metric, metrics[metric])
            overall_score += normalized_value * weight
        
        metrics['overall_fatigue_score'] = min(1.0, overall_score)
        
        return metrics

    def _calculate_eye_aspect_ratio(self, landmarks) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            # Landmark indices for left eye
            left_eye_indices = [36, 37, 38, 39, 40, 41]
            # Landmark indices for right eye  
            right_eye_indices = [42, 43, 44, 45, 46, 47]
            
            def eye_aspect_ratio(eye_indices):
                # Vertical distances
                v1 = self._landmark_distance(landmarks, eye_indices[1], eye_indices[5])
                v2 = self._landmark_distance(landmarks, eye_indices[2], eye_indices[4])
                # Horizontal distance
                h = self._landmark_distance(landmarks, eye_indices[0], eye_indices[3])
                return (v1 + v2) / (2.0 * h)
            
            left_ear = eye_aspect_ratio(left_eye_indices)
            right_ear = eye_aspect_ratio(right_eye_indices)
            
            return (left_ear + right_ear) / 2.0
            
        except:
            return 0.3  # Default open eyes

    def _calculate_mouth_aspect_ratio(self, landmarks) -> float:
        """Calculate Mouth Aspect Ratio for yawning detection"""
        try:
            # Landmark indices for mouth
            mouth_indices = [48, 51, 54, 57]  # Left, top, right, bottom
            
            # Vertical distance
            vertical = self._landmark_distance(landmarks, mouth_indices[1], mouth_indices[3])
            # Horizontal distance  
            horizontal = self._landmark_distance(landmarks, mouth_indices[0], mouth_indices[2])
            
            return vertical / horizontal
            
        except:
            return 0.3  # Default closed mouth

    def _landmark_distance(self, landmarks, idx1: int, idx2: int) -> float:
        """Calculate distance between two landmarks"""
        if hasattr(landmarks, 'part'):  # dlib format
            point1 = landmarks.part(idx1)
            point2 = landmarks.part(idx2)
            return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        else:
            # Fallback calculation
            return 50.0  # Default distance

    def _calculate_head_pose_variability(self, landmarks) -> float:
        """Calculate head pose variability (simplified)"""
        try:
            # Use nose and eye positions to estimate head pose changes
            current_pose = self._estimate_head_pose(landmarks)
            self.head_pose_history.append(current_pose)
            
            # Keep only recent history
            if len(self.head_pose_history) > 10:
                self.head_pose_history.pop(0)
            
            # Calculate variability
            if len(self.head_pose_history) >= 2:
                variability = np.std([pose[0] for pose in self.head_pose_history])  # Use pitch variability
                return min(1.0, variability / 10.0)  # Normalize
            else:
                return 0.1
                
        except:
            return 0.1

    def _estimate_head_pose(self, landmarks) -> Tuple[float, float, float]:
        """Estimate head pose (pitch, yaw, roll) - simplified"""
        # This is a simplified version - in practice, you'd use proper head pose estimation
        return (0.0, 0.0, 0.0)  # Default neutral pose

    def _estimate_blink_rate(self, eye_aspect_ratio: float) -> float:
        """Estimate blink rate from eye aspect ratio"""
        # Track eye state
        is_eye_closed = eye_aspect_ratio < self.eye_aspect_ratio_threshold
        self.eye_state_history.append(is_eye_closed)
        
        # Keep only recent history
        if len(self.eye_state_history) > 30:  # ~1 second at 30fps
            self.eye_state_history.pop(0)
        
        # Calculate blink rate (blinks per minute)
        if len(self.eye_state_history) >= 10:
            blink_count = sum(1 for i in range(1, len(self.eye_state_history)) 
                           if self.eye_state_history[i] and not self.eye_state_history[i-1])
            time_window = len(self.eye_state_history) / 30.0  # Assuming 30fps
            blink_rate = (blink_count / time_window) * 60.0  # Blinks per minute
            
            # Normalize to 0-1 (typical range 15-20 blinks/minute is normal)
            return min(1.0, blink_rate / 30.0)
        else:
            return 0.15  # Default normal blink rate

    def _calculate_perclos(self, eye_aspect_ratio: float) -> float:
        """Calculate PERCLOS (Percentage of Eye Closure)"""
        eye_closure = 1.0 - (eye_aspect_ratio / 0.3)  # Normalize to 0-1
        return max(0.0, min(1.0, eye_closure))

    def _normalize_fatigue_metric(self, metric_name: str, value: float) -> float:
        """Normalize fatigue metrics to 0-1 scale where 1 indicates high fatigue"""
        normalization_rules = {
            'eye_aspect_ratio': lambda x: 1.0 - min(1.0, x / 0.3),  # Lower EAR = more fatigue
            'mouth_aspect_ratio': lambda x: min(1.0, x / 0.8),      # Higher MAR = more yawning
            'head_pose_variability': lambda x: x,                    # Already normalized
            'blink_rate': lambda x: min(1.0, x / 0.5),              # Higher blink rate = more fatigue
            'perclos': lambda x: x                                   # Already normalized
        }
        
        normalizer = normalization_rules.get(metric_name, lambda x: x)
        return normalizer(value)

    def _adjust_with_emotional_state(self, fatigue_metrics: Dict, emotional_state: Dict) -> Dict:
        """Adjust fatigue assessment based on emotional state"""
        adjusted_metrics = fatigue_metrics.copy()
        
        # Emotional states that might correlate with fatigue
        emotional_fatigue_indicators = {
            'sad': 0.3,
            'neutral': 0.1, 
            'tired': 0.4,
            'fatigue': 0.5
        }
        
        # Check if emotional state suggests fatigue
        emotional_hints = emotional_state.get('fused_emotions', {})
        emotional_fatigue_score = 0
        
        for emotion, weight in emotional_fatigue_indicators.items():
            if emotion in emotional_hints:
                emotional_fatigue_score += emotional_hints[emotion] * weight
        
        # Adjust overall fatigue score
        if emotional_fatigue_score > 0.2:
            adjustment = emotional_fatigue_score * 0.3  # Up to 30% adjustment
            adjusted_metrics['overall_fatigue_score'] = min(1.0, 
                adjusted_metrics['overall_fatigue_score'] + adjustment)
        
        return adjusted_metrics

    def _assess_fatigue_level(self, metrics: Dict[str, float]) -> str:
        """Assess overall fatigue level"""
        fatigue_score = metrics['overall_fatigue_score']
        
        if fatigue_score >= 0.8:
            return "severe"
        elif fatigue_score >= 0.6:
            return "high"
        elif fatigue_score >= 0.4:
            return "moderate"
        elif fatigue_score >= 0.2:
            return "mild"
        else:
            return "low"

    def _generate_fatigue_recommendations(self, metrics: Dict[str, float], fatigue_level: str) -> List[str]:
        """Generate personalized fatigue recommendations"""
        recommendations = []
        
        if fatigue_level in ["high", "severe"]:
            recommendations.append("🚨 Significant fatigue detected. Consider immediate rest.")
        
        # Specific recommendations based on metrics
        if metrics['eye_aspect_ratio'] < 0.2:
            recommendations.append("👁️ Frequent eye closures detected. Take an eye break.")
        
        if metrics['mouth_aspect_ratio'] > 0.7:
            recommendations.append("😮 Yawning detected. Your body needs rest.")
        
        if metrics['blink_rate'] > 0.4:
            recommendations.append("👀 High blink rate suggests eye strain. Look away from screens.")
        
        if metrics['perclos'] > 0.5:
            recommendations.append("🔄 High percentage of eye closure. Consider micro-naps.")
        
        # General recommendations based on fatigue level
        if fatigue_level == "severe":
            recommendations.extend([
                "🛌 CRITICAL: Immediate rest required",
                "⚕️ Notify medical team of severe fatigue",
                "📞 Contact ground control for support"
            ])
        elif fatigue_level == "high":
            recommendations.extend([
                "💤 Schedule extended sleep period",
                "🚫 Reduce cognitive workload immediately", 
                "🍎 Ensure proper nutrition and hydration"
            ])
        elif fatigue_level == "moderate":
            recommendations.extend([
                "⏰ Take scheduled 20-minute break",
                "💧 Hydrate and have a light snack",
                "🧘 Practice 5-minute relaxation exercise"
            ])
        
        # Space-specific recommendations
        recommendations.extend([
            "🚀 Microgravity fatigue: Use exercise equipment",
            "🌙 Maintain strict sleep-wake cycle",
            "💡 Adjust lighting for circadian rhythm support"
        ])
        
        return recommendations

    def _get_default_fatigue_analysis(self, emotional_state: Dict = None) -> Dict:
        """Return default analysis when face cannot be detected"""
        base_analysis = {
            'fatigue_level': 'unknown',
            'fatigue_score': 0.3,
            'metrics': {
                'eye_aspect_ratio': 0.25,
                'mouth_aspect_ratio': 0.3,
                'head_pose_variability': 0.1,
                'blink_rate': 0.15,
                'perclos': 0.2,
                'overall_fatigue_score': 0.3
            },
            'recommendations': [
                "No facial data available for fatigue analysis.",
                "Ensure face is visible and well-lit.",
                "Consider self-assessment of energy levels."
            ],
            'face_detected': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Adjust based on emotional state if available
        if emotional_state and 'fused_emotions' in emotional_state:
            emotions = emotional_state['fused_emotions']
            if emotions.get('tired', 0) > 0.5 or emotions.get('fatigue', 0) > 0.5:
                base_analysis['fatigue_level'] = 'moderate'
                base_analysis['fatigue_score'] = 0.6
                base_analysis['recommendations'].insert(0, 
                    "Emotional analysis suggests fatigue. Consider rest.")
        
        return base_analysis

    def _update_fatigue_history(self, analysis: Dict):
        """Update fatigue history for trend monitoring"""
        history_entry = {
            'timestamp': analysis['timestamp'],
            'fatigue_level': analysis['fatigue_level'],
            'fatigue_score': analysis['fatigue_score'],
            'face_detected': analysis['face_detected']
        }
        
        self.fatigue_history.append(history_entry)
        
        # Keep only last 100 analyses
        if len(self.fatigue_history) > 100:
            self.fatigue_history = self.fatigue_history[-100:]

    def get_fatigue_trend(self, window_size: int = 10) -> Dict[str, any]:
        """Get fatigue trend over recent analyses"""
        if len(self.fatigue_history) < window_size:
            return {'status': 'insufficient_data'}
        
        recent_analyses = self.fatigue_history[-window_size:]
        valid_analyses = [a for a in recent_analyses if a['face_detected']]
        
        if not valid_analyses:
            return {'status': 'no_valid_data'}
        
        trend = {
            'average_fatigue_score': np.mean([a['fatigue_score'] for a in valid_analyses]),
            'level_distribution': {},
            'trend_direction': self._calculate_fatigue_trend_direction(valid_analyses),
            'analysis_count': len(valid_analyses),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate level distribution
        for analysis in valid_analyses:
            level = analysis['fatigue_level']
            trend['level_distribution'][level] = trend['level_distribution'].get(level, 0) + 1
        
        return trend

    def _calculate_fatigue_trend_direction(self, analyses: List[Dict]) -> str:
        """Calculate whether fatigue is increasing or decreasing"""
        if len(analyses) < 3:
            return "stable"
        
        scores = [analysis['fatigue_score'] for analysis in analyses]
        
        # Simple trend calculation
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            
            if second_half - first_half > 0.1:
                return "increasing"
            elif first_half - second_half > 0.1:
                return "decreasing"
        
        return "stable"

    def get_fatigue_health_report(self) -> Dict:
        """Generate comprehensive fatigue health report"""
        trend = self.get_fatigue_trend()
        
        report = {
            'current_status': self.fatigue_history[-1] if self.fatigue_history else {},
            'recent_trend': trend,
            'total_analyses': len(self.fatigue_history),
            'detection_rate': len([a for a in self.fatigue_history if a['face_detected']]) / max(1, len(self.fatigue_history)),
            'average_fatigue_score': np.mean([a['fatigue_score'] for a in self.fatigue_history if a['face_detected']]) if any(a['face_detected'] for a in self.fatigue_history) else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
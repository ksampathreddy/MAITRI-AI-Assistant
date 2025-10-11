import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class PostureAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Posture analysis parameters
        self.posture_history = []
        self.good_posture_threshold = 0.8
        self.poor_posture_threshold = 0.4
        
        # Key landmarks for posture analysis
        self.landmark_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_ear': 7,
            'right_ear': 8,
            'left_hip': 23,
            'right_hip': 24
        }
        
        self.logger.info("Posture analyzer initialized")

    def analyze_posture(self, frame: np.ndarray) -> Dict[str, any]:
        """Analyze posture from frame and return comprehensive analysis"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return self._get_default_analysis()
            
            # Extract key landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Calculate posture metrics
            posture_metrics = self._calculate_posture_metrics(landmarks, frame.shape)
            
            # Determine posture quality
            posture_quality = self._assess_posture_quality(posture_metrics)
            
            # Generate recommendations
            recommendations = self._generate_posture_recommendations(posture_metrics, posture_quality)
            
            analysis = {
                'posture_quality': posture_quality,
                'posture_score': posture_metrics['overall_score'],
                'metrics': posture_metrics,
                'recommendations': recommendations,
                'landmarks_detected': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log analysis
            self._log_posture_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Posture analysis error: {e}")
            return self._get_default_analysis()

    def _calculate_posture_metrics(self, landmarks: List, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Calculate various posture metrics from landmarks"""
        frame_height, frame_width = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_coords = {}
        for name, index in self.landmark_indices.items():
            landmark = landmarks[index]
            landmark_coords[name] = {
                'x': landmark.x * frame_width,
                'y': landmark.y * frame_height,
                'visibility': landmark.visibility
            }
        
        metrics = {}
        
        # 1. Shoulder alignment (horizontal)
        left_shoulder = landmark_coords['left_shoulder']
        right_shoulder = landmark_coords['right_shoulder']
        shoulder_height_diff = abs(left_shoulder['y'] - right_shoulder['y'])
        metrics['shoulder_alignment'] = max(0, 1 - (shoulder_height_diff / 50))  # Normalize
        
        # 2. Head position relative to shoulders
        nose = landmark_coords['nose']
        shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        head_shoulder_offset = abs(nose['y'] - shoulder_center_y)
        metrics['head_position'] = max(0, 1 - (head_shoulder_offset / 100))
        
        # 3. Spinal alignment (shoulders to hips)
        left_hip = landmark_coords['left_hip']
        right_hip = landmark_coords['right_hip']
        
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        hip_center_x = (left_hip['x'] + right_hip['x']) / 2
        
        spinal_alignment = abs(shoulder_center_x - hip_center_x)
        metrics['spinal_alignment'] = max(0, 1 - (spinal_alignment / 30))
        
        # 4. Forward head posture
        ear_shoulder_distance = abs(landmark_coords['left_ear']['y'] - left_shoulder['y'])
        metrics['forward_head'] = max(0, 1 - (ear_shoulder_distance / 150))
        
        # 5. Overall posture score (weighted average)
        weights = {
            'shoulder_alignment': 0.25,
            'head_position': 0.25,
            'spinal_alignment': 0.30,
            'forward_head': 0.20
        }
        
        overall_score = 0
        for metric, weight in weights.items():
            overall_score += metrics[metric] * weight
        
        metrics['overall_score'] = overall_score
        
        return metrics

    def _assess_posture_quality(self, metrics: Dict[str, float]) -> str:
        """Assess overall posture quality"""
        overall_score = metrics['overall_score']
        
        if overall_score >= self.good_posture_threshold:
            return "excellent"
        elif overall_score >= 0.7:
            return "good"
        elif overall_score >= 0.5:
            return "fair"
        elif overall_score >= self.poor_posture_threshold:
            return "poor"
        else:
            return "very_poor"

    def _generate_posture_recommendations(self, metrics: Dict[str, float], posture_quality: str) -> List[str]:
        """Generate personalized posture recommendations"""
        recommendations = []
        
        if posture_quality in ["poor", "very_poor"]:
            recommendations.append("⚠️ Significant posture issues detected. Consider immediate adjustment.")
        
        # Specific recommendations based on metrics
        if metrics['shoulder_alignment'] < 0.7:
            recommendations.append("🔧 Shoulders are uneven. Try to level them horizontally.")
        
        if metrics['head_position'] < 0.6:
            recommendations.append("📏 Head is forward. Bring ears in line with shoulders.")
        
        if metrics['spinal_alignment'] < 0.7:
            recommendations.append("🦴 Spinal alignment needs improvement. Sit/stand straight.")
        
        if metrics['forward_head'] < 0.5:
            recommendations.append("👤 Forward head posture detected. Chin tucked, ears over shoulders.")
        
        # General recommendations based on quality
        if posture_quality in ["good", "excellent"]:
            recommendations.append("✅ Great posture! Maintain this position.")
        elif posture_quality == "fair":
            recommendations.append("💡 Minor adjustments needed for optimal posture.")
        
        # Space-specific recommendations
        recommendations.extend([
            "🚀 Remember microgravity posture: Use handholds for support",
            "🪑 In seated position: Keep feet flat, back supported",
            "⏰ Take posture breaks every 30 minutes"
        ])
        
        return recommendations

    def draw_posture_overlay(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        """Draw posture analysis overlay on frame"""
        try:
            overlay = frame.copy()
            height, width = overlay.shape[:2]
            
            # Create semi-transparent overlay
            overlay_bg = np.zeros((height, width, 3), dtype=np.uint8)
            alpha = 0.8
            
            # Color based on posture quality
            color_map = {
                "excellent": (0, 255, 0),      # Green
                "good": (0, 200, 0),           # Light Green
                "fair": (0, 255, 255),         # Yellow
                "poor": (0, 165, 255),         # Orange
                "very_poor": (0, 0, 255)       # Red
            }
            
            color = color_map.get(analysis['posture_quality'], (255, 255, 255))
            overlay_bg[:] = color
            cv2.addWeighted(overlay_bg, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Add posture information
            y_offset = 30
            line_height = 25
            
            # Posture quality
            quality_text = f"Posture: {analysis['posture_quality'].replace('_', ' ').title()}"
            cv2.putText(overlay, quality_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += line_height
            
            # Posture score
            score_text = f"Score: {analysis['posture_score']:.1%}"
            cv2.putText(overlay, score_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
            
            # Key metrics
            metrics = analysis['metrics']
            cv2.putText(overlay, f"Shoulders: {metrics['shoulder_alignment']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height - 5
            
            cv2.putText(overlay, f"Head: {metrics['head_position']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height - 5
            
            cv2.putText(overlay, f"Spine: {metrics['spinal_alignment']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"Posture overlay error: {e}")
            return frame

    def _get_default_analysis(self) -> Dict:
        """Return default analysis when posture cannot be detected"""
        return {
            'posture_quality': 'unknown',
            'posture_score': 0.5,
            'metrics': {
                'shoulder_alignment': 0.5,
                'head_position': 0.5,
                'spinal_alignment': 0.5,
                'forward_head': 0.5,
                'overall_score': 0.5
            },
            'recommendations': [
                "No posture data available. Ensure upper body is visible.",
                "Move to a well-lit area for better detection.",
                "Face the camera directly for posture analysis."
            ],
            'landmarks_detected': False,
            'timestamp': datetime.now().isoformat()
        }

    def _log_posture_analysis(self, analysis: Dict):
        """Log posture analysis for trend monitoring"""
        log_entry = {
            'timestamp': analysis['timestamp'],
            'posture_quality': analysis['posture_quality'],
            'posture_score': analysis['posture_score'],
            'landmarks_detected': analysis['landmarks_detected']
        }
        
        self.posture_history.append(log_entry)
        
        # Keep only last 100 analyses
        if len(self.posture_history) > 100:
            self.posture_history = self.posture_history[-100:]

    def get_posture_trend(self, window_size: int = 10) -> Dict[str, any]:
        """Get posture trend over recent analyses"""
        if len(self.posture_history) < window_size:
            return {'status': 'insufficient_data'}
        
        recent_analyses = self.posture_history[-window_size:]
        valid_analyses = [a for a in recent_analyses if a['landmarks_detected']]
        
        if not valid_analyses:
            return {'status': 'no_valid_data'}
        
        trend = {
            'average_score': np.mean([a['posture_score'] for a in valid_analyses]),
            'quality_distribution': {},
            'improvement_trend': self._calculate_improvement_trend(valid_analyses),
            'analysis_count': len(valid_analyses),
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate quality distribution
        for analysis in valid_analyses:
            quality = analysis['posture_quality']
            trend['quality_distribution'][quality] = trend['quality_distribution'].get(quality, 0) + 1
        
        return trend

    def _calculate_improvement_trend(self, analyses: List[Dict]) -> str:
        """Calculate whether posture is improving or deteriorating"""
        if len(analyses) < 3:
            return "stable"
        
        # Use linear regression to determine trend
        scores = [analysis['posture_score'] for analysis in analyses]
        x = np.arange(len(scores))
        
        # Simple trend calculation
        if len(scores) >= 2:
            slope = (scores[-1] - scores[0]) / len(scores)
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "deteriorating"
        
        return "stable"

    def get_posture_health_report(self) -> Dict:
        """Generate comprehensive posture health report"""
        trend = self.get_posture_trend()
        
        report = {
            'current_status': self.posture_history[-1] if self.posture_history else {},
            'recent_trend': trend,
            'total_analyses': len(self.posture_history),
            'detection_rate': len([a for a in self.posture_history if a['landmarks_detected']]) / max(1, len(self.posture_history)),
            'average_posture_score': np.mean([a['posture_score'] for a in self.posture_history if a['landmarks_detected']]) if any(a['landmarks_detected'] for a in self.posture_history) else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return report

    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
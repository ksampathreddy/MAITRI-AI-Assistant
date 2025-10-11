import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Tuple, List, Dict  # Added Dict import
import os

class CameraHandler:
    def __init__(self, camera_index: int = 0, frame_width: int = 640, frame_height: int = 480):
        self.logger = logging.getLogger(__name__)
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = None
        self.is_initialized = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.is_capturing = False
        
        # Camera status tracking
        self.camera_status = {
            'frames_captured': 0,
            'last_capture_time': None,
            'error_count': 0,
            'fps': 0
        }
        
        self.initialize_camera()

    def initialize_camera(self) -> bool:
        """Initialize camera with specified settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Could not open camera at index {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                self.logger.error("Camera test capture failed")
                self.cap.release()
                return False
            
            self.is_initialized = True
            self.logger.info(f"Camera initialized successfully: {self.frame_width}x{self.frame_height}")
            
            # Start continuous capture
            self.start_continuous_capture()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def start_continuous_capture(self):
        """Start continuous frame capture in separate thread"""
        if not self.is_initialized:
            self.logger.error("Camera not initialized, cannot start capture")
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info("Continuous camera capture started")

    def _capture_loop(self):
        """Continuous frame capture loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_capturing and self.is_initialized:
            try:
                ret, frame = self.cap.read()
                
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    frame_count += 1
                    self.camera_status['frames_captured'] = frame_count
                    self.camera_status['last_capture_time'] = time.time()
                    
                    # Calculate FPS every second
                    current_time = time.time()
                    if current_time - start_time >= 1.0:
                        self.camera_status['fps'] = frame_count / (current_time - start_time)
                        frame_count = 0
                        start_time = current_time
                
                else:
                    self.logger.warning("Frame capture failed")
                    self.camera_status['error_count'] += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                self.camera_status['error_count'] += 1
                time.sleep(0.1)  # Longer delay on error

    def capture_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from camera"""
        if not self.is_initialized or not self.is_capturing:
            self.logger.warning("Camera not ready for capture")
            return None
        
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        
        return None

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame directly (bypassing continuous capture)"""
        if not self.is_initialized:
            self.logger.error("Camera not initialized")
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.logger.error("Single frame capture failed")
                return None
        except Exception as e:
            self.logger.error(f"Single frame capture error: {e}")
            return None

    def save_frame(self, frame: np.ndarray, filename: str = None) -> str:
        """Save frame to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"data/camera/frame_{timestamp}.jpg"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            cv2.imwrite(filename, frame)
            self.logger.info(f"Frame saved: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Frame save error: {e}")
            return ""

    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """Get current frame with timestamp"""
        frame = self.capture_frame()
        timestamp = time.time()
        return frame, timestamp

    def preprocess_frame(self, frame: np.ndarray, 
                        target_size: Tuple[int, int] = None) -> np.ndarray:
        """Preprocess frame for analysis"""
        if target_size is None:
            target_size = (self.frame_width, self.frame_height)
        
        try:
            # Resize if needed
            if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                frame = cv2.resize(frame, target_size)
            
            # Apply basic enhancements
            # Convert to LAB color space for better lighting normalization
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_planes = list(cv2.split(lab))
            
            # Apply CLAHE to L-channel for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            
            # Merge back and convert to BGR
            lab = cv2.merge(lab_planes)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Optional: Gaussian blur for noise reduction
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing error: {e}")
            return frame

    def draw_analysis_overlay(self, frame: np.ndarray, analysis_data: Dict) -> np.ndarray:
        """Draw analysis results overlay on frame"""
        try:
            overlay = frame.copy()
            height, width = overlay.shape[:2]
            
            # Create semi-transparent overlay
            overlay_color = np.zeros((height, width, 3), dtype=np.uint8)
            alpha = 0.7  # Transparency
            
            # Color based on emotion or risk level
            if 'risk_level' in analysis_data:
                color_map = {
                    'normal': (0, 255, 0),      # Green
                    'low': (0, 255, 255),       # Yellow
                    'medium': (0, 165, 255),    # Orange
                    'high': (0, 0, 255)         # Red
                }
                color = color_map.get(analysis_data['risk_level'], (255, 255, 255))
            else:
                color = (0, 255, 0)  # Default green
            
            overlay_color[:] = color
            cv2.addWeighted(overlay_color, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Add analysis information
            y_offset = 30
            line_height = 25
            
            # Dominant emotion
            if 'dominant_emotion' in analysis_data:
                emotion = analysis_data['dominant_emotion']
                confidence = analysis_data.get('confidence', 0)
                text = f"Emotion: {emotion} ({confidence:.2f})"
                cv2.putText(overlay, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += line_height
            
            # Wellbeing score
            if 'wellbeing_score' in analysis_data:
                score = analysis_data['wellbeing_score']
                color = (0, 255, 0) if score > 70 else (0, 255, 255) if score > 40 else (0, 0, 255)
                text = f"Wellbeing: {score:.1f}/100"
                cv2.putText(overlay, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += line_height
            
            # Risk level
            if 'risk_level' in analysis_data:
                risk = analysis_data['risk_level']
                color_map = {'normal': (0, 255, 0), 'low': (0, 255, 255), 
                           'medium': (0, 165, 255), 'high': (0, 0, 255)}
                color = color_map.get(risk, (255, 255, 255))
                text = f"Risk: {risk.upper()}"
                cv2.putText(overlay, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += line_height
            
            # Timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(overlay, timestamp, (width - 200, height - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # MAITRI logo/text
            cv2.putText(overlay, "MAITRI AI Assistant", (10, height - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return overlay
            
        except Exception as e:
            self.logger.error(f"Overlay drawing error: {e}")
            return frame

    def get_camera_status(self) -> Dict:
        """Get current camera status"""
        status = self.camera_status.copy()
        status.update({
            'initialized': self.is_initialized,
            'capturing': self.is_capturing,
            'camera_index': self.camera_index,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'current_frame_available': self.current_frame is not None
        })
        return status

    def test_camera_health(self) -> Dict:
        """Perform camera health test"""
        health_report = {
            'timestamp': time.time(),
            'camera_initialized': self.is_initialized,
            'continuous_capture_running': self.is_capturing,
            'test_frames_captured': 0,
            'test_errors': 0,
            'resolution_test': 'pending',
            'fps_test': 'pending'
        }
        
        if not self.is_initialized:
            health_report['status'] = 'failed'
            return health_report
        
        # Test frame capture
        test_frames = []
        for i in range(5):  # Capture 5 test frames
            frame = self.capture_single_frame()
            if frame is not None:
                test_frames.append(frame)
                health_report['test_frames_captured'] += 1
            else:
                health_report['test_errors'] += 1
        
        # Resolution test
        if test_frames:
            first_frame = test_frames[0]
            actual_resolution = f"{first_frame.shape[1]}x{first_frame.shape[0]}"
            expected_resolution = f"{self.frame_width}x{self.frame_height}"
            health_report['resolution_test'] = 'passed' if actual_resolution == expected_resolution else 'failed'
            health_report['actual_resolution'] = actual_resolution
        
        # FPS test
        if self.camera_status['fps'] > 10:  # Reasonable FPS threshold
            health_report['fps_test'] = 'passed'
        else:
            health_report['fps_test'] = 'warning'
        
        # Overall status
        if (health_report['test_frames_captured'] >= 3 and 
            health_report['resolution_test'] == 'passed'):
            health_report['status'] = 'healthy'
        else:
            health_report['status'] = 'degraded'
        
        return health_report

    def release(self):
        """Release camera resources"""
        self.is_capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.logger.info("Camera released")
        
        self.is_initialized = False

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.release()
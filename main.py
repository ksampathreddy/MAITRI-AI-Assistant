import cv2
import pyaudio
import wave
import threading
import time
import json
from datetime import datetime
import logging
from src.utils.config import EmotionConfig, ConversationConfig, SystemConfig
from src.emotion_detection.facial_emotion import FacialEmotionDetector
from src.emotion_detection.vocal_emotion import VocalEmotionDetector
from src.emotion_detection.multimodal_fusion import MultimodalEmotionFusion
from src.conversation_engine.dialogue_manager import DialogueManager
from src.monitoring.alert_system import AlertSystem

class MAITRIAssistant:
    def __init__(self):
        # Initialize configuration
        self.emotion_config = EmotionConfig()
        self.conversation_config = ConversationConfig()
        self.system_config = SystemConfig()
        
        # Initialize components
        self.facial_detector = FacialEmotionDetector(self.emotion_config.FER_MODEL_PATH)
        self.vocal_detector = VocalEmotionDetector(self.emotion_config.VOCAL_MODEL_PATH)
        self.emotion_fusion = MultimodalEmotionFusion()
        self.dialogue_manager = DialogueManager(self.conversation_config)
        self.alert_system = AlertSystem()
        
        # Initialize audio recording
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        
        # State management
        self.is_recording = False
        self.is_monitoring = False
        self.current_emotional_state = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.system_config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('maitri_assistant.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_emotional_monitoring(self):
        """Start continuous emotional monitoring"""
        self.is_monitoring = True
        self.logger.info("Starting emotional monitoring...")
        
        # Start video monitoring in separate thread
        video_thread = threading.Thread(target=self._video_monitoring_loop)
        video_thread.daemon = True
        video_thread.start()
        
        # Start audio monitoring in separate thread
        audio_thread = threading.Thread(target=self._audio_monitoring_loop)
        audio_thread.daemon = True
        audio_thread.start()
    
    def _video_monitoring_loop(self):
        """Continuous video monitoring loop"""
        cap = cv2.VideoCapture(0)
        
        while self.is_monitoring:
            ret, frame = cap.read()
            if ret:
                try:
                    facial_emotions = self.facial_detector.predict_emotion(frame)
                    self.current_emotional_state['facial'] = facial_emotions
                    
                    # Display emotion on frame (for demo)
                    dominant_emotion, confidence = self.facial_detector.get_dominant_emotion(facial_emotions)
                    cv2.putText(frame, f"Emotion: {dominant_emotion} ({confidence:.2f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('MAITRI - Emotional Monitoring', frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error in video monitoring: {e}")
            
            time.sleep(2)  # Process every 2 seconds
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _audio_monitoring_loop(self):
        """Continuous audio monitoring loop"""
        while self.is_monitoring:
            try:
                # Record audio for 5 seconds
                audio_file = self._record_audio(5)
                vocal_emotions = self.vocal_detector.predict_emotion(audio_file)
                self.current_emotional_state['vocal'] = vocal_emotions
                
            except Exception as e:
                self.logger.error(f"Error in audio monitoring: {e}")
            
            time.sleep(10)  # Process audio every 10 seconds
    
    def _record_audio(self, duration: int = 5) -> str:
        """Record audio for specified duration"""
        filename = f"data/audio/recording_{int(time.time())}.wav"
        
        stream = self.audio.open(format=self.audio_format, channels=self.channels,
                                rate=self.rate, input=True, frames_per_buffer=self.chunk)
        
        frames = []
        
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save recording
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        return filename
    
    def process_emotional_state(self) -> Dict:
        """Process and fuse emotional data"""
        facial_emotions = self.current_emotional_state.get('facial', {})
        vocal_emotions = self.current_emotional_state.get('vocal', {})
        
        # Fuse emotions
        fused_emotions = self.emotion_fusion.fuse_emotions(facial_emotions, vocal_emotions)
        emotional_summary = self.emotion_fusion.get_emotional_summary(fused_emotions)
        
        # Check for alerts
        if emotional_summary["requires_intervention"]:
            self.alert_system.send_alert(emotional_summary)
        
        return emotional_summary
    
    def generate_assistant_response(self, user_input: str = "") -> str:
        """Generate assistant response based on current emotional state"""
        emotional_summary = self.process_emotional_state()
        response = self.dialogue_manager.generate_response(emotional_summary, user_input)
        return response
    
    def start_interactive_session(self):
        """Start interactive session with the assistant"""
        self.logger.info("Starting MAITRI interactive session...")
        print("\n" + "="*50)
        print("🚀 MAITRI AI Assistant - Ready for Interaction")
        print("="*50)
        print("Type 'quit' to exit the session")
        print("Type 'status' to check emotional state")
        print("="*50)
        
        self.start_emotional_monitoring()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'status':
                    emotional_summary = self.process_emotional_state()
                    print(f"\n📊 Emotional Status:")
                    print(f"Dominant Emotion: {emotional_summary['dominant_emotion']}")
                    print(f"Confidence: {emotional_summary['confidence']:.2f}")
                    print(f"Wellbeing Score: {emotional_summary['wellbeing_score']:.1f}")
                    print(f"Risk Level: {emotional_summary['risk_level']}")
                    continue
                elif not user_input:
                    continue
                
                response = self.generate_assistant_response(user_input)
                print(f"MAITRI: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Error in interactive session: {e}")
                print("MAITRI: I'm experiencing some technical difficulties. Please try again.")
        
        self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop all monitoring activities"""
        self.is_monitoring = False
        self.audio.terminate()
        self.logger.info("MAITRI monitoring stopped")

def main():
    """Main function to run MAITRI assistant"""
    assistant = MAITRIAssistant()
    
    try:
        assistant.start_interactive_session()
    except Exception as e:
        logging.error(f"Error running MAITRI: {e}")
    finally:
        assistant.stop_monitoring()

if __name__ == "__main__":
    main()
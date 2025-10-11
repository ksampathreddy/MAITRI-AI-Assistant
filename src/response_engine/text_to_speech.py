import pyttsx3
import threading
import queue
import logging
from typing import Optional

class TextToSpeech:
    def __init__(self, rate=150, volume=0.8, voice_index=0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Set voice properties
        voices = self.engine.getProperty('voices')
        if voices and len(voices) > voice_index:
            self.engine.setProperty('voice', voices[voice_index].id)
        
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.logger = logging.getLogger(__name__)
        
        # Start speech thread
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
        self.logger.info("TTS engine initialized")
    
    def speak(self, text: str, wait: bool = False):
        """Add text to speech queue"""
        if not text or not text.strip():
            return
            
        self.speech_queue.put(text)
        
        if wait:
            while not self.speech_queue.empty() or self.is_speaking:
                threading.Event().wait(0.1)
    
    def _speech_worker(self):
        """Worker thread for speech synthesis"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                self.is_speaking = True
                
                self.engine.say(text)
                self.engine.runAndWait()
                
                self.is_speaking = False
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Speech synthesis error: {e}")
                self.is_speaking = False
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def get_available_voices(self):
        """Get list of available voices"""
        return self.engine.getProperty('voices')
    
    def cleanup(self):
        """Clean up TTS engine"""
        try:
            self.engine.stop()
        except:
            pass
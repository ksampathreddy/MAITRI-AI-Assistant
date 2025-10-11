import speech_recognition as sr
import pyaudio
import wave
import threading
import time
import numpy as np
from typing import Optional
import logging
import os

class VoiceHandler:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.logger = logging.getLogger(__name__)
        self.last_audio_file = None
        
        try:
            self.microphone = sr.Microphone(sample_rate=sample_rate)
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            self.logger.info("Voice handler initialized successfully")
        except Exception as e:
            self.logger.error(f"Voice handler initialization failed: {e}")
            self.microphone = None
    
    def listen_for_wakeword(self, wake_word: str = "maitri") -> Optional[str]:
        """Listen for wake word and return user input"""
        if self.microphone is None:
            self.logger.error("Microphone not available")
            return None
            
        try:
            with self.microphone as source:
                self.logger.debug("Listening for wake word...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=3)
            
            # Convert audio to text
            try:
                text = self.recognizer.recognize_google(audio).lower()
                self.logger.info(f"Recognized: {text}")
                
                if wake_word.lower() in text:
                    # Extract command after wake word
                    command = text.replace(wake_word.lower(), "").strip()
                    if command:
                        return command
                    else:
                        # If only wake word detected, listen for follow-up
                        return self._listen_for_followup()
                        
            except sr.UnknownValueError:
                self.logger.debug("Could not understand audio")
            except sr.RequestError as e:
                self.logger.error(f"Speech recognition error: {e}")
                
        except sr.WaitTimeoutError:
            pass  # No speech detected, continue listening
        
        return None
    
    def _listen_for_followup(self) -> Optional[str]:
        """Listen for follow-up command after wake word"""
        try:
            self.logger.debug("Listening for follow-up command...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Follow-up command: {text}")
            return text
            
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            self.logger.debug("No follow-up command detected")
            return "hello"  # Default greeting
        
        except sr.RequestError as e:
            self.logger.error(f"Follow-up recognition error: {e}")
            return None
    
    def record_audio(self, duration: int = 5) -> str:
        """Record audio for specified duration and save to file"""
        filename = f"data/audio/recording_{int(time.time())}.wav"
        
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            # Save recording
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            self.last_audio_file = filename
            return filename
            
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return ""

    def cleanup(self):
        """Clean up resources"""
        # SpeechRecognition handles cleanup automatically
        pass
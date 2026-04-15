import pyttsx3
import threading

engine = pyttsx3.init()

engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# 🔒 Global lock
tts_lock = threading.Lock()

def speak(text):
    try:
        with tts_lock:   # ✅ prevents multiple threads
            print("Speaking:", text)

            engine.stop()  # stop previous speech (important)
            engine.say(text)
            engine.runAndWait()

    except Exception as e:
        print("TTS Error:", e)
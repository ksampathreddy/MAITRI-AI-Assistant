import pyttsx3
import threading
import queue
import time

speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()

    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)

    while True:
        text = speech_queue.get()

        try:
            print("Speaking:", text)

            engine.say(text)
            engine.runAndWait()

            time.sleep(1)

        except Exception as e:
            print("TTS Error:", e)

        finally:
            speech_queue.task_done()


threading.Thread(target=tts_worker, daemon=True).start()


def speak(text):
    print("Queued:", text)
    speech_queue.put(text)
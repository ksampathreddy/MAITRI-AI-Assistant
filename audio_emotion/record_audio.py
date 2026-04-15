import sounddevice as sd
import numpy as np

def record_audio(duration=4, sr=22050):
    print("Recording...")

    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = audio.flatten()
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    print("AUDIO MEAN:", np.abs(audio).mean())

    return audio
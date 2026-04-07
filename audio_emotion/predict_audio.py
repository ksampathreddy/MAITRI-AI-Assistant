import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from audio_emotion.audio_model import AudioEmotionModel
from audio_emotion.feature_extraction import extract_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

model = AudioEmotionModel().to(DEVICE)
model.load_state_dict(
    torch.load("audio_emotion/models/ravdess_model.pth", map_location=DEVICE)
)
model.eval()

def record_audio():
    fs = 22050
    duration = 3

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    wav.write("temp.wav", fs, recording)

    return recording   # ✅ IMPORTANT

def predict_audio():
    record_audio()

    mfcc = extract_features("temp.wav")

    # ❗ Check silence (VERY IMPORTANT)
    if mfcc is None or np.mean(mfcc) == 0:
        return None   # 👈 signals no audio

    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        output = model(mfcc)
        _, pred = torch.max(output,1)

    return labels[pred.item()]
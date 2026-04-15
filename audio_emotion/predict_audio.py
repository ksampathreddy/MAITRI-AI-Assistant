import torch
import numpy as np
import soundfile as sf

from audio_emotion.audio_model import AudioEmotionModel
from audio_emotion.feature_extraction import extract_features
from audio_emotion.record_audio import record_audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioEmotionModel().to(DEVICE)

model.load_state_dict(
    torch.load("audio_emotion/models/ravdess_model.pth", map_location=DEVICE)
)

model.eval()

labels = [
    "neutral","calm","happy","sad",
    "angry","fear","disgust","surprise"
]

def predict_audio():
    try:
        audio = record_audio(duration=4)

        if audio is None:
            return "No audio detected"

        if np.abs(audio).mean() < 0.001:
            return "No audio detected"

        sf.write("temp.wav", audio, 22050)

        features = extract_features("temp.wav")

        if features is None:
            return "No audio detected"

        features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)

        print("Confidence:", probs.max().item())

        return labels[pred.item()]

    except Exception as e:
        print("Audio Error:", e)
        return "No audio detected"
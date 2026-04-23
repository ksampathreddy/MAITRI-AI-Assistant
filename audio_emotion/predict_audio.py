import torch
import numpy as np
import sounddevice as sd
import librosa

from audio_emotion.audio_model import AudioEmotionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = AudioEmotionModel().to(DEVICE)
model.load_state_dict(
    torch.load("audio_emotion/models/ravdess_model.pth", map_location=DEVICE)
)
model.eval()

labels = [
    "neutral", "calm", "happy", "sad",
    "angry", "fear", "disgust", "surprise"
]

SAMPLE_RATE = 22050
DURATION = 3


# =========================
# RECORD AUDIO (NEW)
# =========================
def record_audio():
    print("🎤 Recording NEW AUDIO...")

    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )

    sd.wait()

    audio = np.squeeze(audio)

    print("NEW AUDIO MEAN:", np.mean(audio))  # ✅ MUST CHANGE EVERY TIME

    return audio


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(audio):
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=40
    )

    if mfcc.shape[1] < 100:
        pad = 100 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :100]

    return mfcc


# =========================
# PREDICT
# =========================
def predict_audio():
    try:
        audio = record_audio()

        if audio is None or len(audio) == 0:
            return "No audio detected"

        if np.abs(audio).mean() < 0.005:
            return "No audio detected"

        features = extract_features(audio)

        features = torch.tensor(features).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)

        emotion = labels[pred.item()]
        confidence = probs.max().item()

        print("🎧 Emotion:", emotion)
        print("Confidence:", confidence)

        return emotion

    except Exception as e:
        print("Audio Error:", e)
        return "No audio detected"
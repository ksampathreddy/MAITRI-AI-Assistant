import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from audio_model import AudioEmotionModel
from feature_extraction import extract_features

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_map = {
    "01":0,"02":1,"03":2,"04":3,
    "05":4,"06":5,"07":6,"08":7
}

labels_list = [
    "neutral", "calm", "happy", "sad",
    "angry", "fear", "disgust", "surprise"
]

class RAVDESSDataset(Dataset):

    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        for actor in os.listdir(root_dir):
            path = os.path.join(root_dir, actor)
            if not os.path.isdir(path):
                continue

            for file in os.listdir(path):
                if not file.endswith(".wav"):
                    continue

                parts = file.split("-")
                if len(parts) < 3:
                    continue

                emotion = parts[2]
                if emotion not in emotion_map:
                    continue

                self.files.append(os.path.join(path, file))
                self.labels.append(emotion_map[emotion])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        while True:
            mfcc = extract_features(self.files[idx])
            if mfcc is not None:
                break
            idx = (idx + 1) % len(self.files)

        mfcc = torch.tensor(mfcc).unsqueeze(0).float()
        label = torch.tensor(self.labels[idx])

        return mfcc, label



dataset = RAVDESSDataset("data/ravdess")
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = AudioEmotionModel().to(DEVICE)

model.load_state_dict(
    torch.load("audio_emotion/models/ravdess_model.pth", map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully!")


all_preds = []
all_labels = []

with torch.no_grad():
    for mfcc, labels in test_loader:
        mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)

        outputs = model(mfcc)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print("\nFINAL MODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=labels_list))
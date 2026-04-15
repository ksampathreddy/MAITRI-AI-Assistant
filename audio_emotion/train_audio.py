import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from audio_model import AudioEmotionModel
from feature_extraction import extract_features
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_map = {
    "01":0,"02":1,"03":2,"04":3,
    "05":4,"06":5,"07":6,"08":7
}

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
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AudioEmotionModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 30

for epoch in range(epochs):
    total_loss = 0

    for mfcc, labels in loader:
        mfcc, labels = mfcc.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

os.makedirs("audio_emotion/models", exist_ok=True)
torch.save(model.state_dict(), "audio_emotion/models/ravdess_model.pth")

print("Model saved!")
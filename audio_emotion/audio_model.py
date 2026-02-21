import torch
import torch.nn as nn

class AudioEmotionModel(nn.Module):

    def __init__(self):
        super(AudioEmotionModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=64 * 10,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 8)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,3,1,2)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
import torch
import torch.nn as nn

class AudioEmotionModel(nn.Module):

    def __init__(self):
        super(AudioEmotionModel, self).__init__()

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

        # Reduce feature size
        self.pool = nn.AdaptiveAvgPool2d((8, 20))

        # LSTM input size = channels * freq
        self.lstm = nn.LSTM(
            input_size=64 * 8,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        # x: (B,1,40,174)

        x = self.cnn(x)
        x = self.pool(x)   # (B,64,8,20)

        # reshape for LSTM
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)      # (B,W,C,H)
        x = x.contiguous().view(B, W, C * H)  # (B,20,64*8)

        # LSTM
        x, _ = self.lstm(x)

        # take last timestep
        x = x[:, -1, :]

        x = self.fc(x)

        return x
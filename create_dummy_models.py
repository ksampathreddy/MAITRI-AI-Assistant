#!/usr/bin/env python3
"""
Create dummy model files for MAITRI
"""

import torch
import torch.nn as nn
from pathlib import Path

def create_dummy_models():
    """Create dummy model files"""
    print("🤖 Creating dummy model files...")
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Simple facial emotion model
    class DummyFacialModel(nn.Module):
        def __init__(self, num_classes=7):
            super(DummyFacialModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 12 * 12, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.pool = nn.MaxPool2d(2)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Simple voice emotion model
    class DummyVoiceModel(nn.Module):
        def __init__(self, input_size=180, num_emotions=7):
            super(DummyVoiceModel, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_emotions)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Create and save models
    facial_model = DummyFacialModel()
    voice_model = DummyVoiceModel()
    
    torch.save(facial_model.state_dict(), 'models/facial_emotion_model.pth')
    torch.save(voice_model.state_dict(), 'models/voice_emotion_model.pth')
    
    print("✅ Dummy models created successfully!")
    print("📁 Models saved in: models/")
    print("   - facial_emotion_model.pth")
    print("   - voice_emotion_model.pth")

if __name__ == "__main__":
    create_dummy_models()
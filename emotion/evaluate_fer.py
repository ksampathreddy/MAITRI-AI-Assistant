import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import FERModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.ImageFolder("data/fer2013/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=64)

model = FERModel().to(DEVICE)
model.load_state_dict(torch.load("emotion/models/fer_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
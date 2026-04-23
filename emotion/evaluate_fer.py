import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import FERModel
from sklearn.metrics import classification_report, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 🔹 Load Data
test_data = datasets.ImageFolder("data/fer2013/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=64)

# 🔹 Load Model
model = FERModel().to(DEVICE)
model.load_state_dict(torch.load("emotion/models/fer_model.pth", map_location=DEVICE))
model.eval()

# 🔹 Store predictions
all_preds = []
all_labels = []

# 🔹 Evaluation
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =========================
# 📊 METRICS
# =========================

accuracy = accuracy_score(all_labels, all_preds)

print(f"\nAccuracy: {accuracy*100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=test_data.classes)) 
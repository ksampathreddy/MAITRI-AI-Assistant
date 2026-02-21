import torch
import cv2
from emotion.utils import FERModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceEmotionDetector:
    def __init__(self):
        self.model = FERModel().to(DEVICE)
        self.model.load_state_dict(
            torch.load("emotion/models/fer_model.pth", map_location=DEVICE)
        )
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(48,48))

            face = face / 255.0
            face = (face - 0.5) / 0.5

            face = torch.tensor(face).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = self.model(face)
                _, pred = torch.max(output,1)

            return self.labels[pred.item()]

        return "No Face"
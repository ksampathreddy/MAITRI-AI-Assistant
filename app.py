from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import numpy as np

from emotion.face_emotion import FaceEmotionDetector
from audio_emotion.predict_audio import predict_audio
from text_emotion.predict_text import predict_text_emotion
from llm.response_generator import ResponseGenerator

app = Flask(__name__)

# ---------------- INIT MODELS ----------------
face_detector = FaceEmotionDetector()
response_generator = ResponseGenerator()

camera = cv2.VideoCapture(0)

latest_face_emotion = "Detecting..."
latest_audio_emotion = "Detecting..."
latest_text_emotion = None
latest_overall_emotion = "Detecting..."
latest_ai_response = "Waiting for emotional analysis..."

# ---------------- EMOTION SCORE MAP ----------------
emotion_score = {
    "angry": 1,
    "disgust": 1,
    "fear": 2,
    "fearful": 2,
    "sad": 3,
    "calm": 4,
    "neutral": 5,
    "happy": 7,
    "surprise": 6,
    "surprised": 6
}

# ---------------- CAMERA STREAM ----------------
def generate_frames():
    global latest_face_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        emotion = face_detector.predict(frame)
        latest_face_emotion = emotion

        cv2.putText(frame, emotion, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- AUDIO THREAD ----------------
def audio_loop():
    global latest_audio_emotion
    while True:
        try:
            latest_audio_emotion = predict_audio()
        except:
            latest_audio_emotion = "Audio Error"
        time.sleep(3)

# ---------------- OVERALL LOGIC ----------------
def compute_overall():
    scores = []

    if latest_face_emotion in emotion_score:
        scores.append(emotion_score[latest_face_emotion])

    if latest_audio_emotion in emotion_score:
        scores.append(emotion_score[latest_audio_emotion])

    if latest_text_emotion and latest_text_emotion in emotion_score:
        scores.append(emotion_score[latest_text_emotion])

    if not scores:
        return "No Data"

    avg_score = np.mean(scores)

    closest = min(emotion_score.items(),
                  key=lambda x: abs(x[1] - avg_score))[0]

    return closest

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotions')
def emotions():
    global latest_overall_emotion, latest_ai_response

    latest_overall_emotion = compute_overall()

    if latest_overall_emotion != "No Data":
        try:
            user_input = latest_text_emotion if latest_text_emotion else "No specific message provided."
            latest_ai_response = response_generator.generate(
                latest_overall_emotion,
                user_input
            )
        except:
            latest_ai_response = "I'm here with you. Let's take a calm breath together."

    return jsonify({
        "face": latest_face_emotion,
        "audio": latest_audio_emotion,
        "text": latest_text_emotion,
        "overall": latest_overall_emotion,
        "response": latest_ai_response
    })

@app.route('/text_emotion', methods=['POST'])
def text_emotion():
    global latest_text_emotion

    data = request.json
    text = data.get("text")

    if text.strip() == "":
        latest_text_emotion = None
    else:
        latest_text_emotion = predict_text_emotion(text)

    return jsonify({"text_emotion": latest_text_emotion})

# Start audio thread
threading.Thread(target=audio_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import numpy as np

from emotion.face_emotion import FaceEmotionDetector
from audio_emotion.predict_audio import predict_audio
from text_emotion.predict_text import predict_text_emotion
# from response_generator import ResponseGenerator
from llm.response_generator import ResponseGenerator

app = Flask(__name__)

# ===============================
# Initialize Systems
# ===============================
face_detector = FaceEmotionDetector()
response_generator = ResponseGenerator()

camera = cv2.VideoCapture(0)

latest_face_emotion = "Detecting..."
latest_audio_emotion = "Detecting..."
latest_text_emotion = None
latest_user_text = ""
latest_ai_response = "Waiting for emotional analysis..."

# ===============================
# Emotion Scoring Map
# ===============================
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

# ===============================
# Camera Streaming
# ===============================
def generate_frames():
    global latest_face_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        emotion = face_detector.predict(frame)
        latest_face_emotion = emotion

        cv2.putText(frame, emotion, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ===============================
# Audio Background Thread
# ===============================
def audio_loop():
    global latest_audio_emotion

    while True:
        try:
            latest_audio_emotion = predict_audio()
        except:
            latest_audio_emotion = "Audio Error"

        time.sleep(2)


# ===============================
# Compute Overall Emotion
# ===============================
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


# ===============================
# ROUTES
# ===============================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 🔹 ONLY returns emotional data (NO LLM here)
@app.route('/emotions')
def emotions():
    return jsonify({
        "face": latest_face_emotion,
        "audio": latest_audio_emotion,
        "text_emotion": latest_text_emotion,
        "overall": compute_overall(),
        "response": latest_ai_response
    })


# 🔹 Text emotion analysis
@app.route('/text_emotion', methods=['POST'])
def text_emotion():
    global latest_text_emotion, latest_user_text

    data = request.get_json(force=True)
    text = data.get("text", "")

    latest_user_text = text

    if text.strip() == "":
        latest_text_emotion = None
    else:
        latest_text_emotion = predict_text_emotion(text)

    print("USER TEXT RECEIVED:", latest_user_text)
    print("TEXT EMOTION:", latest_text_emotion)

    return jsonify({"text_emotion": latest_text_emotion})


import time

@app.route('/generate_response', methods=['POST'])
def generate_response():

    global latest_ai_response, latest_user_text

    print("Calling LLM...")
    start = time.time()

    try:
        # ✅ Get user text properly
        data = request.get_json(force=True)
        user_text = data.get("text", latest_user_text)

        print("INPUT TO LLM:", user_text)

        # ✅ Correct LLM call
        latest_ai_response = response_generator.generate(
            latest_face_emotion,
            latest_audio_emotion,
            latest_text_emotion,
            user_text   # ✅ FIXED
        )

        print("LLM RESPONSE:", latest_ai_response)

    except Exception as e:
        print("LLM ERROR:", e)
        latest_ai_response = "AI temporarily unavailable."

    end = time.time()
    print("Generation time:", round(end - start, 2), "seconds")

    return jsonify({"response": latest_ai_response})


# ===============================
# Start Background Threads
# ===============================
threading.Thread(target=audio_loop, daemon=True).start()


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
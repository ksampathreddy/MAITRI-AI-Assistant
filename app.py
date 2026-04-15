from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import numpy as np

from utils.alert_logger import send_alert, log_emotions, update_stress_history
from emotion.face_emotion import FaceEmotionDetector
from audio_emotion.predict_audio import predict_audio
from text_emotion.predict_text import predict_text_emotion
<<<<<<< HEAD
from llm.response_generator import ResponseGenerator
from tts_engine import speak

lock = threading.Lock()
emotion_history = []

CRITICAL_EMOTIONS = ["sad", "fear", "angry", "disgust"]
last_alert_time = 0
ALERT_COOLDOWN = 120  # seconds

def smooth_emotion(new_emotion):
    emotion_history.append(new_emotion)

    if len(emotion_history) > 5:
        emotion_history.pop(0)

    return max(set(emotion_history), key=emotion_history.count)

def fusion_decision(face, audio, text):

    emotion_to_vector = {
        "angry":      [1,0,0,0,0,0,0],
        "disgust":    [0,1,0,0,0,0,0],
        "fear":       [0,0,1,0,0,0,0],
        "happy":      [0,0,0,1,0,0,0],
        "sad":        [0,0,0,0,1,0,0],
        "surprise":   [0,0,0,0,0,1,0],
        "neutral":    [0,0,0,0,0,0,1]
    }

    labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]

    face_vec = emotion_to_vector.get(face, [0]*7)
    audio_vec = emotion_to_vector.get(audio, [0]*7)
    text_vec = emotion_to_vector.get(text, [0]*7)

    fused = np.array(face_vec)*0.4 + np.array(audio_vec)*0.3 + np.array(text_vec)*0.3

    return labels[np.argmax(fused)]

def check_critical_state(face, audio, text, overall):
    stress = response_generator._compute_stress(face, audio, text)

    if stress == "HIGH":
        return True

    if overall in CRITICAL_EMOTIONS:
        return True

    return False


def should_trigger_alert():
    global last_alert_time

    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN:
        return False

    last_alert_time = current_time
    return True
=======
# from response_generator import ResponseGenerator
from llm.response_generator import ResponseGenerator
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964

app = Flask(__name__)

# ===============================
# Initialize Systems
# ===============================
face_detector = FaceEmotionDetector()
response_generator = ResponseGenerator()

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not accessible")

latest_face_emotion = "Detecting..."
latest_audio_emotion = "No audio detected"
latest_text_emotion = None
latest_user_text = ""
latest_ai_response = "Waiting for emotional analysis..."

<<<<<<< HEAD
=======
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
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
def generate_frames():
    global latest_face_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        emotion = face_detector.predict(frame)

        with lock:
            latest_face_emotion = emotion

        cv2.putText(frame, emotion, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

<<<<<<< HEAD
=======

# ===============================
# Audio Background Thread
# ===============================
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
def audio_loop():
    global latest_audio_emotion

    while True:
        try:
<<<<<<< HEAD
            result = predict_audio()

            with lock:
                latest_audio_emotion = result if result else "No audio detected"

        except Exception as e:
            print("Audio Error:", e)
            with lock:
                latest_audio_emotion = "No audio detected"

        time.sleep(2)

=======
            latest_audio_emotion = predict_audio()
        except:
            latest_audio_emotion = "Audio Error"

        time.sleep(2)


# ===============================
# Compute Overall Emotion
# ===============================
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
def compute_overall():
    face = latest_face_emotion
    audio = latest_audio_emotion
    text = latest_text_emotion

    fused = fusion_decision(face, audio, text)
    return smooth_emotion(fused)

<<<<<<< HEAD
=======
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

>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
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
<<<<<<< HEAD

    with lock:
        face = latest_face_emotion
        audio = latest_audio_emotion
        text = latest_text_emotion
        response = latest_ai_response
        user_text = latest_user_text

    overall = compute_overall()

    if check_critical_state(face, audio, text, overall):
        if should_trigger_alert():
            send_alert(overall, user_text)

    log_emotions(face, audio, text, overall)

    return jsonify({
        "face": face,
        "audio": audio,
        "text_emotion": text,
        "overall": overall,
        "response": response
=======
    return jsonify({
        "face": latest_face_emotion,
        "audio": latest_audio_emotion,
        "text_emotion": latest_text_emotion,
        "overall": compute_overall(),
        "response": latest_ai_response
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
    })


# 🔹 Text emotion analysis
@app.route('/text_emotion', methods=['POST'])
def text_emotion():
    global latest_text_emotion, latest_user_text

    data = request.get_json(force=True)
    text = data.get("text", "")
<<<<<<< HEAD
=======

    latest_user_text = text
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964

    with lock:
        latest_user_text = text

        if text.strip() == "":
            latest_text_emotion = None
        else:
            latest_text_emotion = predict_text_emotion(text)

    print("USER TEXT:", text)
    print("TEXT EMOTION:", latest_text_emotion)

    print("USER TEXT RECEIVED:", latest_user_text)
    print("TEXT EMOTION:", latest_text_emotion)

    return jsonify({"text_emotion": latest_text_emotion})

<<<<<<< HEAD
@app.route('/generate_response', methods=['POST'])
def generate_response():

    global latest_ai_response

    try:
        with lock:
            face = latest_face_emotion
            audio = latest_audio_emotion
            text = latest_text_emotion
            user_text = latest_user_text

        response = response_generator.generate(face, audio, text, user_text)

        if response.strip() == "":
            response = "Stay calm and focused. You are not alone."

        with lock:
            if response != latest_ai_response:
                latest_ai_response = response
                threading.Thread(target=speak, args=(response,)).start()

    except Exception as e:
        print("LLM ERROR:", e)
        with lock:
            latest_ai_response = "AI temporarily unavailable."

    return jsonify({"response": latest_ai_response})

@app.route('/stress_history')
def stress_history_api():
    from utils.alert_logger import stress_history
    return jsonify({"history": stress_history})

=======

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
>>>>>>> 4ea826b18af4bdd97c90f261a667ddc9b09cb964
threading.Thread(target=audio_loop, daemon=True).start()


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
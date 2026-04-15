import time
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def send_alert(emotion, user_text):

    severity = "HIGH" if emotion in ["sad", "fear", "angry"] else "MODERATE"

    message = f"""
    CRITICAL ALERT
    Time: {time.ctime()}
    Severity: {severity}
    Emotion: {emotion}
    Message: {user_text}
    """
    print(message)
    try:
        with open("alerts.log", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print("Logging Error:", e)

def log_emotions(face, audio, text, overall):

    log = f"{time.ctime()} | Face:{face} | Audio:{audio} | Text:{text} | Overall:{overall}\n"

    with open(f"{LOG_DIR}/emotion_log.txt", "a") as f:
        f.write(log)

stress_history = []

def update_stress_history(overall):

    stress_history.append(overall)

    if len(stress_history) > 50:
        stress_history.pop(0)

    return stress_history   
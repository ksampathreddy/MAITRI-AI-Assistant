# 🚀 MAITRI – Multimodal AI Assistant for Astronauts

MAITRI is an AI-powered system designed to provide **real-time psychological and emotional support to astronauts** during space missions.
It uses **multimodal emotion detection (Face + Audio + Text)**, intelligent fusion, and an AI response system to monitor stress and generate supportive responses.

---

## 🧠 Key Features

* 🎥 **Face Emotion Detection** (Computer Vision)
* 🎤 **Audio Emotion Recognition** (CNN + LSTM)
* 💬 **Text Emotion Analysis** (Transformer-based model)
* 🔀 **Multimodal Fusion System**
* 🤖 **AI Response Generator (TinyLlama)**
* 🚨 **Critical Emotion Alert System (with cooldown)**
* 🔊 **Text-to-Speech (TTS) Output**
* 📊 **Stress Monitoring & Logging**
* 🛰️ **Ground Control Simulation (Alert Monitoring + Messaging)**

---

## 🏗️ System Architecture

```
User (Astronaut)
   ↓
Face + Audio + Text Input
   ↓
Emotion Detection Models
   ↓
Fusion Engine
   ↓
Stress Analysis
   ↓
AI Response + Alert System
   ↓
Ground Control Dashboard
```

---

## 🧪 Technologies Used

* **Python**
* **Flask** (Web Backend)
* **PyTorch** (Deep Learning)
* **Transformers (HuggingFace)** (Text Emotion + LLM)
* **OpenCV** (Face Detection)
* **Librosa** (Audio Processing)
* **NumPy, Scikit-learn**
* **pyttsx3 / TTS Engine**

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/MAITRI-AI-Assistant.git
cd MAITRI-AI-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🚨 Alert System

* Detects **high stress or negative emotions**
* Triggers alerts only after **2-minute cooldown**
* Logs alerts in:

```
utils/alerts.log
```

---

## 🧠 Emotion Classes

| Code | Emotion  |
| ---- | -------- |
| 01   | Neutral  |
| 02   | Calm     |
| 03   | Happy    |
| 04   | Sad      |
| 05   | Angry    |
| 06   | Fear     |
| 07   | Disgust  |
| 08   | Surprise |

---

## 🔥 Future Improvements

* 📊 Stress Trend Dashboard
* 📱 Mobile Alert System (WhatsApp/SMS)
* 🧠 Attention-Based Fusion Model
* 🎤 Advanced Voice (Human-like TTS)
* 🌐 Cloud Deployment
* 📡 Real-time communication system

---

## 🎯 Project Highlights

* Multimodal AI system
* Real-time processing
* Alert-based monitoring
* Hybrid AI + Human communication model
* Suitable for **space missions, healthcare, and defense**

---

## 📌 Use Cases

* 🛰️ Astronaut Mental Health Monitoring
* 🏥 Healthcare Emotional Support Systems
* 🧑‍💻 Remote Worker Stress Detection
* 🎓 Research in Multimodal AI

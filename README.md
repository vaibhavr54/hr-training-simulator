# 🧠 HR Training Simulator — AI-Powered Interview Evaluation System

### 📄 Overview

The **HR Training Simulator** is an AI-driven interview evaluation system that processes candidate video responses using:

* 🎤 **Whisper** for transcription
* 😊 **DeepFace + RetinaFace** for emotion detection
* 🧩 **OpenRouter LLM API** for scoring and next-question generation
* 🖥️ **Interactive Frontend Simulator** featuring:

  * AI Interviewer Avatar (voice + lip-sync)
  * Real-time audio & face monitoring
  * Live micro-feedback via speech
  * Automatic question narration
* 💾 **MongoDB** for storing scores, transcripts, emotions, and history
* ⚙️ **FastAPI** backend for processing, evaluation, and data management

---

## 🚀 Features

### ✅ **Speech-to-Text (Transcription)**

Uses OpenAI Whisper to convert spoken responses into accurate text.

### ✅ **Emotion Detection from Video**

Analyzes sampled video frames using DeepFace to determine dominant emotions and confidence scores.

### ✅ **AI-Based Interview Scoring**

Evaluates candidate responses across:

* Communication
* Confidence
* Structure
* Soft Skills

Also generates natural-language feedback.

### ✅ **Dynamic Question Generation**

Creates the next interview question based on previous performance and answer context.

### ✅ **Real-Time Frontend Analysis**

The updated frontend performs live monitoring while recording:

* **Audio loudness detection** → alerts when voice is too low
* **Face visibility detection** → alerts if face is not visible
* **Emotion cues** → detects positive or negative emotional trends
* Feedback is **spoken by the avatar** instead of shown as text

### ✅ **AI Interviewer Avatar**

The frontend now includes:

* Auto lip-sync animation while speaking
* Blinking & expression animations
* Voice output using browser SpeechSynthesis
* Spoken instructions, warnings, and motivation
* Visual pulse animation upon delivering feedback

### ✅ **Candidate History Management**

Stores:

* Past questions
* Transcripts
* Scores
* Emotional analysis
* Attempt timestamps

Viewable and sortable by newest/oldest.

---

## 🧩 Tech Stack

| Category               | Technology                      |
| ---------------------- | ------------------------------- |
| **Backend API**        | FastAPI                         |
| **Transcription**      | OpenAI Whisper                  |
| **Emotion Analysis**   | DeepFace + OpenCV               |
| **AI Text Scoring**    | OpenRouter API                  |
| **Database**           | MongoDB                         |
| **Frontend**           | HTML, CSS, JavaScript, Chart.js |
| **Live Avatar Voice**  | Web Speech API                  |
| **Environment Config** | python-dotenv                   |
| **Async Tasks**        | asyncio + ThreadPoolExecutor    |

---

## 🗂️ Project Structure

```
backend/
│
├── main.py                 # Main FastAPI application
├── .env                    # Environment variables (MONGO_URI, OPENROUTER_API_KEY)
├── requirements.txt        # Python dependencies
├── static/                 # Frontend UI (index.html + avatar logic)
├── uploads/                # Uploaded media files (auto-created)
└── venv/                   # Virtual environment
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/vaibhavr54/hr-training-simulator.git
cd hr-training-simulator/backend
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate       # (Windows)
source venv/bin/activate      # (Linux/Mac)
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🧾 Example `.env` File

```
MONGO_URI=mongodb://localhost:27017
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

The app runs at:

```
http://127.0.0.1:8000
```

---

## 🔍 API Endpoints

| Method   | Endpoint               | Description                           |
| -------- | ---------------------- | ------------------------------------- |
| **GET**  | `/`                    | Serves the frontend interview UI      |
| **POST** | `/upload-audio`        | Upload & analyze interview video      |
| **POST** | `/next-question`       | Generate next question dynamically    |
| **GET**  | `/history/{candidate}` | Fetch interview history from database |
| **GET**  | `/health`              | Server health status                  |

---

## 🧠 How It Works

1. Avatar speaks the interview question and technical/behavioural feedbacks in real time.
2. Candidate records response via webcam + microphone.
3. Frontend performs **live audio, face & emotion checks** and gives spoken feedback.
4. Recording is sent to backend.
5. Backend pipeline:

   * Whisper transcribes the audio
   * DeepFace analyzes emotions
   * OpenRouter LLM evaluates the answer
6. Backend returns:

   * Transcript
   * Scores
   * Feedback
   * Emotion summary
7. Frontend updates charts, transcript box, and avatar feedback.
8. The next question is generated and spoken automatically.
9. Attempt data is stored in MongoDB for history tracking.

---

## Screenshots

<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/d3e1c3fe-04f7-4658-b75b-7aa6f24077d1" />

---

## 🧩 Dependencies

```
fastapi
uvicorn
python-dotenv
pymongo
openai-whisper
torch
torchvision
torchaudio
deepface
tf-keras
opencv-python-headless
numpy
requests
python-multipart
pydantic
```

---

## 🧰 Troubleshooting

| Issue                          | Fix                                              |
| ------------------------------ | ------------------------------------------------ |
| `ModuleNotFoundError: whisper` | Run `pip install -U openai-whisper`              |
| `No module named tf_keras`     | Run `pip install tf-keras`                       |
| MongoDB connection failed      | Check `.env` for correct URI                     |
| Whisper slow                   | Use `tiny` or `small` model                      |
| Avatar not speaking            | Enable browser SpeechSynthesis voice permissions |
| Camera/mic blocked             | Allow permissions in browser                     |

---

## 🧪 Health Check Example

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "timestamp": "2025-12-12T20:55:00",
  "whisper_loaded": true,
  "mongodb_connected": true,
  "openrouter_configured": true
}
```

---

## 📊 Example Output

```json
{
  "status": "ok",
  "filename": "20251031021000_interview.mp4",
  "transcript": "My name is John Doe, and I have three years of experience...",
  "score": {
    "communication": 85,
    "confidence": 78,
    "structure": 80,
    "soft_skills": 82,
    "feedback": "Strong communication and structure. Slight nervousness noted but overall confident response."
  },
  "emotions": {
    "dominant_emotion": "happy",
    "confidence_score": 86,
    "frames_analyzed": 120
  }
}
```

---

## 🧑‍💼 Contributors

* **Vaibhav Rakshe** — Developer & Research Lead
* **Shentinelix Sphere Project Team**

---

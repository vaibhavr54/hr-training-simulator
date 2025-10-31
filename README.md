## 🧠 HR Training Simulator — AI-Powered Interview Evaluation System

### 📄 Overview

The **HR Training Simulator** is an AI-driven backend system that evaluates candidate interview videos by combining **speech analysis**, **emotion detection**, and **AI-based scoring**.

This backend uses:

* 🎤 **OpenAI Whisper** for speech transcription
* 😊 **DeepFace + RetinaFace** for facial emotion recognition
* 🧩 **OpenRouter LLM API** for intelligent interview scoring and next-question generation
* 💾 **MongoDB** for session storage and historical analytics
* ⚙️ **FastAPI** for serving endpoints and handling uploads

---

## 🚀 Features

✅ **Speech-to-Text (Transcription)**
Converts candidate’s spoken responses to text using OpenAI’s Whisper model.

✅ **Emotion Detection from Video**
Analyzes video frames to detect emotions (happy, sad, angry, etc.) and computes a confidence score.

✅ **AI-Based Evaluation**
Integrates with OpenRouter’s LLM to assess:

* Communication
* Confidence
* Structure
* Soft Skills
  and generates human-like feedback.

✅ **Dynamic Question Generation**
Creates next interview questions based on previous answers to simulate real HR interactions.

✅ **Data Storage (MongoDB)**
Stores each candidate’s session data, including transcripts, scores, and emotional insights.

✅ **Health Monitoring & Cleanup**
Includes endpoints for health checks and automatic cleanup of old uploaded files.

---

## 🧩 Tech Stack

| Category           | Technology                   |
| ------------------ | ---------------------------- |
| Backend Framework  | FastAPI                      |
| Speech Recognition | OpenAI Whisper               |
| Emotion Detection  | DeepFace + OpenCV            |
| AI Text Generation | OpenRouter API               |
| Database           | MongoDB                      |
| Environment Config | python-dotenv                |
| Async Tasks        | asyncio + ThreadPoolExecutor |
| Logging            | Python logging module        |

---

## 🗂️ Project Structure

```
backend/
│
├── main.py                 # Main FastAPI application
├── .env                    # Environment variables (MONGO_URI, OPENROUTER_API_KEY)
├── requirements.txt         # Python dependencies
├── static/                 # Frontend files (index.html, etc.)
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
# or
source venv/bin/activate      # (Linux/Mac)
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🧾 Example `.env` File

Create a file named **`.env`** in the `backend/` directory:

```
MONGO_URI=mongodb://localhost:27017
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

---

## ▶️ Running the Server

Start your FastAPI backend:

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

## 🔍 API Endpoints

| Method   | Endpoint               | Description                            |
| -------- | ---------------------- | -------------------------------------- |
| **GET**  | `/`                    | Serve main HTML page                   |
| **POST** | `/upload-audio`        | Upload and analyze interview video     |
| **POST** | `/next-question`       | Generate next interview question       |
| **GET**  | `/history/{candidate}` | Retrieve candidate’s interview history |
| **GET**  | `/health`              | Server health status check             |

---

## 🧠 How It Works

1. Candidate uploads an interview video.
2. Video is validated and converted to `.mp4` if needed.
3. Whisper transcribes audio to text.
4. DeepFace analyzes facial emotions across sampled frames.
5. Combined transcript and emotion summary are sent to OpenRouter for scoring.
6. Results (scores + feedback) are stored in MongoDB.
7. The next interview question is dynamically generated.

---

## Screenshots 
<img width="1919" height="916" alt="image" src="https://github.com/user-attachments/assets/68430a05-9c5c-4615-a6b8-048d30a6e5e0" />


## 🧩 Dependencies

Core dependencies (include in `requirements.txt`):

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
```

---

## 🧰 Troubleshooting

| Issue                          | Fix                                                 |
| ------------------------------ | --------------------------------------------------- |
| `ModuleNotFoundError: whisper` | Run `pip install -U openai-whisper`                 |
| `No module named tf_keras`     | Run `pip install tf-keras`                          |
| MongoDB connection failed      | Start MongoDB service or check your `.env` URI      |
| Whisper slow or memory-heavy   | Use `"tiny"` or `"small"` model instead of `"base"` |
| OpenRouter API errors          | Ensure valid `OPENROUTER_API_KEY` in `.env`         |

---

## 🧪 Health Check Example

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "timestamp": "2025-10-31T02:10:00",
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
  "transcript": "My name is John Doe, and I have three years of experience in HR analytics...",
  "score": {
    "communication": 85,
    "confidence": 78,
    "structure": 80,
    "soft_skills": 82,
    "feedback": "Strong communication and structure. Slight nervousness noted but overall confident response.",
    "emotion_analysis": {
      "dominant_emotion": "happy",
      "confidence_score": 86
    }
  }
}
```

---

## 🧑‍💼 Contributors

* **Vaibhav Rakshe** — Developer & Research Lead
* **Shentinelix Sphere Project Team**

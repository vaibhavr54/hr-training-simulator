import os
import re
import json
import random
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from dotenv import load_dotenv
import whisper
import requests

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["hr_simulator"]
sessions = db["sessions"]

BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI app
app = FastAPI(title="HR Training Simulator - Backend")

# Serve the static directory at /static
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Load Whisper model once at startup
model = whisper.load_model("base")

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # use from .env

def get_openrouter_score(transcript: str):
    """Send transcript to OpenRouter for scoring/feedback"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",  # Change if deployed
        "X-Title": "HR Simulator"
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": [
            {
                "role": "system",
                "content": "You are an experienced HR interviewer. Return ONLY a valid JSON object, no extra text."
            },
            {
                "role": "user",
                "content": f"""
Transcript: "{transcript}"

Evaluate this response. Return ONLY a JSON object:
{{
  "communication": number from 0 to 100,
  "confidence": number from 0 to 100,
  "structure": number from 0 to 100,
  "soft_skills": number from 0 to 100,
  "feedback": "5 lines feedback"
}}
"""
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Extract JSON block if extra text is present
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        parsed = json.loads(content)

        # Validate required keys
        required_keys = {"communication", "confidence", "structure", "soft_skills", "feedback"}
        if not required_keys.issubset(parsed.keys()):
            print("[WARNING] Missing keys in model output → falling back")
            return None

        return parsed

    except Exception as e:
        print(f"[OpenRouter ERROR] {e}")
        return None


def fallback_score(transcript: str):
    """Simple rule-based fallback scoring"""
    return {
        "communication": 0,
        "confidence": 0,
        "structure": 0,
        "soft_skills": 0,
        "feedback": "Model unavailable, no feedback."
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    candidate: str = Form(...),
    role: str = Form(None),
):
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = uploads_dir / filename

    # Save uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Run Whisper transcription
    result = model.transcribe(str(file_path))
    transcript = result["text"]

    # Try scoring with OpenRouter
    score_data = None
    if OPENROUTER_API_KEY:
        score_data = get_openrouter_score(transcript)

    # If model fails or returns bad output, fallback
    if not score_data:
        score_data = fallback_score(transcript)

    # Store metadata + transcript + score in MongoDB
    sessions.insert_one({
        "candidate": candidate,
        "role": role or "General",
        "filename": filename,
        "file_path": str(file_path),
        "transcript": transcript,
        "timestamp": datetime.now(),
        "score": score_data
    })

    print(f"[INFO] Candidate: {candidate} | Role: {role} | File: {filename}")
    print(f"[Transcript] {transcript}")
    print(f"[Score] {score_data}")

    return {
        "status": "ok",
        "filename": filename,
        "path": str(file_path),
        "transcript": transcript,
        "score": score_data
    }


@app.get("/history/{candidate}")
def get_history(candidate: str):
    cursor = sessions.find({"candidate": candidate}, {"_id": 0})
    attempts = []
    for doc in cursor:
        ts = doc.get("timestamp")
        doc["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else ts
        attempts.append(doc)
    return {"history": attempts}


@app.get("/health")
def health():
    return {"status": "ok"}
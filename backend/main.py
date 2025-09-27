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

# --- Define roles + 30 questions each (15 roles total) ---
ROLES = {
    "General": [f"General Question {i+1}" for i in range(30)],
    "Software Engineer": [f"Software Engineer Q{i+1}" for i in range(30)],
    "Data Scientist": [f"Data Scientist Q{i+1}" for i in range(30)],
    "Product Manager": [f"Product Manager Q{i+1}" for i in range(30)],
    "UX Designer": [f"UX Designer Q{i+1}" for i in range(30)],
    "QA Engineer": [f"QA Engineer Q{i+1}" for i in range(30)],
    "DevOps Engineer": [f"DevOps Q{i+1}" for i in range(30)],
    "Business Analyst": [f"BA Q{i+1}" for i in range(30)],
    "Marketing Specialist": [f"Marketing Q{i+1}" for i in range(30)],
    "HR Specialist": [f"HR Q{i+1}" for i in range(30)],
    "Finance Analyst": [f"Finance Q{i+1}" for i in range(30)],
    "Project Coordinator": [f"Project Coordinator Q{i+1}" for i in range(30)],
    "Sales Executive": [f"Sales Q{i+1}" for i in range(30)],
    "Customer Support": [f"Support Q{i+1}" for i in range(30)],
    "Operations Manager": [f"Operations Q{i+1}" for i in range(30)],
}

# --- Helper functions ---
def get_openrouter_score(question: str, transcript: str):
    """Send transcript + question to OpenRouter for scoring/feedback"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HR Simulator"
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are an experienced HR interviewer. Return ONLY a valid JSON object, no extra text."},
            {"role": "user",
             "content": f"""
Question: "{question}"
Answer: "{transcript}"

Evaluate this response. Return ONLY a JSON object:
{{
  "communication": number from 0 to 100,
  "confidence": number from 0 to 100,
  "structure": number from 0 to 100,
  "soft_skills": number from 0 to 100,
  "feedback": "5 lines feedback specific to this question"
}}
"""}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        parsed = json.loads(content)

        required_keys = {"communication", "confidence", "structure", "soft_skills", "feedback"}
        if not required_keys.issubset(parsed.keys()):
            print("[WARNING] Missing keys in model output → fallback")
            return None

        return parsed
    except Exception as e:
        print(f"[OpenRouter ERROR] {e}")
        return None

def fallback_score():
    """Fallback scoring if LLM fails"""
    return {
        "communication": 0,
        "confidence": 0,
        "structure": 0,
        "soft_skills": 0,
        "feedback": "Model unavailable, no feedback."
    }

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/get-question/{role}")
def get_question(role: str):
    questions = ROLES.get(role, ROLES["General"])
    return {"question": random.choice(questions)}

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    candidate: str = Form(...),
    role: str = Form(...),
    question: str = Form(...)
):
    uploads_dir = BASE_DIR / "uploads"
    uploads_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = uploads_dir / filename

    # Save uploaded file (audio-only)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Transcribe with Whisper
    result = model.transcribe(str(file_path))
    transcript = result["text"]

    # Get per-question score
    score_data = None
    if OPENROUTER_API_KEY:
        score_data = get_openrouter_score(question, transcript)
    if not score_data:
        score_data = fallback_score()

    # Store in MongoDB
    sessions.insert_one({
        "candidate": candidate,
        "role": role,
        "question": question,
        "filename": filename,
        "file_path": str(file_path),
        "transcript": transcript,
        "timestamp": datetime.now(),
        "score": score_data
    })

    print(f"[INFO] Candidate: {candidate} | Role: {role} | Question: {question}")
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

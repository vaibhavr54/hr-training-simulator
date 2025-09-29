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
import cv2
import numpy as np
from deepface import DeepFace
import base64
import tempfile

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

# Serve static directory
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Load Whisper model once at startup
model = whisper.load_model("base")

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---- Emotion Detection Helper ----
def analyze_emotions_from_video(video_path):
    """Analyze emotions from video frames using DeepFace"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        emotions_data = []
        feedback_messages = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 30th frame to reduce processing
            if frame_count % 30 == 0:
                try:
                    # Analyze emotion and head pose
                    result = DeepFace.analyze(frame, 
                                            actions=['emotion'],
                                            enforce_detection=False)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    dominant_emotion = result.get('dominant_emotion', 'neutral')
                    emotions = result.get('emotion', {})
                    
                    # Detect if looking away (simple head pose check using face detection confidence)
                    face_region = result.get('region', {})
                    if face_region:
                        face_width = face_region.get('w', 0)
                        face_height = face_region.get('h', 0)
                        # If face is too small or distorted, likely looking away
                        if face_width < 50 or face_height < 50:
                            feedback_messages.append(f"Frame {frame_count}: Please look straight at the camera")
                    
                    emotions_data.append({
                        'frame': frame_count,
                        'dominant_emotion': dominant_emotion,
                        'all_emotions': emotions,
                        'timestamp': frame_count / 30  # Approximate seconds
                    })
                    
                    # Generate feedback based on emotions
                    if dominant_emotion in ['sad', 'disgust', 'fear']:
                        feedback_messages.append(f"Time {frame_count//30}s: Try to maintain a more confident expression")
                    elif dominant_emotion == 'angry':
                        feedback_messages.append(f"Time {frame_count//30}s: Stay calm and composed")
                    elif dominant_emotion == 'surprise':
                        feedback_messages.append(f"Time {frame_count//30}s: Good engagement shown")
                    
                except Exception as e:
                    print(f"Frame {frame_count} analysis error: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate emotion summary
        if emotions_data:
            emotion_counts = {}
            for data in emotions_data:
                emotion = data['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Most frequent emotion
            dominant_overall = max(emotion_counts, key=emotion_counts.get)
            
            return {
                'dominant_emotion': dominant_overall,
                'emotion_distribution': emotion_counts,
                'feedback_messages': feedback_messages[:5],  # Limit to 5 messages
                'confidence_score': calculate_confidence_from_emotions(emotion_counts)
            }
        
        return None
        
    except Exception as e:
        print(f"[Emotion Detection Error] {e}")
        return None

def calculate_confidence_from_emotions(emotion_counts):
    """Calculate confidence score based on emotion distribution"""
    total = sum(emotion_counts.values())
    if total == 0:
        return 50
    
    positive_emotions = emotion_counts.get('happy', 0) + emotion_counts.get('neutral', 0)
    negative_emotions = emotion_counts.get('sad', 0) + emotion_counts.get('fear', 0) + emotion_counts.get('disgust', 0)
    
    confidence_score = (positive_emotions / total) * 100
    return min(100, max(0, confidence_score))

# ---- Helper for OpenRouter ----
def call_openrouter(messages, temperature=0.7):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "HR Simulator"
    }
    payload = {
        "model": "meta-llama/llama-3.3-8b-instruct:free",
        "messages": messages,
        "temperature": temperature
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[OpenRouter ERROR] {e}")
        return None

# ---- Score evaluation ----
def get_openrouter_score(transcript: str):
    messages = [
        {"role": "system", "content": "You are an experienced HR interviewer. Return ONLY a valid JSON object, no extra text."},
        {"role": "user", "content": f"""
Transcript: "{transcript}"

Evaluate this response. Return ONLY a JSON object:
{{
  "communication": number from 0 to 100,
  "confidence": number from 0 to 100,
  "structure": number from 0 to 100,
  "soft_skills": number from 0 to 100,
  "feedback": "5 lines feedback"
}}
"""}
    ]
    try:
        content = call_openrouter(messages, temperature=0.3)
        if not content:
            return None
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        parsed = json.loads(content)
        required_keys = {"communication", "confidence", "structure", "soft_skills", "feedback"}
        if not required_keys.issubset(parsed.keys()):
            return None
        return parsed
    except Exception as e:
        print(f"[Score ERROR] {e}")
        return None

def fallback_score(_):
    return {
        "communication": 0,
        "confidence": 0,
        "structure": 0,
        "soft_skills": 0,
        "feedback": "Model unavailable, no feedback."
    }

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    candidate: str = Form(...),
    role: str = Form(None),
    last_question: str = Form(None)
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

    # Whisper transcription
    result = model.transcribe(str(file_path))
    transcript = result["text"]
    
    # Emotion detection from video
    emotion_data = analyze_emotions_from_video(file_path)

    # Evaluate response
    score_data = None
    if OPENROUTER_API_KEY:
        score_data = get_openrouter_score(transcript)
    if not score_data:
        score_data = fallback_score(transcript)
    
    # Merge emotion feedback into score
    if emotion_data:
        score_data['emotion_analysis'] = emotion_data
        # Adjust confidence score based on emotions
        if emotion_data.get('confidence_score'):
            score_data['confidence'] = int((score_data['confidence'] + emotion_data['confidence_score']) / 2)

    # Store in MongoDB
    candidate_name = candidate.strip() or "unknown"
    sessions.insert_one({
        "candidate": candidate_name,
        "role": role or "General",
        "filename": filename,
        "file_path": str(file_path),
        "transcript": transcript,
        "last_question": last_question,
        "timestamp": datetime.now(),
        "score": score_data,
        "emotion_data": emotion_data  # Store emotion data
    })

    print(f"[INFO] Candidate: {candidate_name} | Role: {role} | File: {filename}")
    print(f"[Transcript] {transcript}")
    print(f"[Score] {score_data}")
    if emotion_data:
        print(f"[Emotions] Dominant: {emotion_data.get('dominant_emotion')}, Distribution: {emotion_data.get('emotion_distribution')}")

    return {
        "status": "ok",
        "filename": filename,
        "path": str(file_path),
        "transcript": transcript,
        "score": score_data,
        "emotions": emotion_data  # Include emotions in response
    }

@app.post("/next-question")
async def next_question(candidate: str = Form(...), role: str = Form(...), last_answer: str = Form(None)):
    """Generate the next interview question dynamically"""
    if not OPENROUTER_API_KEY:
        return {"question": f"[Static Fallback] Describe your experience for role: {role}"}

    if not last_answer:
        user_prompt = f"You are an HR interviewer conducting an interview for the role of {role}. Ask the very first question to the candidate. Keep it open-ended and professional."
    else:
        user_prompt = f"Candidate's last answer: \"{last_answer}\".\nGenerate the next logical interview question for the role {role}. Do NOT give multiple questions, only ONE clear question."

    messages = [
        {"role": "system", "content": "You are an HR interviewer. Always return only ONE interview question."},
        {"role": "user", "content": user_prompt}
    ]

    question = call_openrouter(messages, temperature=0.7)
    if not question:
        question = f"[Fallback] Tell me more about your skills in {role}."

    return {"question": question}

@app.get("/history/{candidate}")
def get_history(candidate: str):
    candidate_name = candidate.strip()
    cursor = sessions.find({"candidate": candidate_name}, {"_id": 0}).sort("timestamp", 1)
    attempts = []
    for doc in cursor:
        ts = doc.get("timestamp")
        doc["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else ts
        attempts.append(doc)
    return {"history": attempts}

@app.get("/health")
def health():
    return {"status": "ok"}
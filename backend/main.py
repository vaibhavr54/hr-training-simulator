import os
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from dotenv import load_dotenv
import requests
import whisper
import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["hr_simulator"]
    sessions = db["sessions"]
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None
    sessions = None

BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI app
app = FastAPI(title="HR Training Simulator - Backend")

# Serve static directory
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create uploads directory
uploads_dir = BASE_DIR / "uploads"
uploads_dir.mkdir(exist_ok=True)

# Load Whisper model once at startup
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

# OpenRouter setup
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.webm', '.mp4', '.avi', '.mov'}

# ---- File Validation ----
def validate_upload_file(filename: str, file_size: int):
    """Validate uploaded file"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not allowed. Use: {ALLOWED_EXTENSIONS}")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")

# ---- Video Format Conversion ----
def convert_to_mp4(input_path: Path) -> Path:
    """Convert video to MP4 format using ffmpeg"""
    try:
        output_path = input_path.with_suffix('.mp4')
        
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        
        if result.returncode != 0:
            logger.warning("FFmpeg not available, using original file")
            return input_path
        
        subprocess.run([
            'ffmpeg', '-i', str(input_path),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            str(output_path)
        ], check=True, capture_output=True, timeout=60)
        
        logger.info(f"Converted {input_path.name} to MP4")
        
        if input_path.suffix == '.webm' and output_path.exists():
            input_path.unlink()
        
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return input_path
    except Exception as e:
        logger.error(f"Video conversion failed: {e}")
        return input_path

# ---- Emotion Detection Helper (WITHOUT FACE VISIBILITY SCORE) ----
def analyze_emotions_from_video(video_path: Path):
    """Analyze emotions from video frames using DeepFace with optimizations"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        emotions_data = []
        feedback_messages = []
        frame_count = 0
        analyzed_count = 0
        
        sample_interval = 60
        max_samples = 20
        
        logger.info(f"Analyzing video: {total_frames} frames at {fps} fps")
        
        while analyzed_count < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                try:
                    small_frame = cv2.resize(frame, (320, 240))
                    
                    result = DeepFace.analyze(
                        small_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    dominant_emotion = result.get('dominant_emotion', 'neutral')
                    emotions = result.get('emotion', {})
                    
                    emotions_data.append({
                        'frame': frame_count,
                        'dominant_emotion': dominant_emotion,
                        'all_emotions': emotions,
                        'timestamp': frame_count / fps
                    })
                    
                    analyzed_count += 1
                    
                    # Generate targeted feedback (without face visibility mentions)
                    time_str = f"At {frame_count//fps:.0f}s"
                    
                    if dominant_emotion in ['sad', 'fear'] and len(feedback_messages) < 5:
                        feedback_messages.append(
                            f"{time_str}: Try to project more confidence"
                        )
                    elif dominant_emotion == 'angry' and len(feedback_messages) < 5:
                        feedback_messages.append(
                            f"{time_str}: Maintain a calm demeanor"
                        )
                    elif dominant_emotion == 'happy' and analyzed_count == 1:
                        feedback_messages.append("Good: Positive engagement detected")
                    
                except Exception as e:
                    logger.warning(f"Frame {frame_count} analysis error: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate emotion summary
        if emotions_data:
            emotion_counts = {}
            
            for data in emotions_data:
                emotion = data['dominant_emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            dominant_overall = max(emotion_counts, key=emotion_counts.get)
            confidence_score = calculate_confidence_from_emotions(emotion_counts)
            
            return {
                'dominant_emotion': dominant_overall,
                'emotion_distribution': emotion_counts,
                'feedback_messages': feedback_messages[:5],
                'confidence_score': confidence_score,
                'frames_analyzed': analyzed_count
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return None

def calculate_confidence_from_emotions(emotion_counts):
    """Calculate confidence score based on emotion distribution"""
    total = sum(emotion_counts.values())
    if total == 0:
        return 50
    
    positive_emotions = emotion_counts.get('happy', 0) + emotion_counts.get('neutral', 0) * 0.7
    negative_emotions = (
        emotion_counts.get('sad', 0) * 1.2 +
        emotion_counts.get('fear', 0) * 1.5 +
        emotion_counts.get('disgust', 0) * 1.3 +
        emotion_counts.get('angry', 0) * 1.4
    )
    
    confidence_score = ((positive_emotions - negative_emotions * 0.5) / total) * 100
    return min(100, max(0, int(confidence_score)))

# ---- Helper for OpenRouter ----
def call_openrouter(messages, temperature=0.7, timeout=30):
    """Call OpenRouter API with error handling"""
    if not OPENROUTER_API_KEY:
        logger.warning("OpenRouter API key not configured")
        return None
    
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
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.Timeout:
        logger.error("OpenRouter request timed out")
        return None
    except requests.RequestException as e:
        logger.error(f"OpenRouter API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenRouter: {e}")
        return None

# ---- Score evaluation ----
def get_openrouter_score(transcript: str, emotion_data: dict = None):
    """Get AI-based scoring with emotion context"""
    emotion_context = ""
    if emotion_data:
        emotion_context = f"\nEmotion Analysis: Dominant emotion was '{emotion_data.get('dominant_emotion', 'neutral')}' with confidence score {emotion_data.get('confidence_score', 50)}/100."
    
    messages = [
        {"role": "system", "content": "You are an experienced HR interviewer. Return ONLY a valid JSON object, no extra text."},
        {"role": "user", "content": f"""
Transcript: "{transcript}"
{emotion_context}

Evaluate this interview response. Return ONLY a JSON object with this exact structure:
{{
  "communication": <number 0-100>,
  "confidence": <number 0-100>,
  "structure": <number 0-100>,
  "soft_skills": <number 0-100>,
  "feedback": "<concise 3-5 line feedback>"
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
            logger.warning("Missing required keys in AI response")
            return None
        
        for key in ["communication", "confidence", "structure", "soft_skills"]:
            parsed[key] = int(parsed[key])
        
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Score evaluation error: {e}")
        return None

def fallback_score(transcript: str):
    """Fallback scoring based on transcript length and basic analysis"""
    words = transcript.split()
    word_count = len(words)
    
    communication = min(100, max(30, word_count * 2))
    confidence = 50
    structure = 40 if word_count > 20 else 30
    soft_skills = 45
    
    return {
        "communication": communication,
        "confidence": confidence,
        "structure": structure,
        "soft_skills": soft_skills,
        "feedback": "AI scoring unavailable. Please check API configuration. Basic scoring applied based on response length."
    }

# ---- Cleanup old files ----
def cleanup_old_files():
    """Delete files older than 7 days"""
    try:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=7)
        
        deleted_count = 0
        for file_path in uploads_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff:
                    file_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old files")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page"""
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "index.html not found in static directory")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    candidate: str = Form(...),
    role: str = Form("General"),
    last_question: str = Form("")
):
    """Handle video upload, transcription, and analysis"""
    
    contents = await file.read()
    file_size = len(contents)
    
    try:
        validate_upload_file(file.filename, file_size)
    except HTTPException as e:
        return {"status": "error", "message": str(e.detail)}
    
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = uploads_dir / filename
    
    try:
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved: {filename} ({file_size/1024/1024:.2f}MB)")
        
        video_path = await asyncio.get_event_loop().run_in_executor(
            executor, convert_to_mp4, file_path
        )
        
        # Transcription
        if model is None:
            transcript = "[Whisper model not loaded]"
        else:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, model.transcribe, str(video_path)
                )
                transcript = result["text"].strip()
                logger.info(f"Transcription complete: {len(transcript)} chars")
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                transcript = "[Transcription failed]"
        
        # Emotion detection (async)
        emotion_data = await asyncio.get_event_loop().run_in_executor(
            executor, analyze_emotions_from_video, video_path
        )
        
        # Score evaluation
        score_data = await asyncio.get_event_loop().run_in_executor(
            executor, get_openrouter_score, transcript, emotion_data
        )
        
        if not score_data:
            score_data = fallback_score(transcript)
        
        # Merge emotion feedback
        if emotion_data:
            score_data['emotion_analysis'] = emotion_data
            
            # Adjust confidence based on emotions
            emotion_confidence = emotion_data.get('confidence_score', 50)
            score_data['confidence'] = int((score_data['confidence'] * 0.6 + emotion_confidence * 0.4))
        
        # Store in MongoDB
        if sessions is not None:
            try:
                candidate_name = candidate.strip() or "unknown"
                sessions.insert_one({
                    "candidate": candidate_name,
                    "role": role,
                    "filename": video_path.name,
                    "file_path": str(video_path),
                    "transcript": transcript,
                    "last_question": last_question,
                    "timestamp": datetime.now(),
                    "score": score_data,
                    "emotion_data": emotion_data
                })
                logger.info(f"Session saved for {candidate_name}")
            except Exception as e:
                logger.error(f"MongoDB insert error: {e}")
        
        # Cleanup old files periodically
        if timestamp.endswith("00"):
            await asyncio.get_event_loop().run_in_executor(executor, cleanup_old_files)
        
        return {
            "status": "ok",
            "filename": video_path.name,
            "transcript": transcript,
            "score": score_data,
            "emotions": emotion_data
        }
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/next-question")
async def next_question(
    candidate: str = Form(...),
    role: str = Form("General"),
    last_answer: str = Form(None)
):
    """Generate the next interview question dynamically"""
    
    if not OPENROUTER_API_KEY:
        return {"question": f"Tell me about your experience and skills relevant to the {role} role."}
    
    if not last_answer:
        user_prompt = f"""You are conducting an interview for a {role} position. 
Ask an engaging opening question that allows the candidate to introduce themselves and their relevant experience. 
Keep it professional and conversational."""
    else:
        user_prompt = f"""Based on the candidate's previous answer: "{last_answer[:500]}"

Generate ONE thoughtful follow-up question for a {role} interview. 
The question should:
- Build on their previous answer
- Probe deeper into their experience
- Be specific and relevant to the role
Return ONLY the question, no preamble."""
    
    messages = [
        {"role": "system", "content": "You are an experienced HR interviewer. Return only ONE clear interview question."},
        {"role": "user", "content": user_prompt}
    ]
    
    question = await asyncio.get_event_loop().run_in_executor(
        executor, call_openrouter, messages, 0.7, 20
    )
    
    if not question:
        question = f"Can you elaborate on your qualifications for the {role} position?"
    
    return {"question": question}

@app.get("/history/{candidate}")
async def get_history(candidate: str):
    """Retrieve candidate's interview history"""
    
    if sessions is None:
        return {"history": [], "error": "Database not connected"}
    
    try:
        candidate_name = candidate.strip()
        cursor = sessions.find(
            {"candidate": candidate_name},
            {"_id": 0}
        ).sort("timestamp", -1).limit(50)
        
        attempts = []
        for doc in cursor:
            ts = doc.get("timestamp")
            doc["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            attempts.append(doc)
        
        return {"history": attempts}
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        return {"history": [], "error": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "whisper_loaded": model is not None,
        "mongodb_connected": sessions is not None,
        "openrouter_configured": OPENROUTER_API_KEY is not None
    }
    return status

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 50)
    logger.info("HR Training Simulator Starting...")
    logger.info(f"Whisper Model: {'Loaded' if model is not None else 'Not Loaded'}")
    logger.info(f"MongoDB: {'Connected' if sessions is not None else 'Not Connected'}")
    logger.info(f"OpenRouter: {'Configured' if OPENROUTER_API_KEY else 'Not Configured'}")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    executor.shutdown(wait=False)
import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from dotenv import load_dotenv
import whisper

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

    # Store metadata + transcript in MongoDB
    sessions.insert_one({
        "candidate": candidate,
        "role": role or "General",
        "filename": filename,
        "file_path": str(file_path),
        "transcript": result["text"],
        "timestamp": datetime.utcnow(),
        "score": None  # placeholder for future GPT scoring
    })

    print(f"[INFO] Candidate: {candidate} | Role: {role} | File: {filename}")
    print(f"[Transcript] {result['text']}")

    return {
        "status": "ok",
        "filename": filename,
        "path": str(file_path),
        "transcript": result["text"],
    }


@app.get("/history/{candidate}")
def get_history(candidate: str):
    cursor = sessions.find({"candidate": candidate}, {"_id": 0})
    attempts = []
    for doc in cursor:
        # Ensure timestamp is JSON-serializable
        ts = doc.get("timestamp")
        doc["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else ts
        attempts.append(doc)
    return {"history": attempts}


@app.get("/health")
def health():
    return {"status": "ok"}
import whisper

# Load a small model for faster test
model = whisper.load_model("base")

# Path to one of your uploaded files
audio_path = "uploads/20250923170048_response.webm"

print("Transcribing:", audio_path)
result = model.transcribe(audio_path)
print("Transcript:", result["text"])
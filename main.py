import io
import torch
import whisper
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# -----------------------------
# Load Whisper (OpenAI)
# -----------------------------
whisper_model = whisper.load_model("tiny")   

# -----------------------------
# Load IndicBERT Model
# -----------------------------
INDIC_MODEL_PATH = "indic_model"
LABEL_ENCODER_PATH = "label_encoder.pkl"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_PATH)
indic_model = AutoModelForSequenceClassification.from_pretrained(INDIC_MODEL_PATH).to(device)
indic_model.eval()

label_encoder = joblib.load(LABEL_ENCODER_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="🎤 Whisper + IndicBERT API", version="1.0")

# -----------------------------
# Helper: Intent Classification
# -----------------------------
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = indic_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        intent = label_encoder.inverse_transform(preds)[0]
    return intent

# -----------------------------
# API Endpoint (WAV-only)
# -----------------------------
@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is WAV
        if not file.filename.lower().endswith(".wav"):
            return JSONResponse({"error": "Only WAV files are supported"}, status_code=400)

        # Read WAV bytes into memory
        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        # Whisper transcription
        result = whisper_model.transcribe(y, fp16=False)
        transcription = result["text"].strip()

        # IndicBERT intent classification
        intent = classify_intent(transcription)

        return JSONResponse({"transcription": transcription, "intent": intent})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

import io
import torch
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import joblib

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Hugging Face Whisper
# -----------------------------
WHISPER_MODEL_NAME = "openai/whisper-tiny"   # or "base" if you have more RAM
processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(device)
whisper_model.eval()

# -----------------------------
# Load IndicBERT Model
# -----------------------------
INDIC_MODEL_PATH = "indic_model"
LABEL_ENCODER_PATH = "label_encoder.pkl"

tokenizer = AutoTokenizer.from_pretrained(INDIC_MODEL_PATH)
indic_model = AutoModelForSequenceClassification.from_pretrained(INDIC_MODEL_PATH).to(device)
indic_model.eval()

label_encoder = joblib.load(LABEL_ENCODER_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="🎤 Whisper (HF) + IndicBERT API", version="1.0")

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

        # Read WAV bytes
        audio_bytes = await file.read()

        # Load audio into librosa (convert to 16kHz mono)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        # Convert waveform to Whisper input
        input_features = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Transcribe using Whisper
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        # IndicBERT intent classification
        intent = classify_intent(transcription)

        return JSONResponse({"transcription": transcription, "intent": intent})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

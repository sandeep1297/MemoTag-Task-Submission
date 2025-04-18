# file: main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn
import tempfile
import os
import whisper
import librosa
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

class Features(BaseModel):
    pauses_per_sentence: float
    hesitation_rate: float
    speech_rate: float
    pitch_std: float
    risk_score: float
    transcript: str


def extract_features(audio_path: str, transcript: str) -> Dict:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    words = transcript.split()
    num_words = len(words)
    num_ums = sum(1 for w in words if w.lower() in ["uh", "um"])

    intervals = librosa.effects.split(y, top_db=30)
    pauses = 0
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i-1][1]) / sr
        if gap > 0.4:
            pauses += 1

    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr, frame_length=2048)

    return {
        "pauses_per_sentence": pauses / max(1, transcript.count(".")),
        "hesitation_rate": num_ums / max(1, num_words),
        "speech_rate": num_words / (duration / 60),
        "pitch_std": float(np.std(pitch)),
        "transcript": transcript
    }


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/analyze_audio", response_model=Features)
async def analyze_audio(file: UploadFile = File(...)):
    asr_model = whisper.load_model("tiny")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = asr_model.transcribe(tmp_path)
    transcript = result["text"]

    features = extract_features(tmp_path, transcript)

    X = np.array([[
        features["pauses_per_sentence"],
        features["hesitation_rate"],
        features["speech_rate"],
        features["pitch_std"]
    ]])

    clf = IsolationForest()
    clf.fit(X)
    risk_score = -clf.decision_function(X)[0]

    os.remove(tmp_path)

    return Features(**features, risk_score=risk_score)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

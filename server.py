"""FastAPI Server — REST API for the voice agent pipeline.

Endpoints:
    GET  /health       → Healthcheck
    POST /tts          → Text → WAV audio
    POST /stt          → Audio file → transcribed text
    POST /agent/text   → Text + history → agent response (sentences + updated history)
    POST /pipeline     → Audio file → full pipeline (STT → Agent → TTS), returns JSON + audio

Run:
    uvicorn server:app --reload
"""

import io
import os
import base64                                       # Encodes binary audio as string — needed because JSON can't carry raw bytes
import tempfile                                     # Creates temp files for audio uploads — Whisper needs a filepath, not bytes

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response              # For returning raw bytes (WAV) instead of JSON
from pydantic import BaseModel                      # Request body validation + auto-generated /docs
from datetime import datetime, timezone

import stt
import tts
import agent

app = FastAPI(
    title="Realtime Voice Agent",
    description="Voice conversation agent: STT → Claude → TTS",
)
START_TIME = datetime.now(timezone.utc)

# ── Root Entry Point ──────────────────────────────────────────
@app.get("/")
def root():
    """Starting Page of API"""
    return {
        "message": "Welcome to Realtime Voice Agent",
        "version": "1.0.0",
        "endpoints": ["/health", "/tts", "/stt", "/agent/text", "/pipeline"]
    }

# ── Health Check ──────────────────────────────────────────

@app.get("/health")
async def health():
    uptime = (datetime.now(timezone.utc) - START_TIME).total_seconds()
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "environment": os.getenv("ENVIRONMENT", "development")              # Recalls the environment variable "environment" -> "production" on railway, "development" local (because not defined)
    }


# ── TTS Demo ─────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str

# No response class because no JSON is returned
# Instead we return bytes that are needed for the WAV


@app.post("/tts")
def tts_demo(req: TTSRequest):
    """Text input, WAV audio output. Direct audio response — playable in browser/curl."""
    audio_data, samplerate = tts.generate_audio(req.text)

    buffer = io.BytesIO()                                               # keeps audio im RAM
    sf.write(buffer, audio_data, samplerate, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav")


# ── STT Demo ─────────────────────────────────────────────

class STTResponse(BaseModel):
    text: str

@app.post("/stt", response_model=STTResponse)
async def stt_demo(file: UploadFile = File(...)):
    """Audio file in, transcribed text out."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:   # Whisper needs a data path, so we need to create temp file
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = stt.transcribe_to_text(tmp_path)                             # Try to translate temp file into text
    finally:
        os.unlink(tmp_path)                                                 # Removes the temp file again

    return {"text": text}


# ── Agent Text Demo ──────────────────────────────────────

class AgentRequest(BaseModel):
    text: str
    history: list[dict] = []

class AgentResponse(BaseModel):
    response: str               # Full text as one string
    sentences: list[str]        # Individual sentences (useful for per-sentence TTS)
    history: list[dict]         # Updated — send back in next request for multi-turn

@app.post("/agent/text", response_model=AgentResponse)
def agent_text_demo(req: AgentRequest):
    """
    Text + conversation history in, agent response out.

    Returns updated history — send it back in the next request for multi-turn.
    No streaming yet (all sentences collected, then returned).
    """
    history = req.history.copy()
    sentences = list(agent.stream_text(req.text, history))

    return {
        "response": " ".join(sentences),
        "sentences": sentences,
        "history": history,
    }


# ── Full Pipeline Demo ───────────────────────────────────

class PipelineResponse(BaseModel):
    user_text: str                          # User query
    agent_response: str                     # Agent full response
    sentences: list[str]                    # Agent response as single sentences
    audio_base64: str | None = None         # base64-encoded WAV — None if TTS produced no audio
    audio_sample_rate: int | None = None

@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_demo(file: UploadFile = File(...), include_audio: bool = False):
    """
    Audio file in → STT → Agent → TTS → JSON response with text + base64 audio.

    Single-turn only (no history across requests).
    """
    # ── STT ──
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        user_text = stt.transcribe_to_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    # ── Agent + TTS ──
    history = []
    audio_chunks = []
    sr = None
    sentences = []

    for sentence in agent.stream_text(user_text, history):
        sentences.append(sentence)
        audio_data, sr = tts.generate_audio(sentence)
        audio_chunks.append(audio_data)

    # ── Build response ──
    result = {
        "user_text": user_text,
        "agent_response": " ".join(sentences),
        "sentences": sentences,
    }

    # Only include base64 audio when explicitly requested (avoids huge default responses)
    if include_audio and audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sr, format="WAV")
        result["audio_base64"] = base64.b64encode(buffer.getvalue()).decode()
        result["audio_sample_rate"] = sr

    return result


# ── Run directly ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
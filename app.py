"""Gradio Web Interface — Browser-based voice agent.

Browser Mic → Whisper STT → Claude Streaming → TTS → Audio Streaming.
Imports stt, tts, agent modules directly — does not go through the REST API.

Conversation history accumulates in a Chatbot component. Each turn's combined
audio is saved as a playable entry in the chat. Live audio streams via a
separate gr.Audio(streaming=True) component during generation.

Run locally:
    python app.py
"""

import io
import string
import time
import logging
import signal
import sys
import tempfile
from collections import defaultdict

import numpy as np
import soundfile as sf
import gradio as gr

import stt
import tts
import agent
import config


# ── Logging ───────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("voice-agent")


# ── Rate Limiter (in-memory, per IP) ─────────────────────

class RateLimiter:
    """Tracks pipeline runs per IP. Resets on server restart."""

    def __init__(self, max_per_hour: int = 30, max_per_day: int = 100):
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> tuple[bool, str]:
        now = time.time()
        self.requests[ip] = [t for t in self.requests[ip] if now - t < 86400]
        hour_count = sum(1 for t in self.requests[ip] if now - t < 3600)
        day_count = len(self.requests[ip])

        if hour_count >= self.max_per_hour:
            return False, f"Stundenlimit erreicht ({self.max_per_hour}/h). Bitte warte etwas."
        if day_count >= self.max_per_day:
            return False, f"Tageslimit erreicht ({self.max_per_day}/Tag)."
        return True, ""

    def record(self, ip: str):
        self.requests[ip].append(time.time())


rate_limiter = RateLimiter(max_per_hour=30, max_per_day=100)


# ── Helpers ───────────────────────────────────────────────

def get_client_ip(request: gr.Request | None) -> str:
    """Extract real client IP from X-Forwarded-For (Railway reverse proxy)."""
    if request is None:
        return "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _audio_to_wav_bytes(audio_data: np.ndarray, samplerate: int) -> bytes:
    """Convert numpy audio to in-memory WAV bytes for Gradio streaming."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, samplerate, format="WAV")
    return buffer.getvalue()


# ── Pipeline (generator) ─────────────────────────────────

def process_turn(audio_filepath, chatbot, history, turn_count, request: gr.Request):
    """
    Full voice turn: STT → stop-word check → Agent streaming → TTS per sentence.

    Generator yields (chatbot, audio_chunk, history, turn_count, audio_input):
    - chatbot:     message list — grows with each turn (text + audio entries)
    - audio_chunk: WAV bytes appended to gr.Audio(streaming=True) per sentence
    - history:     conversation state for agent.stream_text()
    - turn_count:  incremented per turn, used for labeling
    - audio_input: yielded as None to reset mic after turn
    """

    # ── Rate limit ────────────────────────────────────────
    client_ip = get_client_ip(request)
    allowed, limit_message = rate_limiter.is_allowed(client_ip)
    if not allowed:
        logger.warning(f"Rate limit exceeded: {client_ip}")
        chatbot = chatbot + [{"role": "assistant", "content": limit_message}]
        yield chatbot, None, history, turn_count, None
        return

    # ── No audio recorded ─────────────────────────────────
    if audio_filepath is None:
        chatbot = chatbot + [{"role": "assistant", "content": "Keine Aufnahme erkannt. Bitte erneut versuchen."}]
        yield chatbot, None, history, turn_count, None
        return

    turn_count += 1
    turn_label = f"Turn {turn_count}"
    rate_limiter.record(client_ip)
    logger.info(f"[{turn_label}] started — IP: {client_ip}")

    # ── STT ───────────────────────────────────────────────
    try:
        user_text = stt.transcribe_to_text(audio_filepath)
    except Exception as e:
        logger.error(f"[{turn_label}] STT failed: {e}")
        chatbot = chatbot + [{"role": "assistant", "content": f"Transkription fehlgeschlagen: {e}"}]
        yield chatbot, None, history, turn_count, None
        return

    logger.info(f"[{turn_label}] User: {user_text[:80]}")

    # Add user message + agent placeholder to chatbot
    chatbot = chatbot + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "..."},
    ]
    yield chatbot, None, history, turn_count, None

    # ── Stop word check ───────────────────────────────────
    # Only trigger if the ENTIRE transcription is a single stop word.
    # "Ende" alone → stop. "Geschichte mit dramatischem Ende" → continue.
    cleaned = user_text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    if cleaned in config.STOP_WORDS:
        logger.info(f"[{turn_label}] Stop word detected")
        farewell = "Tschüss! Bis zum nächsten Mal."
        chatbot[-1] = {"role": "assistant", "content": farewell}
        audio_data, sr = tts.generate_audio(farewell)
        yield chatbot, _audio_to_wav_bytes(audio_data, sr), [], turn_count, None
        return

    # ── Agent + TTS: stream sentence by sentence ──────────
    agent_text = ""
    audio_chunks = []
    sr = None

    for sentence in agent.stream_text(user_text, history):
        agent_text += sentence + " "
        logger.info(f"[{turn_label}] Agent: {sentence[:60]}")

        # Update the last chatbot message (grows with each sentence)
        chatbot[-1] = {"role": "assistant", "content": agent_text.strip()}

        try:
            audio_data, sr = tts.generate_audio(sentence)
            audio_chunks.append(audio_data)
            yield chatbot, _audio_to_wav_bytes(audio_data, sr), history, turn_count, None
        except Exception as e:
            logger.error(f"[{turn_label}] TTS failed: {e}")
            yield chatbot, None, history, turn_count, None
            continue

    # ── Save combined audio as replayable chatbot entry ───
    if audio_chunks and sr:
        full_audio = np.concatenate(audio_chunks)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix=f"turn{turn_count:03d}_"
        )
        sf.write(tmp.name, full_audio, sr)
        tmp.close()

        # gr.Audio as chatbot content renders an embedded audio player
        chatbot = chatbot + [
            {"role": "assistant", "content": gr.Audio(value=tmp.name)}
        ]

    # Final yield: chatbot with audio entry + reset mic for next turn
    yield chatbot, None, history, turn_count, None
    logger.info(f"[{turn_label}] complete — {len(history)} messages in history")


# ── Gradio UI ─────────────────────────────────────────────

with gr.Blocks(title="Realtime Voice Agent") as demo:

    gr.Markdown(
        "# 🎤 Realtime Voice Agent\n"
        "Sprich etwas in das Mikrofon — der Agent antwortet in Echtzeit.\n\n"
        "*Sage 'Stop' oder 'Ende' zum Beenden der Konversation.*"
    )

    # Session state (per browser tab, managed by Gradio)
    history = gr.State(value=[])
    turn_count = gr.State(value=0)

    # Chatbot — accumulates full conversation with text + audio entries
    chatbot = gr.Chatbot(label="Gespräch", height=500)

    with gr.Row():
        # Mic input — resets after each turn via None yield
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="🎤 Deine Frage",
        )
        # Live streaming player — plays audio chunks as they arrive during a turn
        audio_output = gr.Audio(
            label="🔊 Live-Wiedergabe",
            streaming=True,
            autoplay=True,
        )

    reset_btn = gr.Button("🔄 Neues Gespräch", variant="secondary")

    # Reset: clear chatbot, history, turn counter, mic, and streaming player
    reset_btn.click(
        fn=lambda: ([], [], 0, None, None),
        outputs=[chatbot, history, turn_count, audio_input, audio_output],
    )

    # Stop recording → clear old audio → pipeline streams new audio
    # Two chained events: first resets the streaming player so it doesn't
    # hold stale audio from the previous turn, then runs the pipeline.
    audio_input.stop_recording(
        fn=lambda: None,
        outputs=[audio_output],
    ).then(
        fn=process_turn,
        inputs=[audio_input, chatbot, history, turn_count],
        outputs=[chatbot, audio_output, history, turn_count, audio_input],
    )


# ── Graceful Shutdown ─────────────────────────────────────

def graceful_shutdown(signum, frame):
    logger.info(f"Received {signal.Signals(signum).name} — shutting down.")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


# ── Launch ────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
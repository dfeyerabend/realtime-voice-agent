"""Text-to-Speech using OpenAI TTS API."""

import io
import numpy as np
import sounddevice as sd
import soundfile as sf
import dotenv
from openai import OpenAI

import config

dotenv.load_dotenv()
openai_client = OpenAI()  # Needs OPENAI_API_KEY in .env


def generate_audio(text: str) -> tuple[np.ndarray, int]:
    """
    Converts text to audio via OpenAI TTS API.

    Reads audio in-memory via BytesIO (no temp file on disk).
    Returns (numpy_array, sample_rate) — ready for playback or concatenation.
    """
    # Call OpenAI TTS API — returns raw audio bytes
    response = openai_client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
    )

    # Read audio bytes into numpy array without writing a temp file.
    # BytesIO wraps the bytes so soundfile can read them as if they were a file.
    audio_data, samplerate = sf.read(io.BytesIO(response.content))
    return audio_data, samplerate


def speak(text: str) -> tuple[np.ndarray, int]:
    """Like generate_audio, but also plays back through speakers. For CLI usage."""
    audio_data, samplerate = generate_audio(text)
    sd.play(audio_data, samplerate)
    sd.wait()  # Block until playback finishes
    return audio_data, samplerate


def save_audio(audio_data: np.ndarray, samplerate: int, filepath: str):
    """Saves an audio numpy array as a WAV file."""
    sf.write(filepath, audio_data, samplerate)


# ── Standalone test: generate and play a test sentence ──
if __name__ == "__main__":
    print("=== TTS Test ===\n")
    print("Generiere Audio...")
    speak("Hallo, ich bin der Voice Agent. Das ist ein Test.")
    print("Fertig.")

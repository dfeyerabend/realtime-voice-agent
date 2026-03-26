import io
import re
import time
import string
import sounddevice as sd
import soundfile as sf
import dotenv
import numpy as np
from openai import OpenAI
from anthropic import Anthropic

import config

# ──────────────────────────────────────────────
#  Initialization
# ──────────────────────────────────────────────

dotenv.load_dotenv()

openai_client = OpenAI()
anthropic_client = Anthropic()

# ──────────────────────────────────────────────
#  STT — Speech to Text (Whisper)
# ──────────────────────────────────────────────

def record_audio(
        recording_id: int = 1,
        max_duration: int = config.MAX_RECORDING_DURATION,
        sample_rate: int = config.SAMPLE_RATE,
        silence_threshold: float = config.SILENCE_THRESHOLD,
        silence_duration: float = config.SILENCE_DURATION
) -> str | None:
    """
    Records audio from the microphone until silence is detected.

    Waits for the user to start speaking before arming the silence timer.
    Returns None if no speech is detected within max_duration.

    State machine with two phases:
      Phase 1 (waiting):  has_spoken=False → collecting audio, ignoring silence
      Phase 2 (recording): has_spoken=True  → silence timer is armed, stops after silence_duration
    """

    filepath = f"{config.RECORDING_DIR}/{recording_id:03d}_user.wav"
    chunk_size = int(sample_rate * 0.1)  # Each chunk = 100ms of audio. At 16kHz that's 1600 samples per chunk.
    chunks = []  # Raw audio data — every chunk gets appended, trimmed later

    silent_time = 0.0  # Consecutive seconds of silence AFTER user started speaking
    total_time = 0.0  # Total elapsed recording time (safety limit)
    speaking_time = 0.0  # Consecutive seconds of sound — must reach MIN_SPEAKING_DURATION
    has_spoken = False  # Phase flag: flips once, never goes back
    speech_start_index = 0  # Which chunk index to start saving from (trims leading silence)

    print(f"\nBitte jetzt sprechen... (max {max_duration}s, stoppt nach {int(silence_duration)}s Stille)")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while total_time < max_duration:
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())

            volume = np.abs(data).mean()        # Typical values: silence ~0.001, normal speech ~0.01-0.05

            if volume >= silence_threshold:
                # ── Sound detected ──
                speaking_time += 0.1
                silent_time = 0.0               # Reset silence counter — user is actively speaking

                if not has_spoken and speaking_time >= config.MIN_SPEAKING_DURATION:
                    # Enough consecutive sound to count as real speech (not a cough/click).
                    # Transition: Phase 1 → Phase 2. From here, silence timer is armed.
                    has_spoken = True

                    # Mark where speech began, minus a small buffer (~500ms), buffer so the first syllable is not clipped
                    speech_start_index = max(0, len(chunks) - int(config.MIN_SPEAKING_DURATION / 0.1) - 5)

            else:
                # ── Silence detected ──
                if has_spoken:
                    # Phase 2: user was speaking, now quiet → count towards stop
                    silent_time += 0.1
                else:
                    # Phase 1: still waiting for user to start speaking
                    # Reset speaking_time so short noise bursts don't accumulate
                    # Example: cough at t=2s and cough at t=7s should NOT sum to 0.5s
                    speaking_time = 0.0  # Resets speaking time

            total_time += 0.1

            # ── Stop condition: user spoke, then was silent long enough ──
            if has_spoken and silent_time >= silence_duration:
                print(f"Anhaltende Stille erkannt, Aufnahme gestoppt nach {total_time:.1f}s")
                break

    # ── Post-recording: decide what to return ──

    if not has_spoken:
        # Phase 1 never ended → user never spoke → don't send silence to Whisper
        print("Keine Sprache erkannt.")
        return None

    if total_time >= max_duration:
        print(f"Max-Dauer erreicht ({max_duration}s)")

    # Trim: only save from speech_start_index onwards.
    # This removes leading silence so Whisper doesn't hallucinate on empty audio.
    audio = np.concatenate(chunks[speech_start_index:])
    sf.write(filepath, audio, sample_rate)
    print("Aufnahme gespeichert.")
    return filepath

def transcribe_to_text(filepath: str) -> str:
    """Sends a recorded audio file to the OpenAI Whisper API for transcription."""
    with open(filepath, 'rb') as f:
        result = openai_client.audio.transcriptions.create(
            model=config.WHISPER_MODEL,
            file=f,
            language="de"
        )
    return result.text


def agent_audio(text: str, speak: bool = True, save: bool = False, filepath: str = None) -> tuple[np.ndarray, int]:
    """
    Generates audio from text via OpenAI TTS.

    Returns the audio data as numpy array (for concatenation later).
    Optionally plays it back and/or saves it to disk.
    """

    response = openai_client.audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
    )

    # Read audio bytes directly from memory
    audio_data, samplerate = sf.read(io.BytesIO(response.content))

    if speak:
        sd.play(audio_data, samplerate)
        sd.wait()

    if save and filepath:
        sf.write(filepath, audio_data, samplerate)

    return audio_data, samplerate

# ──────────────────────────────────────────────
#  Agent Stream
# ──────────────────────────────────────────────

def stream_and_speak(user_text: str, recording_id: int, history: list) -> str:
    """
    Streams Claude's response token by token and speaks each sentence as soon as it's complete.

    Without streaming: wait for ENTIRE response → then send ALL text to TTS → long delay
    With streaming:    speak each sentence the moment it ends → first audio plays much sooner

    Key insight: Claude's API sends text in small chunks (often single words or fragments).
    We accumulate chunks into sentences and fire TTS the moment we detect a sentence boundary.
    """
    history.append({"role": "user", "content": user_text})

    current_sentence = ""                   # Buffer — accumulates chunks until a sentence boundary is found
    full_response = ""                      # Complete response text — needed for conversation history
    audio_chunks = []                       # Numpy arrays of audio per sentence — concatenated at the end for saving
    sentence_end = re.compile(r'[.!?]\s')   # Sentence boundary = punctuation + whitespace

    # ── Open a streaming connection to Claude ─
    with anthropic_client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=history
    ) as stream:

        for text in stream.text_stream:
            current_sentence += text        # Accumulate into sentence buffer
            full_response += text           # Accumulate into full response
            print(text, end="", flush=True)

            # ── Sentence boundary detection ──

            match = sentence_end.search(current_sentence)
            while match:
                # Extract everything up to and including the boundary
                sentence = current_sentence[:match.end()].strip()
                current_sentence = current_sentence[match.end():]

                if sentence:
                    audio, sr = agent_audio(sentence, speak=True, save=False)
                    audio_chunks.append(audio)

                match = sentence_end.search(current_sentence)

    # ── Handle leftover text ──
    # Claude might end without punctuation, e.g. "Das wars"
    if current_sentence.strip():
        audio, sr = agent_audio(current_sentence.strip(), speak=True, save=False)
        audio_chunks.append(audio)

    # ── Save complete response audio as one file ──
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        sf.write(f"{config.OUTPUT_DIR}/{recording_id:03d}_agent.wav", full_audio, sr) # sampling rate from agent_audio .


    # Add complete response to history so follow-up questions have context
    history.append({"role": "assistant", "content": full_response})
    return full_response

# ──────────────────────────────────────────────
#  Main Loop
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Continuous Voice Agent ===")
    print("Sage 'stop' oder 'ende' zum Beenden.\n")
    print("Pipeline: Mikrofon → Whisper(STT) → Claude(Streaming) → TTS(satzweise) → Lautsprecher\n")

    conversation_history = []
    running = True
    turn_id = 0

    while running:
        turn_id += 1
        filepath = record_audio(recording_id=turn_id, max_duration=20)

        if filepath is None:
            print("[Keine Eingabe erkannt, nächster Versuch...]")
            turn_id -= 1
            continue

        # ── Ab hier wartet der User ──
        t_start = time.time()

        # Whisper: Audio → Text (API call, nicht die Aufnahme)
        t_stt_start = time.time()
        user_text = transcribe_to_text(filepath)
        t_stt = time.time() - t_stt_start

        print(f"[USER]  {user_text}")

        words_in_text = user_text.lower().translate(str.maketrans("", "", string.punctuation)).split() # Removes ! and . from words
        if any(word in words_in_text for word in config.STOP_WORDS):
            print("\n[Agent beendet. Tschüss!]")
            agent_audio("Tschüss! Bis zum nächsten Mal.", speak=True, save=True,
                        filepath=f"{config.OUTPUT_DIR}/{turn_id:03d}_agent.wav")
            running = False
            continue

        print("[AGENT streamt...]")
        # LLM + TTS: Text → Streaming → Satzweise Audio
        t_llm_start = time.time()
        agent_text = stream_and_speak(user_text, recording_id=turn_id, history=conversation_history)
        t_llm_tts = time.time() - t_llm_start
        print(f"\n[AGENT komplett] {agent_text}")
        t_total = time.time() - t_start
        print(f"[Timing] STT: {t_stt:.1f}s | LLM+TTS: {t_llm_tts:.1f}s | Total: {t_total:.1f}s")

    print(f"\nGespräch beendet. {len(conversation_history) // 2} Austausche.")


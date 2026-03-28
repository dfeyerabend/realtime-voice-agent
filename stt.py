"""Speech-to-Text: Microphone recording + Whisper transcription."""

import numpy as np
import sounddevice as sd
import soundfile as sf
import dotenv
from openai import OpenAI

import config

dotenv.load_dotenv()
openai_client = OpenAI()  # Needs OPENAI_API_KEY in .env


def record_audio(
        recording_id: int = 1,
        max_duration: int = config.MAX_RECORDING_DURATION,
        sample_rate: int = config.SAMPLE_RATE,
        silence_threshold: float = config.SILENCE_THRESHOLD,
        silence_duration: float = config.SILENCE_DURATION,
        verbose: bool = False
) -> str | None:
    """
    Records audio from the microphone until silence is detected.

    Two-phase state machine:
      Phase 1 (waiting):   Ignores silence, waits for real speech.
      Phase 2 (recording): Records audio, stops after consecutive seconds
                           of silence (silence_duration) or after the maximum
                           recording duration has been reached (max_duration).

    Args:
        recording_id:      Turn number, used for the filename (e.g. 001_user.wav, 002_user.wav).
        max_duration:       Maximum recording length in seconds. Safety limit — recording stops
                           even if the user is still speaking.
        sample_rate:        Audio sample rate in Hz. 16000 = 16kHz, standard for speech recognition.
        silence_threshold:  Minimum average amplitude to count as "sound". Below this = silence.
                           Hardware-dependent — may need calibration per microphone.
        silence_duration:   How many consecutive seconds of silence (in Phase 2) before
                           recording stops. E.g. 2.0 = stops 2 seconds after user stops talking.

    Returns:
        Filepath to saved WAV file (e.g. "./recordings/001_user.wav"),
        or None if no speech was detected within max_duration.
    """

    filepath = f"{config.RECORDING_DIR}/{recording_id:03d}_user.wav"

    # Each chunk = 100ms of audio. At 16kHz sample rate that's 1600 samples.
    chunk_size = int(sample_rate * 0.1)
    chunks = []  # Collects ALL audio chunks, trimmed later

    silent_time = 0.0       # Seconds of consecutive silence AFTER speech started
    total_time = 0.0        # Total elapsed time (safety limit)
    speaking_time = 0.0     # Seconds of consecutive sound — must reach MIN_SPEAKING_DURATION
    has_spoken = False       # Phase flag: False=waiting, True=recording. Flips once, never back.
    speech_start_index = 0   # Which chunk index to keep from (trims leading silence)

    if verbose:
        print(f"\n  ⚙ max {max_duration}s, Stille-Timeout {int(silence_duration)}s")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:

        # Reading and discarding 5 chunks (~500ms) clears the buffer so
        # calibration measures actual room silence, not leftover playback.
        for _ in range(5):
            stream.read(chunk_size)

        # ── Ambient noise calibration (~1 second) ──────────────
        calibration_chunks = 10  # 10 chunks × 100ms = 1 second
        noise_levels = []

        if verbose:
            print("  ⚙ Kalibriere Mikrofon...", end=" ", flush=True)
        for _ in range(calibration_chunks):
            data, _ = stream.read(chunk_size)
            noise_levels.append(np.abs(data).mean())

        ambient = np.mean(noise_levels)
        silence_threshold = max(ambient * 2, 0.003)  # 2x ambient, minimum floor 0.003
        if verbose:
            print(f"Ambient: {ambient:.4f}, Threshold: {silence_threshold:.4f}")

        print("🎤 Bitte sprechen...")
        while total_time < max_duration:
            data, _ = stream.read(chunk_size)
            chunks.append(data.copy())

            # Average absolute amplitude. Typical: silence ~0.001, speech ~0.01-0.05
            volume = np.abs(data).mean()

            if volume >= silence_threshold:
                # ── Sound detected ──
                speaking_time += 0.1
                silent_time = 0.0  # Reset silence counter

                if not has_spoken and speaking_time >= config.MIN_SPEAKING_DURATION:
                    # Enough consecutive sound to count as real speech (not a cough/click).
                    # Transition: Phase 1 → Phase 2. Silence timer is now armed.
                    has_spoken = True

                    # Mark where speech began, minus ~500ms buffer so first syllable isn't clipped
                    speech_start_index = max(0, len(chunks) - int(config.MIN_SPEAKING_DURATION / 0.1) - 5)

            else:
                # ── Silence detected ──
                if has_spoken:
                    # Phase 2: was speaking, now quiet → count towards stop threshold
                    silent_time += 0.1
                else:
                    # Phase 1: still waiting → reset so short noise bursts don't accumulate
                    # Example: cough at t=2s and cough at t=7s should NOT sum to 0.5s
                    speaking_time = 0.0

            total_time += 0.1

            # ── Stop condition: spoke, then silent long enough ──
            if has_spoken and silent_time >= silence_duration:
                if verbose:
                    print(f"  ⚙ Stille erkannt nach {total_time:.1f}s")
                break

    # ── Post-recording ──

    if not has_spoken:
        # Phase 1 never ended → no speech → don't send silence to Whisper
        print("   Keine Sprache erkannt.")
        return None

    if total_time >= max_duration:
        if verbose:
            print(f"  ⚙ Max-Dauer erreicht ({max_duration}s)")

    # Trim leading silence so Whisper doesn't hallucinate on empty audio
    audio = np.concatenate(chunks[speech_start_index:])
    sf.write(filepath, audio, sample_rate)
    if verbose:
        print(f"  ⚙ Gespeichert: {filepath}")
    return filepath


def transcribe_to_text(filepath: str) -> str:
    """Sends a WAV file to OpenAI Whisper API. Returns transcribed text."""
    with open(filepath, 'rb') as f:
        result = openai_client.audio.transcriptions.create(
            model=config.WHISPER_MODEL,
            file=f,
            language="de"
        )
    return result.text


# ── Standalone test: record + transcribe ──
if __name__ == "__main__":
    print("=== STT Test: Aufnahme + Transkription ===\n")

    filepath = record_audio(recording_id=999, verbose=True)

    if filepath is None:
        print("Keine Aufnahme — nichts zu transkribieren.")
    else:
        text = transcribe_to_text(filepath)
        print(f"\nTranskription: {text}")

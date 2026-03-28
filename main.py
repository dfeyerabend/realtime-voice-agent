"""Voice Agent — CLI Voice Loop.

Microphone → Whisper → Claude Streaming → TTS → Speaker.
Imports individual modules: stt, tts, agent.
"""

import time
import string
import numpy as np

import config
import stt
import tts
import agent

# Set to True to see calibration values and recording details in stt.py
VERBOSE = False


def voice_turn(turn_id: int, history: list, verbose: bool = False) -> bool:
    """
    One complete conversation turn: Recording → Transcription → Agent → Speech.

    Returns:
        True  → continue
        False → user wants to stop (stop word detected)
    """

    # ── Recording ──
    filepath = stt.record_audio(recording_id=turn_id, verbose=verbose)

    if filepath is None:
        print("   Keine Eingabe erkannt, nächster Versuch...\n")
        return True  # Keep going, try again

    if verbose:
        print(f"  ⚙ Aufnahme: {filepath}")

    # ── From here the user is waiting ──
    t_start = time.time()

    # ── STT: Audio → Text ──
    t_stt_start = time.time()
    user_text = stt.transcribe_to_text(filepath)
    t_stt = time.time() - t_stt_start

    if verbose:
        print(f"  ⚙ STT: {t_stt:.1f}s")

    print(f"Du:     {user_text}\n")

    # ── Stop-Word Check ──
    # Strip punctuation first so "Ende!" matches "ende"
    words = user_text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    if verbose:
        matched = [w for w in words if w in config.STOP_WORDS]
        if matched:
            print(f"  ⚙ Stop-Word erkannt: {matched}")
        else:
            print(f"  ⚙ Wörter geprüft: {words} — kein Stop-Word")
    if any(word in words for word in config.STOP_WORDS):
        print("\n👋 Tschüss!")
        audio, sr = tts.speak("Tschüss! Bis zum nächsten Mal.")
        tts.save_audio(audio, sr, f"{config.OUTPUT_DIR}/{turn_id:03d}_agent.wav")
        return False  # Signal to stop

    # ── Agent + TTS: Streaming with per-sentence playback ──
    print("Agent:")
    t_llm_start = time.time()

    audio_chunks = []
    sr = None

    # agent.stream_text() yields one sentence at a time via generator.
    # Each iteration: Claude has finished a sentence → we speak it immediately.
    # While sd.play() is blocking, Claude might already be generating the next sentence
    # (the stream stays open in agent.py, buffering tokens).
    for sentence in agent.stream_text(user_text, history):
        print(f"  {sentence}", flush=True)
        audio, sr = tts.speak(sentence)  # Plays each sentence immediately
        audio_chunks.append(audio)

    t_llm_tts = time.time() - t_llm_start

    # ── Save complete audio as one file ──
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        output_path = f"{config.OUTPUT_DIR}/{turn_id:03d}_agent.wav"
        tts.save_audio(full_audio, sr, output_path)
        if verbose:
            print(f"  ⚙ Audio gespeichert: {output_path}")

    t_total = time.time() - t_start
    print(f"\n  ⏱ STT: {t_stt:.1f}s | LLM+TTS: {t_llm_tts:.1f}s | Total: {t_total:.1f}s\n")

    return True


if __name__ == "__main__":
    print("═══════════════════════════════════════")
    print("  Voice Agent")
    print("  Sage 'stop' oder 'ende' zum Beenden")
    print("═══════════════════════════════════════\n")

    conversation_history = []
    turn_id = 0

    while True:
        turn_id += 1
        keep_going = voice_turn(turn_id, conversation_history, verbose=True)
        if not keep_going:
            break

    print(f"\nGespräch beendet. {len(conversation_history) // 2} Austausche.")
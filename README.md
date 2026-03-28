# Realtime Voice Agent

> A streaming voice conversation agent that listens, thinks, and speaks — built with Whisper STT, Claude, and OpenAI TTS in a real-time pipeline.

**Showcase project** — built during the [Morphos GmbH](https://adz-weiterbildung.de) advanced AI engineering program (March 2026).

---

## What This Is

A voice-to-voice conversation system where three AI models work in sequence:

**You speak → Whisper transcribes → Claude thinks → TTS speaks back**

The key feature is **streaming response**: instead of waiting for Claude's full answer before speaking, the agent speaks each sentence the moment it's complete — reducing perceived latency significantly.

```
Microphone
    │
    ▼
┌───────────────────────────────┐
│   Whisper STT (OpenAI API)    │  Audio → Text
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│   Claude (Anthropic API)      │  Text → Text (Streaming via SSE)
└──────────────┬────────────────┘
               │  sentence by sentence
               ▼
┌───────────────────────────────┐
│   TTS (OpenAI API)            │  Text → Audio → Playback
└───────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| Streaming responses | Claude streams token by token; TTS fires per sentence |
| Automatic mic calibration | Measures ambient noise before each turn, adapts silence threshold dynamically |
| Silence detection | Two-phase state machine — waits for speech, then stops after configurable silence |
| Leading silence trimming | Only speech audio is sent to Whisper, preventing hallucinations on empty input |
| Conversation memory | Full conversation history is passed to Claude for follow-up questions |
| Audio logging | Each turn is saved as numbered WAV files (`001_user.wav`, `001_agent.wav`) |
| Pipeline timing | STT, LLM+TTS, and total latency measured per turn |
| Verbose debug mode | `VERBOSE = True` in main.py reveals calibration values, file paths, stop-word checks |

---

## Project Structure

```
realtime-voice-agent/
├── stt.py                  # Speech-to-Text (recording + Whisper)
├── tts.py                  # Text-to-Speech (OpenAI TTS)
├── agent.py                # Claude streaming (text only, no audio)
├── main.py                 # CLI voice loop (orchestrates stt, tts, agent)
├── config.py               # All model settings, audio parameters, prompts
├── cleanup.py              # Utility: deletes all WAV files
├── requirements.txt
├── .env                    # API keys (not tracked)
├── .env_example            # Template for .env
├── .gitignore
├── recordings/             # User audio (per-turn WAV files)
│   └── .gitkeep
└── outputs/                # Agent audio (per-turn WAV files)
    └── .gitkeep
```

Each pipeline step is a standalone module that can be tested independently.

---

## Getting Started

### Prerequisites

- Python 3.12+
- A working microphone
- API keys for OpenAI and Anthropic

### Installation

```bash
git clone https://github.com/yourusername/realtime-voice-agent.git
cd realtime-voice-agent

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env_example .env
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Full Voice Agent

```bash
python main.py
```

Speak into your microphone. The agent responds via audio. Say "stop" or "ende" to end the conversation.

Expected output:

```
═══════════════════════════════════════
  Voice Agent
  Sage 'stop' oder 'ende' zum Beenden
═══════════════════════════════════════

🎤 Bitte sprechen...
Du:     Erzähle mir eine Geschichte über eine kosmische Eule.

Agent:
  Im weiten Universum lebte eine magische Eule namens Stella...
  Sie flog zwischen den Galaxien umher und sammelte Träume...
  Eines Nachts entdeckte sie einen einsamen Asteroiden...

  ⏱ STT: 1.5s | LLM+TTS: 34.3s | Total: 35.8s

🎤 Bitte sprechen...
Du:     Ende.

👋 Tschüss!
Gespräch beendet. 1 Austausche.
```

Set `VERBOSE = True` in main.py to see debug details (calibration values, file paths, stop-word matching).

### Test Individual Pipeline Steps

Each module runs standalone for isolated testing:

**Speech-to-Text only** — records from microphone, transcribes, prints text:

```bash
python stt.py
```

```
=== STT Test ===

  ⚙ max 20s, Stille-Timeout 2s
  ⚙ Kalibriere Mikrofon... Ambient: 0.0141, Threshold: 0.0281
🎤 Bitte sprechen...
  ⚙ Stille erkannt nach 5.2s
  ⚙ Gespeichert: ./recordings/999_user.wav

Transkription: Das ist ein Test.
```

Note: stt.py runs with `verbose=True` by default for debugging.

**Text-to-Speech only** — generates and plays a test sentence:

```bash
python tts.py
```

```
=== TTS Test ===

Generiere Audio...
Fertig.
```

You should hear "Hallo, ich bin der Voice Agent. Das ist ein Test." through your speakers.

**Claude streaming only** — text chat in the terminal, no audio:

```bash
python agent.py
```

```
=== Agent Test: Text-Chat mit Claude ===
Tippe 'quit' zum Beenden.

Du: Was ist Python?
Agent: Python ist eine vielseitige Programmiersprache. Sie wird häufig in der Webentwicklung und Datenanalyse eingesetzt. Besonders beliebt ist sie auch im Bereich maschinelles Lernen.
Du: quit
```

**Clean up audio files:**

```bash
python cleanup.py
```

---

## How Streaming Works

The non-streaming approach waits for Claude's complete response before sending it to TTS — a 3-sentence answer takes ~5-10 seconds before any audio plays.

The streaming approach sends each sentence to TTS the moment it's complete:

```
Claude generates:  [Sentence 1 ......] [Sentence 2 ......] [Sentence 3 ......]
TTS plays:         [Sentence 1 plays ▶]
                                       [Sentence 2 plays ▶]
                                                           [Sentence 3 plays ▶]
```

First audio plays after ~1-2 seconds instead of ~10.

Sentence boundaries are detected with regex (`[.!?]\s`) on the incoming token stream. Each completed sentence is yielded by `agent.py` as a generator, and the caller (`main.py`) immediately sends it to TTS.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Anthropic API** | Agent reasoning (Claude Sonnet 4, streaming via SSE) |
| **OpenAI Whisper API** | Speech-to-text transcription |
| **OpenAI TTS API** | Text-to-speech synthesis |
| **sounddevice** | Microphone recording and audio playback |
| **soundfile** | WAV file I/O |
| **Python 3.12** | Runtime |

---

## Author

**Dennis Feyerabend**
March 2026

---

## License

MIT

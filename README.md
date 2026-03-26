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
│   Encoder-Decoder             │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│   Claude (Anthropic API)      │  Text → Text (Streaming via SSE)
│   Decoder-Only                │
└──────────────┬────────────────┘
               │  sentence by sentence
               ▼
┌───────────────────────────────┐
│   TTS (OpenAI API)            │  Text → Audio → Playback
│   Encoder-Decoder             │
└───────────────────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| Streaming responses | Claude streams token by token; TTS fires per sentence |
| Silence detection | Two-phase state machine — waits for speech, then stops after configurable silence |
| Leading silence trimming | Only speech audio is sent to Whisper, preventing hallucinations on empty input |
| Conversation memory | Full conversation history is passed to Claude for follow-up questions |
| Audio logging | Each turn is saved as numbered WAV files (`001_user.wav`, `001_agent.wav`) |
| Pipeline timing | STT, LLM+TTS, and total latency measured per turn |

---

## Project Structure

```
realtime-voice-agent/
├── agent_pipeline.py       # Main pipeline: record → transcribe → stream → speak
├── config.py               # All model settings, audio parameters, prompts
├── cleanup.py              # Utility: deletes all WAV files from recordings/ and outputs/
├── requirements.txt
├── .env                    # API keys (not tracked)
├── .env_example            # Template for .env
├── .gitignore
├── recordings/             # User audio (per-turn WAV files)
│   └── .gitkeep
└── outputs/                # Agent audio (per-turn WAV files)
    └── .gitkeep
```

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

### Usage

```bash
python agent_pipeline.py
```

Speak into your microphone. The agent responds via audio. Say "stop" or "ende" to end the conversation.

```bash
# Clean up recorded audio files
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

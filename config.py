# Models
WHISPER_MODEL = "whisper-1"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"

# Audio Settings
SAMPLE_RATE = 16000
MAX_RECORDING_DURATION = 20
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2.0
MIN_SPEAKING_DURATION = 0.5

# Agent
SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Sprachassistent. Deine Antworten werden vorgelesen. "
    "Antworte kurz auf Deutsch in maximal 3 Sätzen. "
    "Verwende KEINE Listen, Aufzählungen, Markdown oder Sonderzeichen. "
    "Schreibe nur fließenden Text."
)
MAX_TOKENS = 500

# Paths
RECORDING_DIR = "./recordings"
OUTPUT_DIR = "./outputs"

# Stop words
STOP_WORDS = ["stop", "stopp", "ende", "aufhören", "tschüss", "quit"]
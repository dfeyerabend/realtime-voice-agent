"""Claude Streaming Agent — text only, no audio."""

import re
import dotenv
from anthropic import Anthropic

import config

dotenv.load_dotenv()
anthropic_client = Anthropic()  # Needs ANTHROPIC_API_KEY in .env


def stream_text(user_text: str, history: list):
    """
    Streams Claude's response and yields each completed sentence.

    Uses regex [.!?]\\s to detect sentence boundaries in the token stream.
    Each sentence is yielded immediately once complete.

    After the final yield, the full response is appended to history.

    Usage:
        history = []
        for sentence in stream_text("Hallo", history):
            print(sentence)
        # history now contains user + assistant messages
    """

    # Add user message to history BEFORE streaming starts
    history.append({"role": "user", "content": user_text})

    current_sentence = ""   # Buffer — accumulates tokens until a sentence boundary
    full_response = ""      # Complete response — needed for conversation history
    sentence_end = re.compile(r'[.!?]\s')  # Sentence boundary = punctuation + whitespace

    # Open streaming connection to Claude
    with anthropic_client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=config.SYSTEM_PROMPT,
        messages=history
    ) as stream:

        for text in stream.text_stream:
            # Claude sends small chunks (often single words or fragments)
            current_sentence += text
            full_response += text

            # Check if buffer contains a complete sentence
            match = sentence_end.search(current_sentence)
            while match:
                # Extract everything up to and including the boundary
                sentence = current_sentence[:match.end()].strip()
                # Keep everything after the boundary for the next sentence
                current_sentence = current_sentence[match.end():]

                if sentence:
                    yield sentence  # Hand sentence to caller, pause here

                # Check again — buffer might contain multiple sentences
                match = sentence_end.search(current_sentence)

    # Leftover text without trailing whitespace after punctuation
    # Example: Claude ends with "Das wars" (no period + space at the end)
    if current_sentence.strip():
        yield current_sentence.strip()

    # Add complete response to history so follow-up questions have context
    history.append({"role": "assistant", "content": full_response})


# ── Standalone test: text chat with Claude in terminal ──
if __name__ == "__main__":
    print("=== Agent Test: Text-Chat mit Claude ===")
    print("Tippe 'quit' zum Beenden.\n")

    history = []

    while True:
        user_input = input("Du: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        print("Agent: ", end="", flush=True)
        for sentence in stream_text(user_input, history):
            print(sentence, end=" ", flush=True)
        print()

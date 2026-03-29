"""Unified Entrypoint — FastAPI + Gradio on a single port.

Mounts the Gradio demo from app.py onto a root FastAPI app,
and the API from server.py under /api.
Single process, single port — suitable for Railway deployment.

Routes:
    /           → Gradio Web UI
    /api/       → FastAPI root (welcome message)
    /api/health → Healthcheck
    /api/tts    → Text-to-Speech
    /api/stt    → Speech-to-Text
    /api/agent/text → Agent text endpoint
    /api/pipeline   → Full pipeline
    /api/docs   → Swagger UI for API endpoints

Run:
    python deploy.py
    OR:
    uvicorn deploy:app --host 0.0.0.0 --port 8000
"""

import signal
import sys
import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI

# Import existing apps — no modifications to server.py or app.py
from server import app as api_app       # FastAPI instance with all endpoints
from app import demo, graceful_shutdown  # Gradio Blocks instance + shutdown handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deploy")


# ── Build unified app ─────────────────────────────────────

# New root app — owns the process
main_app = FastAPI(title="Realtime Voice Agent — Unified")

# Mount server.py's FastAPI under /api
main_app.mount("/api", api_app)

# Mount Gradio at root — serves the web UI
app = gr.mount_gradio_app(main_app, demo, path="/")


# ── Signals ───────────────────────────────────────────────

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


# ── Launch ────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting unified server: Gradio (/) + FastAPI (/api)")
    uvicorn.run(
        "deploy:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

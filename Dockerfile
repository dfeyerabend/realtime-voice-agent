FROM python:3.12-slim

# sounddevice imports libportaudio even though app.py doesn't use the mic.
# Without this, the import chain (stt.py → sounddevice) crashes on startup.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libportaudio2 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching — only re-runs when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY stt.py tts.py agent.py config.py cleanup.py ./
COPY main.py server.py app.py deploy.py ./

# Create directories that modules might reference (even if unused on server)
RUN mkdir -p recordings outputs

# Railway sets PORT env var, but we hardcode 8000 in deploy.py.
# Railway will map its external port to whatever we EXPOSE here.
EXPOSE 8000

# No .env in container — API keys come from Railway environment variables
CMD ["python", "deploy.py"]

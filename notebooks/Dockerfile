# Dockerfile  – compact, comment-free CMD
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config libffi-dev \
        libasound2-dev libportaudio2 libportaudiocpp0 portaudio19-dev \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
CMD ["jupyter-lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--ServerApp.token=","--ServerApp.password="]

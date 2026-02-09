FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    accelerate \
    transformers \
    diffusers \
    sentencepiece \
    opencv-python-headless \
    pillow \
    einops \
    imageio \
    imageio-ffmpeg \
    decord \
    easydict \
    onnxruntime-gpu \
    insightface \
    mediapipe

WORKDIR /app/Wan2.2
RUN pip install --no-cache-dir -e .

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone the official WAN 2.2 repo
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub[cli] \
    torch torchvision torchaudio \
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

# Install WAN 2.2 dependencies from repo
WORKDIR /app/Wan2.2
RUN pip install --no-cache-dir -e .

# Download model weights (this takes a while - ~28GB)
RUN huggingface-cli download Wan-AI/Wan2.2-Animate-14B \
    --local-dir /app/Wan2.2/Wan2.2-Animate-14B

WORKDIR /app

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

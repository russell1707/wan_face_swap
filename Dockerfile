FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/runpod-volume/.cache/torch
ENV HF_HOME=/runpod-volume/.cache/huggingface

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone WAN 2.2 repo
RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

# Install PyTorch first (CUDA 12.1)
RUN pip3 install --no-cache-dir \
    torch>=2.4.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install WAN 2.2 base requirements
RUN pip3 install --no-cache-dir -r /app/Wan2.2/requirements.txt || true

# Install flash-attn separately (can be finicky)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || true

# Install animate-specific dependencies
RUN pip3 install --no-cache-dir \
    loguru \
    decord \
    peft \
    onnxruntime-gpu \
    pandas \
    matplotlib \
    sentencepiece \
    huggingface_hub[cli]

# Install SAM2 (required for animate preprocessing)
RUN pip3 install --no-cache-dir "sam2 @ git+https://github.com/facebookresearch/sam2.git@0e78a118995e66bb27d78518c4bd9a3e95b4e266"

# Install RunPod SDK
RUN pip3 install --no-cache-dir runpod>=1.6.0

# Copy worker handler
COPY handler.py .

CMD ["python3", "handler.py"]

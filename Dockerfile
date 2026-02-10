FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/runpod-volume/.cache/torch
ENV HF_HOME=/runpod-volume/.cache/huggingface

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
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

RUN git clone https://github.com/Wan-Video/Wan2.2.git /app/Wan2.2

RUN pip3 install --no-cache-dir \
    torch>=2.4.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r /app/Wan2.2/requirements.txt || true

RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "flash-attn install failed, will use default attention"

RUN pip3 install --no-cache-dir xformers || echo "xformers install failed"

RUN pip3 install --no-cache-dir \
    loguru \
    decord \
    peft \
    onnxruntime-gpu \
    pandas \
    matplotlib \
    sentencepiece \
    huggingface_hub[cli] \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    opencv-python-headless \
    Pillow \
    moviepy \
    scipy \
    scikit-image \
    librosa \
    easydict \
    ftfy \
    imageio[ffmpeg] \
    imageio-ffmpeg \
    tokenizers \
    tqdm \
    "numpy>=1.23.5,<2"

RUN pip3 install --no-cache-dir "git+https://github.com/facebookresearch/sam2.git"

RUN pip3 install --no-cache-dir runpod>=1.6.0

COPY handler.py .

CMD ["python3", "handler.py"]

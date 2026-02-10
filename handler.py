import runpod
import os
import sys
import subprocess
import base64
import tempfile
import uuid

sys.path.insert(0, "/app/Wan2.2")

CKPT_DIR = "/runpod-volume/Wan2.2-Animate-14B"
MODEL_READY = False


def ensure_model_downloaded():
    global MODEL_READY
    if MODEL_READY:
        return True

    marker = os.path.join(CKPT_DIR, ".download_complete")
    if os.path.exists(marker):
        MODEL_READY = True
        return True

    print("Downloading WAN 2.2 Animate model weights... This will take a few minutes on first run.")
    os.makedirs(CKPT_DIR, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "Wan-AI/Wan2.2-Animate-14B",
            local_dir=CKPT_DIR
        )
        with open(marker, "w") as f:
            f.write("done")
        MODEL_READY = True
        print("Model download complete!")
        return True
    except Exception as e:
        print(f"Model download failed: {e}")
        return False


def save_base64_file(b64_data, suffix):
    file_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{suffix}")
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return file_path


def encode_file_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    job_input = job["input"]

    if not ensure_model_downloaded():
        return {"error": "Model weights not available. Download failed."}

    character_image_b64 = job_input.get("character_image")
    driver_video_b64 = job_input.get("driver_video")
    width = job_input.get("width", 1280)
    height = job_input.get("height", 720)

    if not character_image_b64 or not driver_video_b64:
        return {"error": "Both character_image and driver_video are required"}

    image_path = save_base64_file(character_image_b64, ".png")
    video_path = save_base64_file(driver_video_b64, ".mp4")
    output_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")

    try:
        preprocess_dir = os.path.join(tempfile.gettempdir(), f"preprocess_{uuid.uuid4()}")
        os.makedirs(preprocess_dir, exist_ok=True)

        preprocess_cmd = [
            "python3", "/app/Wan2.2/wan/modules/animate/preprocess/preprocess_data.py",
            "--ckpt_path", os.path.join(CKPT_DIR, "process_checkpoint"),
            "--video_path", video_path,
            "--refer_path", image_path,
            "--save_path", preprocess_dir,
            "--resolution_area", str(width), str(height),
            "--retarget_flag"
        ]

        result = subprocess.run(preprocess_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"error": f"Preprocessing failed: {result.stderr}"}

        generate_cmd = [
            "python3", "/app/Wan2.2/generate.py",
            "--task", "animate",
            "--size", f"{width}*{height}",
            "--ckpt_dir", CKPT_DIR,
            "--offload_model", "True",
            "--convert_model_dtype",
            "--video_path", video_path,
            "--refer_path", image_path,
            "--output_path", output_path
        ]

        result = subprocess.run(generate_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            return {"error": f"Inference failed: {result.stderr}"}

        if not os.path.exists(output_path):
            for f in os.listdir("/app/Wan2.2"):
                if f.endswith(".mp4"):
                    output_path = os.path.join("/app/Wan2.2", f)
                    break

        if not os.path.exists(output_path):
            return {"error": "Output video not found"}

        output_b64 = encode_file_base64(output_path)
        return {"output_video": output_b64, "status": "completed"}

    except subprocess.TimeoutExpired:
        return {"error": "Processing timed out"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for f in [image_path, video_path]:
            if os.path.exists(f):
                os.remove(f)


runpod.serverless.start({"handler": handler})

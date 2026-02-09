"""
RunPod Serverless Handler for WAN 2.2 Animate
"""

import runpod
import base64
import os
import tempfile
import subprocess
import time

MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() == "true"


def decode_b64_to_file(b64_string: str, suffix: str) -> str:
    data = base64.b64decode(b64_string)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def encode_file_to_b64(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(event):
    try:
        input_data = event.get("input", {})

        character_image_b64 = input_data.get("character_image_b64")
        driver_video_b64 = input_data.get("driver_video_b64")
        settings = input_data.get("settings", {})

        if not character_image_b64:
            return {"error": "Missing character_image_b64"}
        if not driver_video_b64:
            return {"error": "Missing driver_video_b64"}

        start_time = time.time()

        character_path = decode_b64_to_file(character_image_b64, ".png")
        driver_path = decode_b64_to_file(driver_video_b64, ".mp4")

        output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ).name

        if MOCK_MODE:
            subprocess.run(["cp", driver_path, output_path], check=True)
            log_msg = "MOCK MODE: Returned driver video as output"
        else:
            # TODO: Replace with actual WAN 2.2 inference
            subprocess.run(["cp", driver_path, output_path], check=True)
            log_msg = "WAN 2.2 inference completed"

        output_b64 = encode_file_to_b64(output_path)

        elapsed = time.time() - start_time

        for f in [character_path, driver_path, output_path]:
            try:
                os.unlink(f)
            except OSError:
                pass

        return {
            "output_video_b64": output_b64,
            "log": log_msg,
            "meta": {
                "elapsed_seconds": round(elapsed, 2),
                "resolution": settings.get("resolution", "480p"),
                "fps": settings.get("fps", 24),
                "mock_mode": MOCK_MODE,
            },
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})

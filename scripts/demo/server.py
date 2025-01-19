import os
import base64
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from PIL import Image
from io import BytesIO
from scripts.demo.test_json import main
import sys

app = FastAPI()

# Configure the model with command line arguments
sys.argv = [
    "scripts.demo.test_json",
    "--output_dir", "/content",
    "--config_path", "configs/model/diffsensei.yaml",
    "--inference_config_path", "configs/inference/diffsensei.yaml",
    "--ckpt_path", "checkpoints/diffsensei",
    "--bit_8"
]

# Initialize the processing function with these configurations
process_json_data = main()

@app.post("/generate")
async def generate(data: dict):
    # Generate images
    images = process_json_data(data, "/content")
    
    if not images:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Image generation failed"}
        )
    
    # Convert images to base64
    base64_images = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        base64_images.append(img_str)
    
    return {
        "status": "success",
        "images": base64_images
    }
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List
import shutil
import os
import subprocess
import torch
from PIL import Image
import io
import cv2
import numpy as np
import sys
import zipfile
sys.path.append("/teamspace/studios/this_studio/ArrowDetection")
sys.path.append("/teamspace/studios/this_studio/ArrowDetection/ML")
from ML import InferencePipeline, label_decoder

app = FastAPI()

# Setup device and pipeline
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

inference_pipeline = InferencePipeline(device=device)

toate_clasele = [v for _, v in label_decoder.items()]
inverse_decoder = {k: v for v, k in label_decoder.items()}


# Image Detection Endpoint
@app.post("/detect/image/")
async def detect_objects_image(
    image: UploadFile = File(...),
    clase_interes: List[str] = Form(...),
    confidenta_model: float = Form(0.9),
    model_selector: str = Form("small")
):
    # Read the image data from the uploaded file
    image_data = await image.read()

    # Convert image to NumPy array
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Read image as a NumPy array (BGR format)

    # Decode classes
    clase_interes = [inverse_decoder[k] for k in clase_interes]

    # Process the image
    output_image, json_path_labels = inference_pipeline.predict_one_image(
        image=img,  # Pass the NumPy array directly to the pipeline
        confidenta=float(confidenta_model),
        clase_interes=clase_interes,
        model_type=model_selector
    )
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Convert the output image to a format suitable for saving (PIL or NumPy array to image)
    output_image_path = "/tmp/output_image.png"
    output_image_pil = Image.fromarray(output_image)  # Assuming the output is a NumPy array
    output_image_pil.save(output_image_path)

    zip_output_path = "/tmp/output_files.zip"
    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        zipf.write(output_image_path, arcname="output_image.png")  # Add the video file to the zip
        zipf.write(json_path_labels, arcname="output_img.json")  # Add the zip file to the zip

    # Return the processed image as a downloadable file
    # return FileResponse(output_image_path, media_type="image/png", filename="output_image.png")
    return FileResponse(zip_output_path, media_type="application/zip", filename="output_files.zip")


# Video Detection Endpoint
@app.post("/detect/video/")
async def detect_objects_video(
    video: UploadFile = File(...),
    clase_interes: List[str] = Form(...),
    confidenta_model: float = Form(0.8),
    fps: int = Form(15),
    model_selector: str = Form("small")
):
    # Save the uploaded video temporarily
    video_data = await video.read()
    video_path = "/tmp/temp_video.mp4"
    
    with open(video_path, "wb") as f:
        f.write(video_data)

    # Decode classes
    clase_interes = [inverse_decoder[k] for k in clase_interes]

    # Process the video
    video_displayed, zip_path = inference_pipeline.video_inference(
        video_name=video_path,
        batch_size=64,
        fps_final=fps,
        confidenta=float(confidenta_model),
        clase_interes=clase_interes,
        model_type=model_selector
    )

    fixed_path = video_displayed.replace(".mp4", "_fixed.mp4")

    # Run FFmpeg to re‑encode/remux into H.264 + AAC + yuv420p + faststart
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_displayed,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",              # re‑encode audio to AAC (even if silent)
        "-b:a", "128k",             # audio bitrate
        fixed_path
    ], check=True)

    zip_output_path = "/tmp/output_files.zip"
    with zipfile.ZipFile(zip_output_path, 'w') as zipf:
        zipf.write(fixed_path, arcname="output_video.mp4")  # Add the video file to the zip
        zipf.write(zip_path, arcname="annotated_video.zip")  # Add the zip file to the zip

    # Return the zip file containing both video and the zip
    return FileResponse(zip_output_path, media_type="application/zip", filename="output_files.zip")

    # Return the processed video as a downloadable zip file
    # return FileResponse(zip_path, media_type="application/zip", filename="annotated_video.zip")
    # return FileResponse(video_displayed, media_type="video/mp4", filename="output_video.mp4")

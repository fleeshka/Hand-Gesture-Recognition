from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from pydantic import BaseModel
import os
from model_downloader import maybe_download_model
from model_service import ModelService, get_model_path_from_env


log = logging.getLogger(__name__)

app = FastAPI(title="Mafia Gesture Recognition API", version="1.0.0")

log.info("🚀 Starting Mafia Gesture Recognition API")

# Allow communication from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve model path: supports remote/local URIs via downloader
MODEL_URI = os.getenv("MODEL_URI")  # Optional alternative to MODEL_PATH for remote/local copy
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

try:
    resolved_model_path = None
    if MODEL_URI:
        log.info(f"MODEL_URI provided, attempting to fetch: {MODEL_URI}")
        resolved_model_path = maybe_download_model(MODEL_URI, MODEL_PATH)
    else:
        resolved_model_path = MODEL_PATH

    log.info(f"Attempting to load model from: {resolved_model_path}")
    model_service = ModelService.get_instance(resolved_model_path)
    model_service.load()
    log.info("✅ Model loaded successfully")
except Exception as e:
    log.error(f"❌ Model not found or failed to load: {e}")
    log.warning("Using placeholder predictions - this is for demonstration only")
    model_service = None


class HealthResponse(BaseModel):
    status: str


def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize and normalize
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


@app.get("/")
async def root():
    log.info("Root endpoint accessed")
    return {
        "message": "Mafia Gesture Recognition API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    log.debug("Health check requested")
    return HealthResponse(status="healthy" if model_service and model_service.is_loaded() else "degraded")


class PredictResponse(BaseModel):
    predictions: list


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not model_service or not model_service.is_loaded():
        return JSONResponse(status_code=503, content={"detail": "Model not available"})

    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    if input_tensor is None:
        return JSONResponse(status_code=400, content={"detail": "Invalid image"})

    preds = model_service.predict(input_tensor)
    # Convert to list for JSON serialization
    return PredictResponse(predictions=preds.tolist())


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    try:
        log.info("Starting Uvicorn server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
    except Exception as e:
        log.critical(f"Failed to start server: {e}")
        raise

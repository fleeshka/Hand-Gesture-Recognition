from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import numpy as np
import cv2
# TensorFlow is imported lazily during model loading to allow running the API
# in demo mode on machines without TF installed.
tf = None
from pydantic import BaseModel
import os
import io
from typing import List, Optional

# MediaPipe will be imported lazily when needed to avoid heavy imports at startup

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Hand Gesture Recognition API", version="1.0.0")

log.info("🚀 Starting Hand Gesture Recognition API")

# Allow communication from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gesture classes (model trained for 3 classes: open hand, closed fist, ok)
GESTURE_CLASSES = [
    "open_hand",
    "closed_fist",
    "ok",
]

# Model loading: support MODEL_DIR (directory) and MODEL_FILE (specific path).
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MODEL_FILE = os.getenv("MODEL_FILE", None)

def find_model_path():
    # Candidate locations (first that exists will be used)
    candidates = []
    if MODEL_FILE:
        candidates.append(MODEL_FILE)
    # Common filenames inside MODEL_DIR
    candidates.append(os.path.join(MODEL_DIR, "keypoint_classifier.keras"))
    candidates.append(os.path.join(MODEL_DIR, "keypoint_classifier"))
    candidates.append(MODEL_DIR)

    # Try repository-relative location (useful for local development)
    # repo-relative path: go up three levels from api folder to repo root, then into models/
    repo_rel = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "keypoint_classifier", "keypoint_classifier.keras"
    )
    candidates.append(repo_rel)

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            continue
    return None

model = None
try:
    selected = find_model_path()
    if selected:
        log.info(f"Attempting to load model from: {selected}")
        try:
            # Import tensorflow only when we need to load the model
            import tensorflow as _tf
            tf = _tf
            model = tf.keras.models.load_model(selected)
            log.info("✅ Model loaded successfully")
            try:
                log.info(f"Model input shape: {model.input_shape}")
                log.info(f"Model output shape: {model.output_shape}")
            except Exception:
                pass
        except Exception as e:
            # If tensorflow isn't available or model fails to load, log and continue with demo mode.
            log.error(f"TensorFlow/model load error: {e}")
            model = None
    else:
        raise FileNotFoundError(f"No model file found. Checked candidates. Set MODEL_FILE or place model under {MODEL_DIR}")
except Exception as e:
    log.error(f"❌ Model load failed: {e}")
    log.warning("Using placeholder predictions - this is for demonstration only")
    model = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gesture_classes: List[str]


class PredictionResponse(BaseModel):
    gesture: str
    confidence: float
    all_predictions: Optional[List[dict]] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


def preprocess_image(
    image_bytes: bytes, target_size: tuple = (224, 224)
) -> Optional[np.ndarray]:
    """Preprocess image for model prediction"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("Failed to decode image")
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img = cv2.resize(img, target_size)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        log.error(f"Image preprocessing error: {e}")
        return None


def predict_gesture(image_bytes: bytes) -> dict:
    """Predict gesture from image bytes"""
    try:
        # If no model, return demo
        if model is None:
            return demo_prediction()

        # Decide whether model expects keypoints (e.g., input shape (None, 42))
        try:
            input_shape = model.input_shape
        except Exception:
            input_shape = None

        # If model expects 42 features, extract keypoints from image via MediaPipe
        if input_shape and len(input_shape) >= 2 and int(input_shape[1]) == 42:
            keypoints = extract_keypoints_from_image(image_bytes)
            if keypoints is None:
                return {"gesture": "unknown", "confidence": 0.0}
            predictions = model.predict(keypoints, verbose=0)
        else:
            # Fallback: treat model as image-based and preprocess image
            processed_image = preprocess_image(image_bytes)
            if processed_image is None:
                return {"gesture": "unknown", "confidence": 0.0}
            predictions = model.predict(processed_image, verbose=0)
        # Normalize outputs to probabilities (softmax) to get reliable confidences
        try:
            arr = np.array(predictions[0], dtype=np.float32)
        except Exception:
            arr = np.array(predictions, dtype=np.float32).reshape(-1)

        try:
            if tf is not None:
                probs = tf.nn.softmax(arr).numpy()
            else:
                exps = np.exp(arr - np.max(arr))
                probs = exps / np.sum(exps)
        except Exception:
            exps = np.exp(arr - np.max(arr))
            probs = exps / np.sum(exps)

        predicted_class_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_class_idx])

        # Get gesture name
        if predicted_class_idx < len(GESTURE_CLASSES):
            gesture_name = GESTURE_CLASSES[predicted_class_idx]
        else:
            gesture_name = f"class_{predicted_class_idx}"

        # Get all predictions for debugging
        all_predictions = []
        for i, prob in enumerate(probs):
            class_name = (
                GESTURE_CLASSES[i] if i < len(GESTURE_CLASSES) else f"class_{i}"
            )
            all_predictions.append({"class": class_name, "confidence": float(prob)})

        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "gesture": gesture_name,
            "confidence": confidence,
            "all_predictions": all_predictions[:3],  # Return top 3 predictions
        }

    except Exception as e:
        log.error(f"Prediction error: {e}")
        return {"gesture": "error", "confidence": 0.0}


def extract_keypoints_from_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Use MediaPipe to extract 21 hand landmarks (x,y) -> 42 values and return shape (1,42) array.

    Returns None if no hand detected or on error.
    """
    try:
        import mediapipe as mp

        mp_hands = mp.solutions.hands

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            log.error("Failed to decode image for keypoint extraction")
            return None

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None

            hand_landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand_landmarks.landmark:
                # Use normalized x,y coordinates as provided by MediaPipe
                coords.append(float(lm.x))
                coords.append(float(lm.y))

            if len(coords) != 42:
                log.error(f"Unexpected number of keypoints: {len(coords)}")
                return None

            arr = np.array(coords, dtype=np.float32).reshape(1, 42)
            return arr

    except Exception as e:
        log.error(f"Keypoint extraction error: {e}")
        return None


def demo_prediction() -> dict:
    """Generate a demo prediction when no model is loaded"""
    import random

    demo_gestures = ["thumbs_up", "thumbs_down", "peace", "ok", "fist", "open_palm"]
    gesture = random.choice(demo_gestures)
    confidence = round(random.uniform(0.7, 0.95), 2)

    all_predictions = []
    for g in demo_gestures:
        if g == gesture:
            all_predictions.append({"class": g, "confidence": confidence})
        else:
            all_predictions.append(
                {"class": g, "confidence": round(random.uniform(0.01, 0.3), 2)}
            )

    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "gesture": gesture,
        "confidence": confidence,
        "all_predictions": all_predictions[:3],
    }


@app.get("/")
async def root():
    log.info("Root endpoint accessed")
    return {
        "message": "Hand Gesture Recognition API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
        "model_loaded": model is not None,
        "gesture_classes": GESTURE_CLASSES,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    log.debug("Health check requested")
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        gesture_classes=GESTURE_CLASSES,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture_endpoint(file: UploadFile = File(...)):
    """Predict hand gesture from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        log.info(f"Processing image: {file.filename}, size: {file.size} bytes")

        # Read image data
        image_data = await file.read()

        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Make prediction
        if model:
            result = predict_gesture(image_data)
        else:
            result = demo_prediction()
            log.warning("Using demo prediction - no model loaded")

        log.info(
            f"Prediction: {result['gesture']} (confidence: {result['confidence']:.2f})"
        )

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Prediction endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )


@app.get("/gestures")
async def get_gesture_classes():
    """Get available gesture classes"""
    return {"gesture_classes": GESTURE_CLASSES, "count": len(GESTURE_CLASSES)}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Make sure this points to your file and app variable
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Enable auto-reload for development
    )

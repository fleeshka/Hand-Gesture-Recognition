from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import numpy as np
import cv2
import pickle
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

# Gesture classes (will be loaded from model's label encoder)
GESTURE_CLASSES = []

# Model loading: support MODEL_DIR (directory) and MODEL_FILE (specific path).
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MODEL_FILE = os.getenv("MODEL_FILE", "/app/models/gesture_lr.pkl")

def find_model_path():
    # Candidate locations (first that exists will be used)
    candidates = []
    if MODEL_FILE:
        candidates.append(MODEL_FILE)
    # Common filenames inside MODEL_DIR
    candidates.append(os.path.join(MODEL_DIR, "gesture_lr.pkl"))

    # Try repository-relative locations (useful for local development)
    # repo-relative path: go up three levels from api folder to repo root, then into models/
    repo_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
    )
    candidates.append(os.path.join(repo_root, "models", "gesture_lr.pkl"))
    candidates.append(os.path.join(repo_root, "notebooks", "gesture_lr.pkl"))

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            continue
    return None

# Model and label encoder
model = None
label_encoder = None

try:
    selected = find_model_path()
    if selected:
        log.info(f"Attempting to load model from: {selected}")
        try:
            with open(selected, "rb") as f:
                model_data = pickle.load(f)
                model = model_data["model"]
                label_encoder = model_data["label_encoder"]
                GESTURE_CLASSES = list(label_encoder.classes_)
            log.info("✅ Model loaded successfully (sklearn LogisticRegression)")
            log.info(f"Gesture classes: {GESTURE_CLASSES}")
            try:
                log.info(f"Model type: {type(model).__name__}")
                log.info(f"Number of classes: {len(GESTURE_CLASSES)}")
            except Exception:
                pass
        except Exception as e:
            log.error(f"Model load error: {e}")
            import traceback
            log.error(traceback.format_exc())
            model = None
            label_encoder = None
    else:
        raise FileNotFoundError(f"No model file found. Checked candidates. Set MODEL_FILE or place model under {MODEL_DIR}")
except Exception as e:
    log.error(f"❌ Model load failed: {e}")
    log.warning("Using placeholder predictions - this is for demonstration only")
    model = None
    label_encoder = None


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


def predict_gesture(image_bytes: bytes) -> dict:
    """Predict gesture from image bytes using sklearn LogisticRegression"""
    try:
        # If no model, return demo
        if model is None or label_encoder is None:
            return demo_prediction()

        keypoints = extract_keypoints_from_image(image_bytes)
        if keypoints is None:
            return {"gesture": "No hand detected", "confidence": 0.0}

        # Get prediction probabilities
        probs = model.predict_proba(keypoints)[0]
        
        # Get predicted class index
        predicted_class_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_class_idx])

        # Get gesture name from label encoder
        gesture_name = label_encoder.inverse_transform([predicted_class_idx])[0]

        # Get all predictions for debugging
        all_predictions = []
        for i, prob in enumerate(probs):
            class_name = label_encoder.inverse_transform([i])[0]
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
        import traceback
        log.error(traceback.format_exc())
        return {"gesture": "error", "confidence": 0.0}

def extract_keypoints_from_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """Use MediaPipe to extract 21 hand landmarks (x,y,z) + is_right_hand -> 64 values and return shape (1,64) array.

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

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:

            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None

            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label

            is_right = 1 if handedness_label == "Right" else 0

            # Нормализация относительно wrist (landmark[0]), как в notebook lr.ipynb
            wrist = hand_landmarks.landmark[0]
            wrist_x = wrist.x
            wrist_y = wrist.y
            wrist_z = wrist.z

            # Вычисляем расстояния от wrist до всех landmarks
            distances = []
            for lm in hand_landmarks.landmark:
                dist = np.linalg.norm([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])
                distances.append(dist)

            # Масштаб = максимальное расстояние
            scale = max(distances) if max(distances) > 1e-6 else 1e-6

            # Нормализованные координаты относительно wrist
            normalized_coords = []
            for lm in hand_landmarks.landmark:
                x = (lm.x - wrist_x) / scale
                y = (lm.y - wrist_y) / scale
                z = (lm.z - wrist_z) / scale
                normalized_coords.extend([x, y, z])

            # Добавляем is_right_hand в конец
            normalized_coords.append(is_right)

            arr = np.array(normalized_coords, dtype=np.float32).reshape(1, -1)
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
        "model_loaded": (model is not None and label_encoder is not None),
        "gesture_classes": GESTURE_CLASSES,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    log.debug("Health check requested")
    return HealthResponse(
        status="healthy" if (model is not None and label_encoder is not None) else "degraded",
        model_loaded=(model is not None and label_encoder is not None),
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
        if model and label_encoder:
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

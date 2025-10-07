from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
import numpy as np
import cv2
import tensorflow as tf
try:
    import keras
except Exception:
    keras = None
from pydantic import BaseModel, field_validator
from typing import List, Optional
import os


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

# Default path to project model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "keypoint_classifier", "keypoint_classifier.keras")
ALT_MODEL_PATH = os.path.join(os.getcwd(), "models", "keypoint_classifier", "keypoint_classifier.keras")

# Allow overriding via env
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
if not os.path.exists(MODEL_PATH) and os.path.exists(ALT_MODEL_PATH):
    log.warning(f"Primary model path not found: {MODEL_PATH}. Falling back to: {ALT_MODEL_PATH}")
    MODEL_PATH = ALT_MODEL_PATH

model = None
_load_errors: list[str] = []
log.info(f"Attempting to load model from: {MODEL_PATH} (exists={os.path.exists(MODEL_PATH)})")

def _try_load_with_keras(path: str):
    if keras is None:
        raise RuntimeError("keras not installed")
    # Prefer Keras 3 API if available
    if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
        return keras.saving.load_model(path)
    return keras.models.load_model(path)

try:
    model = _try_load_with_keras(MODEL_PATH)
    log.info("✅ Model loaded successfully via Keras")
except Exception as e:
    _load_errors.append(f"keras load failed: {e}")
    model = None

if model is None:
    try:
        # Some environments may still expose tf.keras
        if hasattr(tf, "keras"):
            model = tf.keras.models.load_model(MODEL_PATH)
            log.info("✅ Model loaded successfully via tf.keras")
        else:
            raise AttributeError("tf.keras is not available in this TensorFlow build")
    except Exception as e:
        _load_errors.append(f"tf.keras load failed: {e}")
        log.error("❌ Model not found or failed to load")
        for err in _load_errors:
            log.error(err)
        log.warning("Using placeholder predictions - this is for demonstration only")

def _load_labels(model_path: str) -> List[str]:
    # Priority: env LABELS -> labels file near model -> default indices
    env_labels = os.getenv("LABELS")
    if env_labels:
        return [lbl.strip() for lbl in env_labels.split(",") if lbl.strip()]
    candidates = [
        os.path.join(os.path.dirname(model_path), "labels.txt"),
        os.path.join(os.path.dirname(model_path), "labels.csv"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    # Support comma or newline separated
                    content = f.read().strip()
                    if "," in content and "\n" not in content:
                        labels = [x.strip() for x in content.split(",") if x.strip()]
                    else:
                        labels = [x.strip() for x in content.splitlines() if x.strip()]
                    if labels:
                        log.info(f"Loaded {len(labels)} labels from {cand}")
                        return labels
            except Exception as e:
                log.warning(f"Failed to read labels from {cand}: {e}")
    # Fallback default set; adjust in env or labels file for correctness
    return ["open_hand", "thumbs_up", "ok", "peace", "stop", "rock"]

LABELS = _load_labels(MODEL_PATH)


class HealthResponse(BaseModel):
    status: str


class KeypointsPayload(BaseModel):
    # 21 landmarks, each with x,y,z floats as produced by MediaPipe
    keypoints: List[List[float]]

    @field_validator("keypoints")
    @classmethod
    def validate_keypoints(cls, value):
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError("keypoints must be a non-empty list")
        if len(value) != 21:
            raise ValueError("keypoints must contain 21 landmarks")
        for lm in value:
            if not isinstance(lm, list) or len(lm) < 2:
                raise ValueError("each landmark must be a list with at least [x, y] (z optional)")
        return value


def preprocess_keypoints(keypoints: List[List[float]]):
    """Preprocess 21x(2-3) keypoints to model input shape (42).
    Use raw normalized x,y from MediaPipe (0..1) without centering to match training CSV.
    """
    arr = np.array(keypoints, dtype=np.float32)
    # Use only x,y to match 42-dim input (21*2)
    if arr.shape[1] >= 2:
        arr = arr[:, :2]
    else:
        # pad y with zeros if missing
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=np.float32)], axis=1)
    # flatten to 42-dim vector (x0,y0,x1,y1,...)
    flat = arr.reshape(-1)
    # expand batch dimension
    return np.expand_dims(flat, axis=0)


def extract_hand_keypoints_from_image_bytes(image_bytes: bytes) -> Optional[List[List[float]]]:
    try:
        import mediapipe as mp
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append([lm.x, lm.y, lm.z])
            return keypoints
    except Exception as e:
        log.error(f"Failed to extract keypoints: {e}")
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
    return HealthResponse(
        status="healthy" if model else "degraded",
    )


@app.post("/predict")
async def predict(file: UploadFile | None = File(default=None), payload: KeypointsPayload | None = None):
    if model is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    try:
        # If a file is provided, extract landmarks from the image
        if file is not None:
            image_bytes = await file.read()
            keypoints = extract_hand_keypoints_from_image_bytes(image_bytes)
            if keypoints is None:
                return {"gesture": None, "confidence": None}
            x = preprocess_keypoints(keypoints)
        else:
            if payload is None:
                return JSONResponse(status_code=400, content={"detail": "No input provided"})
            x = preprocess_keypoints(payload.keypoints)
        preds = model.predict(x, verbose=0)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0].astype(float).tolist()
        else:
            # fallback for unexpected model outputs
            probs = preds.astype(float).tolist()
        # Log top-3 for debugging
        try:
            top_indices = np.argsort(probs)[-3:][::-1]
            log.info(f"Top-3: " + ", ".join([f"{LABELS[i] if i < len(LABELS) else i}:{probs[i]:.3f}" for i in top_indices]))
        except Exception:
            pass
        best_idx = int(np.argmax(probs))
        label = LABELS[best_idx] if best_idx < len(LABELS) else str(best_idx)
        confidence = float(max(probs)) if isinstance(probs, list) and len(probs) > 0 else None
        return {
            "gesture": label,
            "confidence": confidence,
            "index": best_idx,
            "confidences": probs,
        }
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        return JSONResponse(status_code=400, content={"detail": "Invalid input or prediction error"})


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    try:
        log.info("Starting Uvicorn server...")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
    except Exception as e:
        log.critical(f"Failed to start server: {e}")
        raise

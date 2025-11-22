# Scripts Directory

Contains utility scripts for testing and validating the hand gesture recognition system.

## Files

* **`model_testing.py`**: Real-time testing of the trained model via webcam using OpenCV and MediaPipe.
* **`hand_tracking.py`**: Additional utilities for hand tracking.

### Model Testing (`model_testing.py`)

Real-time gesture recognition using a Logistic Regression model:

* **Load Model**: Loads a pickled classifier (`gesture_lr.pkl`) and label encoder.
* **Hand Detection**: MediaPipe Hands detects hands with configurable thresholds (detection: 0.8, tracking: 0.5).
* **Feature Extraction**: Extracts 21 hand landmarks (x, y, z), normalizes relative to the wrist, scales by max distance, and adds handedness (right=1, left=0). Generates a 64-dimensional feature vector.
* **Prediction**: Logistic Regression predicts the gesture. Measures inference latency for performance monitoring.
* **Visualization**: Displays predicted gesture, confidence, FPS, and latency on the mirrored webcam feed.
* **Performance Metrics**: Smooths FPS and latency over recent frames for stable display.
* **User Interface**: Runs until 'q' is pressed.

This script allows standalone testing of the model’s real-time performance without using an API.

### How to Test

1. Place the trained model file (`gesture_lr.pkl`) in the `models/` directory.
2. Update `MODEL_FILEPATH` in `scripts/model_testing.py` if needed:
   ```python
   MODEL_FILEPATH = "models/gesture_lr.pkl"
   ```
3. Run:
   ```bash
   python scripts/model_testing.py
   ```
4. The webcam feed will show:
   * Predicted gesture
   * Detection confidence
   * Inference latency
   * FPS
5. Press 'q' to exit.

# Hand-Gesture-Recognition

## Project Topic
**Hand Gesture Recognition (HGR)**

This project aims to develop a **real-time hand gesture recognition system** for the game *Sports Mafia*. The system captures video frames from a webcam, detects and processes hand images using computer vision, classifies gestures with a CNN, and translates them into in-game commands (e.g., ✊ for "mafia kills"). 

In the future, the system will be expanded to support **dynamic gestures** for richer interaction.

### Target Users
- **Beginner players:** Quickly learn and adapt to game rules through clear gestures.  
- **Experienced players:** Gain faster, more engaging ways to interact during the game.  
- **Sports Mafia federations/communities:** Improve clarity and understanding of gameplay during live broadcasts.  

Beyond *Mafia*, this system can benefit **gamers, accessibility users, and contactless interface users**.

---

# Approach
Based on a review of SOTA solutions and commercial competitors, the project uses a **landmark-based pipeline** for the MVP, focusing on static gestures.

**Key Components:**
- **Detection:** [MediaPipe Hands](https://mediapipe.dev/) for robust 21-keypoint detection, independent of background, lighting, and camera quality.  
- **Classification:** Custom MLP or MobileNetV2 fine-tuned on our dataset for gesture-to-command mapping.  
- **Deployment:** Docker for portability and easy distribution.  
- **Future scalability:** Temporal models (LSTM or Transformers) on landmark sequences to support dynamic gestures.

**Benefits of this approach:**
- Accuracy > 90%  
- Real-time performance (latency < 100ms)  
- Lightweight and efficient compared to heavy multimodal SOTA models like CLIP-LSTM or GestureGPT  
- Open-source and hardware-agnostic  
- Extensible for broader HCI applications beyond *Sports Mafia*

---

## Realization

The implementation consists of the following components:

### Data Collection and Preparation

- **Data Collection**: Hand gesture samples are captured using a webcam and MediaPipe Hands library. Each sample includes 21 hand landmarks (x, y, z coordinates) normalized relative to the wrist position and scaled by the maximum distance from the wrist. Additional features include handedness (left/right hand).
- **Gesture Classes**: The system recognizes 7 gestures: don, if, question, mafia, cool, civilian, and potentially others.
- **Dataset Aggregation**: Individual CSV files from different recording sessions are combined into a single training dataset located in `data/processed/data.csv`.

### Model Architecture and Training

- **Logistic Regression Model**: A multinomial logistic regression model is trained using scikit-learn with L2 regularization (C=10) and SAGA solver. This model serves as a baseline and is saved as `models/gesture_lr.pkl`.


### Real-Time Inference

- **Hand Detection**: MediaPipe Hands processes webcam frames to extract keypoints in real-time.
- **Feature Extraction**: Keypoints are normalized and flattened into a 64-dimensional feature vector.
- **Prediction**: The logistic regresion predicts the gesture class with confidence scores.

### API and Deployment

- **FastAPI Backend**: A REST API is implemented with endpoints for health checks, gesture prediction via image upload, and listing gesture classes.
- **Deployment**: The backend is containerized with Docker and can be run locally or deployed. A frontend interface (HTML/JS) provides a web-based demo for testing gesture recognition.

### Scripts and Tools

- **Notebooks**: Jupyter notebooks handle data collection (`dataset_creation.ipynb`), data preparation (`prepare_data_training.ipynb`), and model training (`model_training.ipynb`).
- **Scripts**: Utility scripts like `hand_tracking.py` and `model_testing.py` assist in testing and validation.

This realization achieves the project goals with efficient, real-time performance and cross-platform compatibility.

---

### Success Criteria

1. Accuracy: ≥80% classification accuracy on validation set
2. Robustness: works across different people, skin tones, lighting, and backgrounds
3. Performance
   * Prediction delay ≤ 200 ms per frame
   * Target ≥ 25 FPS on a standard laptop webcam
4. Usability & Stability: recognizes gestures consistently without false positives
5. Deployment: Successful containerization and API deployment for easy distribution and access

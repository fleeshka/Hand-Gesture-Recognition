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
<<<<<<< HEAD
5. Frontend: intuitive web interface with real-time recognition and gamification features

---

## Frontend

### Overview
The frontend is a modern, responsive web application built with HTML, CSS, and vanilla JavaScript. It provides a user-friendly interface for real-time hand gesture recognition and includes an interactive game mode with multiple difficulty levels.

### Structure

```
frontend/
├── css/
│   └── style.css    # All application styles
├── js/
│   ├── game.js      # Game logic and game mode functionality
│   └── home.js      # Home page gesture recognition logic
├── pages/           # HTML pages
│   ├── home.html    # Main recognition page
│   └── game.html    # Game mode page
├── index.html       # Entry point (redirects to pages/home.html)
├── Dockerfile       # Docker configuration for deployment
├── nginx.conf       # Nginx server configuration
└── package.json     # Node.js dependencies (if any)
```

### Features

#### 🏠 Home Page (`pages/home.html`)
- **Real-time Gesture Recognition**
  - Live camera feed with automatic frame processing
  - Displays detected gestures with confidence scores
  - Shows gesture count and average confidence statistics
  - Visual feedback with emoji-enhanced gesture names

- **Supported Gestures**
  - Basic gestures: Civilian 👍, Mafia 👎, Sheriff 👌, Don 🎩
  - Actions: If 🤙, Question ❓, Cool 🤘
  - Pronouns: You 🫵, Me 👉
  - Numbers: Zero 0️⃣, One 1️⃣, Two 2️⃣, Three 3️⃣, Four 4️⃣, Five 5️⃣

- **User Experience**
  - Clean, modern UI with animated gradient background
  - Responsive design that works on desktop and mobile
  - Camera access management with error handling
  - Status indicators and error messages

#### 🎮 Game Mode (`pages/game.html`)
- **Three Difficulty Levels**
  - **Easy (⭐)**: Show single gestures (15 seconds per round)
  - **Medium (⭐⭐)**: Complete simple sentences with gesture sequences (25 seconds per round)
  - **Hard (⭐⭐⭐)**: Complete complex sentences with gesture sequences (40 seconds per round)

- **Game Mechanics**
  - **10 rounds** per game
  - **Time limits** per round (15/25/40 seconds depending on difficulty)
  - **Scoring system**:
    - Base points: 10 (Easy), 20 (Medium), 30 (Hard)
    - Time bonus: +⌊remaining_time/3⌋ points
    - Error penalty: -3 points per incorrect gesture
  - **Challenge tracking**: Each challenge appears maximum 2 times per game, never consecutively
  - **Visual feedback**: Real-time gesture sequence display with correct/incorrect indicators

- **Game Features**
  - Countdown timer before each round
  - Live score tracking (Score, Round, Correct, Time)
  - Timer turns red when ≤10 seconds remain
  - Automatic progression between rounds
  - Final results screen with total score and statistics

- **Medium Level Sentences** (29 total)
  - Simple questions and statements
  - Sentences with numbers (1-5) and roles
  - Variations with random numbers for replayability

- **Hard Level Sentences** (23 total)
  - Complex multi-gesture sequences
  - Conditional statements
  - Combinations of numbers, roles, and actions
  - Multiple variations for increased challenge

### Design

- **Color Palette**
  - Soft Cyan (#90f1ef)
  - Petal Frost (#ffd6e0)
  - Vanilla Custard (#ffef9f)
  - Light Green (#c1fba4, #7bf1a8)
  - Dark text (#1a2424) for readability

- **Visual Elements**
  - Animated gradient background
  - Floating shapes for visual interest
  - Glass-morphism UI components (semi-transparent with backdrop blur)
  - Smooth transitions and animations
  - High contrast for readability

### Technical Implementation

- **Camera Access**: `navigator.mediaDevices.getUserMedia()` API
- **API Communication**: Fetch API with FormData for image uploads
- **Frame Processing**: Canvas API for frame capture and conversion
- **State Management**: Class-based JavaScript architecture
- **Responsive Design**: CSS Grid and Flexbox with mobile breakpoints

### API Integration

- **Endpoint**: `http://localhost:8000/predict`
- **Method**: POST
- **Format**: FormData with image blob
- **Response**: JSON with `gesture` and `confidence` fields

### Browser Compatibility

- Requires modern browser with:
  - Camera access support (`getUserMedia`)
  - ES6+ JavaScript support
  - Canvas API support
  - Fetch API support

### Deployment

- **Docker**: Ready for containerized deployment
- **Nginx**: Configured for static file serving and API proxying
- **Static Assets**: Optimized for production with caching headers

=======
5. Deployment: Successful containerization and API deployment for easy distribution and access
>>>>>>> 0c067371d76007d098bc42c23a534d96f6f62062

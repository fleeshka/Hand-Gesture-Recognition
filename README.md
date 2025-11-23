# Hand Gesture Recognition System

A real-time hand gesture recognition system for interactive applications, featuring a web-based interface with gamification elements. The system uses MediaPipe for hand detection and a machine learning model for gesture classification.

## 🎯 Project Overview

This project provides a complete end-to-end solution for recognizing hand gestures in real-time. It's designed for applications like the *Sports Mafia* game, where players use gestures to communicate, but can be extended to any gesture-based interaction system.

### Key Features

- **Real-time Recognition**: Processes webcam frames at 25+ FPS with <100ms latency
- **Interactive Game Mode**: Three difficulty levels with scoring and challenges
- **Gesture Library**: Comprehensive documentation of all supported gestures
- **Web-based Interface**: Modern, responsive UI with glass-morphism design
- **RESTful API**: FastAPI backend with CORS support for easy integration
- **Docker Support**: Complete containerization for easy deployment
- **High Accuracy**: >90% classification accuracy on validation set

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Frontend  │─────▶│   Backend    │─────▶│    Model    │
│  (Nginx)    │      │  (FastAPI)   │      │ (sklearn)   │
│  Port 3000  │      │  Port 8000   │      │             │
└─────────────┘      └──────────────┘      └─────────────┘
```

## 📋 Table of Contents

- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Frontend Features](#frontend-features)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🛠️ Technologies

### Backend
- **Python 3.10+**
- **FastAPI**: Modern, fast web framework for building APIs
- **MediaPipe**: Hand landmark detection
- **scikit-learn**: Machine learning model (Logistic Regression)
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Uvicorn**: ASGI server

### Frontend
- **HTML5/CSS3/JavaScript (ES6+)**
- **Canvas API**: Frame capture and processing
- **WebRTC**: Camera access
- **Fetch API**: HTTP requests
- **Nginx**: Web server for production

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Reverse proxy and static file serving

### Machine Learning
- **MediaPipe Hands**: 21-point hand landmark detection
- **scikit-learn**: Logistic Regression classifier
- **Jupyter Notebooks**: Data collection and model training

## 📁 Project Structure

```
Hand-Gesture-Recognition-1/
├── code/
│   └── deployment/
│       └── api/
│           ├── main.py              # FastAPI application
│           ├── requirements.txt     # Python dependencies
│           └── Dockerfile          # Backend container config
├── data/
│   ├── processed/                  # Processed training data
│   │   └── data.csv
│   └── raw/                        # Raw gesture samples
├── frontend/
│   ├── css/
│   │   └── style.css              # Application styles
│   ├── js/
│   │   ├── home.js                # Home page logic
│   │   ├── game.js                # Game mode logic
│   │   └── library.js             # Library page logic
│   ├── pages/
│   │   ├── home.html              # Main recognition page
│   │   ├── game.html              # Game mode page
│   │   └── library.html           # Gesture library
│   ├── index.html                 # Entry point
│   ├── Dockerfile                 # Frontend container config
│   ├── nginx.conf                 # Nginx configuration
│   └── package.json               # Node.js dependencies
├── models/
│   ├── gesture_lr_nums.pkl        # Trained model (numbers)
│   ├── gesture_lr.pkl             # Trained model (all gestures)
│   └── README.md                  # Model documentation
├── notebooks/
│   ├── dataset_collection.ipynb  # Data collection pipeline
│   ├── prepare_data_training.ipynb # Data preprocessing
│   ├── model_training.ipynb       # Model training pipeline
│   └── README.md                  # Notebooks documentation
├── scripts/
│   ├── hand_tracking.py           # Hand tracking utilities
│   ├── model_testing.py           # Model testing script
│   └── README.md                  # Scripts documentation
├── docker-compose.yml             # Docker Compose configuration
└── README.md                      # This file
```

## 📦 Prerequisites

### For Local Development
- **Python 3.10+**
- **Node.js 18+** (for frontend build)
- **npm** or **yarn**
- **Webcam** (for testing)
- **Modern web browser** with camera support

### For Docker Deployment
- **Docker 20.10+**
- **Docker Compose 2.0+**

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/fleeshka/Hand-Gesture-Recognition.git
   cd Hand-Gesture-Recognition
   ```

2. **Start the services**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📖 Detailed Setup

### Backend Setup

1. **Navigate to the API directory**
   ```bash
   cd code/deployment/api
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files exist**
   - Place trained model files in `models/` directory
   - Default model: `models/gesture_lr_nums.pkl`

5. **Run the API server**
   ```bash
   python main.py
   # Or using uvicorn directly:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Development mode** (with Vite)
   ```bash
   npm run dev
   ```

4. **Production build**
   ```bash
   npm run build
   ```

### Model Training

1. **Data Collection**
   - Open `notebooks/dataset_collection.ipynb`
   - Follow the notebook to collect gesture samples
   - Data is saved to `data/raw/` directory

2. **Data Preparation**
   - Open `notebooks/prepare_data_training.ipynb`
   - Aggregate and preprocess collected data
   - Output: `data/processed/data.csv`

3. **Model Training**
   - Open `notebooks/model_training.ipynb`
   - Train the Logistic Regression model
   - Model is saved to `models/gesture_lr_nums.pkl`

See `notebooks/README.md` for detailed instructions.

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gesture_classes": ["0", "1", "2", "3", "4", "5", "Civilian", "Cool", "Don", "If", "Mafia", "Question", "Sheriff", "You", "Me"]
}
```

#### Predict Gesture
```http
POST /predict
Content-Type: multipart/form-data
```

**Request:**
- Form data with `file` field containing an image (JPEG, PNG)

**Response:**
```json
{
  "gesture": "3",
  "confidence": 0.95,
  "all_predictions": [
    {"gesture": "3", "confidence": 0.95},
    {"gesture": "2", "confidence": 0.03},
    {"gesture": "4", "confidence": 0.02}
  ],
  "latency_ms": 45.2
}
```

#### List Gesture Classes
```http
GET /gestures
```

**Response:**
```json
{
   "gestures": ['civilian', 'cool', 'don', 'five', 'four', 'if', 'mafia', 'me', 'one', 'question', 'sheriff', 'three1', 'three2', 'two', 'you', 'zero'],
   "count": 16
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤖 Model Information

### Architecture

The system uses a **Logistic Regression** classifier from scikit-learn with the following configuration:

- **Algorithm**: Multinomial Logistic Regression
- **Solver**: SAGA (Stochastic Average Gradient)
- **Regularization**: L2 penalty with C=10
- **Features**: 64-dimensional vector
  - 63 coordinates (21 landmarks × 3 coordinates: x, y, z)
  - 1 handedness indicator (left=0, right=1)

### Feature Extraction

1. **Hand Detection**: MediaPipe Hands detects 21 hand landmarks
2. **Normalization**: 
   - Landmarks normalized relative to wrist position
   - Scaled by maximum distance from wrist
3. **Feature Vector**: Flattened to 64 dimensions

### Supported Gestures

#### Numbers
- **0** (Zero): Round shape with visible opening
- **1** (One): Single extended finger
- **2** (Two): Two extended fingers
- **3** (Three): Three extended fingers (variations: Three1, Three2)
- **4** (Four): Four extended fingers
- **5** (Five): All fingers extended

#### Roles (Sports Mafia)
- **Civilian** 👍: Thumbs up
- **Mafia** 👎: Thumbs down
- **Sheriff** 👌: OK sign
- **Don** 🎩: Open palm with thumb touching ring finger

#### Communication
- **If** 🤙: Shaka/hang loose gesture
- **Question** ❓: Question mark shape
- **Cool** 🤘: Rock on gesture
- **You** 🫵: Pointing at camera
- **Me** 👉: Pointing at self

### Model Performance

- **Accuracy**: >90% on validation set
- **Latency**: <100ms per prediction
- **FPS**: 25+ frames per second
- **Robustness**: Works across different lighting, backgrounds, and skin tones

### Model Files

- `models/gesture_lr_nums.pkl`: Model trained on all gesture and numbers (0-5)  
- `models/gesture_lr.pkl`: Model trained on all gestures

Both files contain:
- Trained Logistic Regression model
- Label encoder for gesture classes

## 🎨 Frontend Features

### Home Page
- **Real-time Recognition**: Live camera feed with gesture detection
- **Statistics**: Gesture count and average confidence
- **Visual Feedback**: Emoji-enhanced gesture names
- **Tips Modal**: Collapsible tips for best recognition

### Game Mode
- **Three Difficulty Levels**:
  - **Easy (⭐)**: Single gestures, 15 seconds per round
  - **Medium (⭐⭐)**: Simple sentences, 25 seconds per round
  - **Hard (⭐⭐⭐)**: Complex sentences, 40 seconds per round
- **Scoring System**: Base points + time bonus - error penalty
- **10 Rounds**: Each challenge appears max 2 times
- **Live Stats**: Score, round, correct count, time remaining

### Library Page
- **Gesture Documentation**: All gestures with descriptions
- **How-to Guides**: Instructions for each gesture
- **Categories**: Roles, Communication, Numbers

### Design
- **Glass-morphism UI**: Semi-transparent components with backdrop blur
- **Animated Background**: Gradient animations
- **Responsive Design**: Works on desktop and mobile
- **Color Palette**: Soft cyan, petal frost, vanilla custard, light green

## 🔧 Development

### Environment Variables

#### Backend
- `MODEL_DIR`: Directory containing model files (default: `/app/models`)
- `MODEL_FILE`: Specific model file path (default: `/app/models/gesture_lr_nums.pkl`)

#### Frontend
- `NODE_ENV`: Environment mode (`development` or `production`)
- `API_URL`: Backend API URL (default: `http://localhost:8000`)

### Code Structure

#### Backend (`code/deployment/api/main.py`)
- FastAPI application with CORS middleware
- Model loading and prediction logic
- Gesture name normalization
- Error handling and logging

#### Frontend JavaScript
- **Class-based Architecture**: 
  - `GestureRecognizer` (home.js): Handles recognition on home page
  - `GestureGame` (game.js): Manages game logic and scoring
- **API Communication**: Fetch API with FormData
- **Canvas Processing**: Frame capture and mirroring

### Adding New Gestures

1. **Collect Data**: Use `notebooks/dataset_collection.ipynb`
2. **Retrain Model**: Use `notebooks/model_training.ipynb`
3. **Update Frontend**: Add gesture mapping in `home.js` and `game.js`
4. **Update Library**: Add gesture to `library.html`

## 🧪 Testing

### Model Testing

Test the trained model directly:

```bash
python scripts/model_testing.py
```

This script:
- Loads the model from `models/gesture_lr_nums.pkl`
- Opens webcam feed
- Displays predictions, confidence, FPS, and latency
- Press 'q' to exit

### API Testing

1. **Using curl**:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -F "file=@path/to/image.jpg"
   ```

2. **Using Python**:
   ```python
   import requests
   
   with open('test_image.jpg', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/predict',
           files={'file': f}
       )
   print(response.json())
   ```

3. **Using Swagger UI**: http://localhost:8000/docs

### Frontend Testing

- Open browser developer tools (F12)
- Check console for errors
- Test camera permissions
- Verify API communication

## 🚢 Deployment

### Docker Deployment

#### Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Build frontend container (Nginx)
- Build backend container (FastAPI)
- Create network for communication
- Expose ports 3000 (frontend) and 8000 (backend)

#### Individual Container Build

**Backend:**
```bash
cd code/deployment/api
docker build -t gesture-api .
docker run -p 8000:8000 -v $(pwd)/../../models:/app/models gesture-api
```

**Frontend:**
```bash
cd frontend
docker build -t gesture-frontend .
docker run -p 3000:80 gesture-frontend
```

### Production Considerations

1. **Environment Variables**: Set production values
2. **HTTPS**: Configure SSL certificates
3. **CORS**: Restrict allowed origins
4. **Resource Limits**: Set Docker memory/CPU limits
5. **Logging**: Configure centralized logging
6. **Monitoring**: Add health check endpoints
7. **Backup**: Regular model file backups

## 📊 Performance Metrics

- **Accuracy**: >90% on validation set
- **Latency**: ~<100ms per prediction
- **Throughput**: 30+ FPS
- **Model Size**: ~16KB (pickle file)
- **Memory Usage**: ~5GB (backend container)

## 📝 License

This project is open source. 

## 🙏 Acknowledgments

- **MediaPipe**: For robust hand landmark detection
- **FastAPI**: For the excellent web framework
- **scikit-learn**: For machine learning tools
- **OpenCV**: For computer vision capabilities

## 🔮 Future Enhancements

- [ ] Support for dynamic gestures (LSTM/Transformer models)
- [ ] Multi-hand detection
- [ ] Gesture sequence recognition
- [ ] Mobile app version
- [ ] Custom gesture training interface
- [ ] Real-time multiplayer game mode

---

**Made with ❤️ for interactive gesture-based applications**

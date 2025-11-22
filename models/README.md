# Models Directory

This directory is used to store trained machine learning models for hand gesture recognition. The models are created through the training process described in `notebooks/model_training.ipynb`.

## Files

- `gesture_lr.pkl`: A pickle file containing the trained Logistic Regression model and its associated label encoder. This model recognizes hand gestures from normalized MediaPipe hand landmarks.

- `model_testing.py`: A Python script for testing the trained model in real-time using your webcam. It uses OpenCV and MediaPipe to detect hand landmarks, processes them, and predicts gestures with the loaded model.

## Training Models

To train a new model or understand the training process:

1. Open `notebooks/model_training.ipynb` in Jupyter Notebook.
2. Follow the step-by-step guide, which includes:
   - Loading and preprocessing the dataset from `data/processed/data.csv`.
   - Feature selection and label encoding.
   - Training a Logistic Regression model with optimized hyperparameters.
   - Evaluating the model performance.
   - Saving the model to this directory.

The notebook demonstrates the complete pipeline from data preparation to model saving.

## Testing Models

To test a trained model:

1. Ensure you have a model file (e.g., `gesture_lr.pkl`) in this directory.
2. Open `models/model_testing.py` and check the `MODEL_FILEPATH` variable at the top of the file. Update it to match the name of your model file if necessary.
   - Example: `MODEL_FILEPATH="models/gesture_lr.pkl"`
3. Run the script: `python models/model_testing.py`
4. The script will open your webcam and display real-time gesture recognition with:
   - Detected gesture text.
   - Hand detection confidence.
   - FPS counter.
5. Press 'q' to quit the application.

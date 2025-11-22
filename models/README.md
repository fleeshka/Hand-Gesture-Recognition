# Models Directory

This directory is used to store trained machine learning models for hand gesture recognition. The models are created through the training process described in `notebooks/model_training.ipynb`.

## Files

- `gesture_lr.pkl`: A pickle file containing the trained Logistic Regression model and its associated label encoder. This model recognizes hand gestures from normalized MediaPipe hand landmarks.


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
2. Open `scripts/model_testing.py` and check the `MODEL_FILEPATH` variable at the top of the file. Update it to match the name of your model file if necessary.
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


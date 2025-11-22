# Notebooks

This directory contains Jupyter notebooks and related files for the hand gesture recognition project pipeline. These notebooks handle data collection, preprocessing, model training, and evaluation.

## Pipeline Flow
1. **dataset_collection.ipynb** → Collects raw gesture data
2. **prepare_data_training.ipynb** → Aggregates and analyzes data
3. **model_training.ipynb** → Trains model (optionally uses gridsearch results for hyperparameter selection)
4. **gridsearch_cv_results.csv** → Supports hyperparameter optimization during training

## Overview

### dataset_collection.ipynb

Capture hand gesture samples in real-time using a webcam and MediaPipe Hands for landmark detection.

**Key Features:**
- Real-time hand detection with MediaPipe Hands
- Normalization of landmarks relative to the wrist for consistency
- Scaling based on hand size
- Labeling with gesture name and left/right hand indicator
- Mirrored preview for user comfort
- Configurable parameters: number of samples, capture interval, camera resolution
- Saves data to CSV files, allowing incremental dataset building

**Usage:** Collects samples for specific gestures (e.g., thumbs_up, question, if, civilian, cool, don, mafia, sheriff, you, me) by running the capture function with specified parameters. Each sample includes 21 hand landmarks (63 coordinates) plus gesture label and handedness flag.

### prepare_data_training.ipynb

Aggregate individual gesture CSV files into a unified dataset and perform exploratory data analysis (EDA) to ensure data quality and understand distributions.

**Key Steps:**
1. Search and load all CSV files from `../data/raw`
2. Concatenate into a single DataFrame with reset indexes
3. Integrity checks: verify columns, missing values, expected structure
4. Summarize dataset: counts per gesture, left/right hand distribution with pivot table visualization
5. Analyze landmark distributions: descriptive statistics, boxplots per gesture for selected landmarks
6. Visualize mean hand shapes: Compute and plot average 3D hand skeletons per gesture using MediaPipe connections

**Purpose:** Creates the processed dataset (`../data/processed/data.csv`) for training, validates data integrity, and provides insights into gesture variability and class balance to inform model development.

### model_training.ipynb

Train, evaluate, and deploy a logistic regression model for hand gesture classification, culminating in real-time gesture recognition.

**Key Steps:**
1. Load and preprocess the dataset (hand landmarks + labels)
2. Feature selection (63 landmark coordinates + handedness)
3. Label encoding for gesture classes
4. Train/test split with stratification
5. Train logistic regression with optimized hyperparameters (`multinomial`, `saga` solver, `C=10`, `L2` penalty)
6. Evaluate model performance with classification report
7. Save trained model and label encoder to pickle file
8. Implement real-time inference using webcam and MediaPipe for live gesture recognition

**Outcome:** Produces a trained model capable of classifying gestures from live video feed with FPS tracking and confidence display.


    NOTE: All notebooks use MediaPipe for hand landmark detection and follow consistent normalization and feature extraction methods.
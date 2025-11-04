import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import json
import os
import cv2
import mediapipe as mp

class GestureModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def create_advanced_model(self, input_shape):
        """Create advanced neural network model for gesture recognition"""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            # First block with regularization
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Second block - deeper representation
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Third block - feature compression
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Fourth block - final features
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer for classification
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with appropriate optimizer and metrics
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def load_and_preprocess_data(self, csv_files):
        """Load and preprocess data from CSV files including hand information"""
        all_data = []
        all_labels = []
        
        for file_path in csv_files:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Extract landmark coordinates (x0,y0,x1,y1,...,x20,y20)
            feature_columns = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
            landmarks = df[feature_columns].values
            
            # Extract hand information and add as additional feature
            is_right_hand = df['is_right_hand'].values.reshape(-1, 1)
            
            # Combine landmarks with hand information
            features_with_hand = np.hstack([landmarks, is_right_hand])
            
            # Extract gesture labels
            labels = df['gesture'].values
            
            all_data.append(features_with_hand)
            all_labels.append(labels)
        
        # Combine all datasets
        X = np.vstack(all_data)
        y = np.hstack(all_labels)
        
        return X, y
    
    def prepare_data(self, X, y):
        """Prepare data for training: encoding labels and feature scaling"""
        # Encode string labels to integers
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded, self.num_classes)
        
        # Normalize features for better training stability
        X_normalized = self.scaler.fit_transform(X)
        
        return X_normalized, y_categorical
    
    def train(self, X, y, validation_split=0.2, epochs=200, batch_size=32):
        """Train the gesture recognition model"""
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y_processed, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_processed  # Maintain class distribution
        )
        
        # Define training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                mode='max'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_gesture_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        return history
    
    def evaluate(self, X, y):
        """Evaluate model performance on given data"""
        X_processed, y_processed = self.prepare_data(X, y)
        return self.model.evaluate(X_processed, y_processed, verbose=0)
    
    def predict(self, X):
        """Make predictions on new data"""
        X_processed = self.scaler.transform(X)
        predictions = self.model.predict(X_processed, verbose=0)
        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    def predict_proba(self, X):
        """Get prediction probabilities for all classes"""
        X_processed = self.scaler.transform(X)
        return self.model.predict(X_processed, verbose=0)
    
    def save_model(self, model_path='gesture_model.keras', metadata_path='model_metadata.json'):
        """Save trained model and metadata"""
        self.model.save(model_path)
        
        # Save model metadata
        metadata = {
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'num_classes': self.num_classes,
            'input_shape': self.model.input_shape[1:],
            'uses_hand_info': True,
            'feature_description': '42 landmarks (x0,y0,...,x20,y20) + is_right_hand'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Save scaler parameters for future use
        np.save('scaler_mean.npy', self.scaler.mean_)
        np.save('scaler_scale.npy', self.scaler.scale_)
    
    def load_model(self, model_path='gesture_model.h5', metadata_path='model_metadata.json'):
        """Load trained model and metadata"""
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.label_encoder.classes_ = np.array(metadata['label_encoder_classes'])
        self.num_classes = metadata['num_classes']
        
        # Load scaler parameters
        self.scaler.mean_ = np.load('scaler_mean.npy')
        self.scaler.scale_ = np.load('scaler_scale.npy')

class RealTimeGesturePredictor:
    def __init__(self, model_path='gesture_model.keras', metadata_path='model_metadata.json'):
        """Initialize real-time gesture predictor with trained model"""
        self.gesture_model = GestureModel(num_classes=3)
        self.gesture_model.load_model(model_path, metadata_path)
        
        # Initialize MediaPipe Hands for landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_frame(self, image):
        """Extract hand landmarks and hand information from video frame"""
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None, None
        
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        
        # Determine if right or left hand (1.0 for right, 0.0 for left)
        is_right_hand = 1.0 if handedness.classification[0].label == 'Right' else 0.0
        
        # Extract all landmark coordinates (x, y)
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y])
        
        # Combine landmarks with hand information as additional feature
        features_with_hand = np.hstack([landmarks, [is_right_hand]])
        
        return features_with_hand.reshape(1, -1), is_right_hand
    
    def predict_gesture(self, image):
        """Predict gesture from video frame"""
        features, hand_info = self.extract_landmarks_from_frame(image)
        if features is None:
            return None, 0.0, None
        
        # Get prediction and confidence
        prediction = self.gesture_model.predict(features)[0]
        probabilities = self.gesture_model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence, hand_info


def plot_training_history(history):
    """Plot training history metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Precision plot
    ax3.plot(history.history['precision'], label='Training Precision')
    ax3.plot(history.history['val_precision'], label='Validation Precision')
    ax3.set_title('Model Precision')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True)
    
    # Recall plot
    ax4.plot(history.history['recall'], label='Training Recall')
    ax4.plot(history.history['val_recall'], label='Validation Recall')
    ax4.set_title('Model Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    csv_files = ['mafia.csv', 'if.csv', 'sheriff.csv']
    
    # Initialize and train model
    gesture_model = GestureModel(num_classes=3)
    
    print("\nLoading data...")
    X, y = gesture_model.load_and_preprocess_data(csv_files)
    
    print(f"Loaded {len(X)} samples")
    print(f"Feature dimension: {X.shape}")  # Should be (n_samples, 43)
    print(f"Classes: {np.unique(y)}")
    
    # Analyze hand feature distribution
    hand_feature = X[:, -1]  # Last feature is is_right_hand
    print(f"\nHand feature distribution:")
    print(f"  Right hand samples: {np.sum(hand_feature == 1)} ({np.mean(hand_feature == 1)*100:.1f}%)")
    print(f"  Left hand samples: {np.sum(hand_feature == 0)} ({np.mean(hand_feature == 0)*100:.1f}%)")
    
    # Create model architecture
    print("\nCreating model...")
    model = gesture_model.create_advanced_model(input_shape=(43,))  # 42 landmarks + 1 hand info
    
    model.summary()
    
    # Train the model
    print("Starting training...")
    history = gesture_model.train(X, y, epochs=10, batch_size=32)
    
    # Evaluate model performance
    print("Evaluating model...")
    loss, accuracy, precision, recall = gesture_model.evaluate(X, y)
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Precision: {precision:.4f}")
    print(f"Final Recall: {recall:.4f}")
    
    # Save trained model
    print("Saving model...")
    gesture_model.save_model()
    
    # Plot training history
    plot_training_history(history)
    
    # Analyze accuracy by hand type
    print("\nAnalyzing accuracy by hand type:")
    predictions = gesture_model.predict(X)
    hand_info = X[:, -1]  # is_right_hand feature
    
    # Accuracy for right hand
    right_mask = hand_info == 1
    right_accuracy = np.mean(predictions[right_mask] == y[right_mask])
    print(f"Accuracy for right hand: {right_accuracy:.4f} ({np.sum(right_mask)} samples)")
    
    # Accuracy for left hand
    left_mask = hand_info == 0
    left_accuracy = np.mean(predictions[left_mask] == y[left_mask])
    print(f"Accuracy for left hand: {left_accuracy:.4f} ({np.sum(left_mask)} samples)")
    
    # Test on random samples
    print("\nTesting on random samples:")
    sample_indices = np.random.randint(0, len(X), 10)
    correct = 0
    for idx in sample_indices:
        sample = X[idx:idx+1]
        true_label = y[idx]
        pred_label = gesture_model.predict(sample)[0]
        is_correct = true_label == pred_label
        correct += is_correct
        hand_type = "right" if hand_info[idx] == 1 else "left"
        print(f"True: {true_label:10} Predicted: {pred_label:10} Hand: {hand_type:5} {'✓' if is_correct else '✗'}")
    
    print(f"Sample accuracy: {correct/10:.2f}")

def real_time_demo():
    """Real-time gesture recognition demo"""
    predictor = RealTimeGesturePredictor()
    
    cap = cv2.VideoCapture(0)
    
    # Statistics for performance analysis
    gesture_counts = {}
    
    print("Starting real-time gesture recognition...")
    print("Press ESC to exit, 's' to show statistics")
    
    while True:
        success, image = cap.read()
        if not success:
            continue
        
        gesture, confidence, hand_info = predictor.predict_gesture(image)
        
        if gesture:
            hand_text = "Right" if hand_info == 1.0 else "Left"
            
            # Update statistics
            key = f"{gesture}_{hand_text}"
            if key not in gesture_counts:
                gesture_counts[key] = 0
            gesture_counts[key] += 1
            
            # Choose color based on confidence level
            if confidence > 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence
            
            # Display prediction information
            cv2.putText(image, f"Gesture: {gesture}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(image, f"Confidence: {confidence:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image, f"Hand: {hand_text}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show warning for low confidence predictions
            if confidence < 0.6:
                cv2.putText(image, "LOW CONFIDENCE", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Gesture Recognition (Press ESC to exit)', image)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('s'):  # 's' key to show statistics
            print("\nCurrent recognition statistics:")
            for gesture_hand, count in gesture_counts.items():
                print(f"  {gesture_hand}: {count} frames")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\nFinal recognition statistics:")
    for gesture_hand, count in sorted(gesture_counts.items()):
        print(f"  {gesture_hand}: {count} frames")

if __name__ == "__main__":
    main()
    
    # Launch real-time demo after training
    print("\nLaunching real-time gesture recognition...")
    real_time_demo()
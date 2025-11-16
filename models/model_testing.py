import cv2
import torch
from torch import nn
import mediapipe as mp
import numpy as np
import pickle
import os

# --------------------------
# 1. Model definition
# --------------------------
class GestureMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)


# --------------------------
# 2. Load artifacts
# --------------------------
ARTIFACT_DIR = "models"

with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

input_dim = 21 * 3 + 1  # 63 coords + is_right_hand = 64
num_classes = len(le.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GestureMLP(input_dim, num_classes)
model.load_state_dict(torch.load(os.path.join(ARTIFACT_DIR, "model_small.pth"), map_location=device))
model.to(device)
model.eval()

print("Model loaded. Classes:", le.classes_)


# --------------------------
# 3. Mediapipe
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)


# --------------------------
# 4. Camera
# --------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Starting camera...")


# --------------------------
# 5. Frame loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    prediction_text = "No hand detected"

    if result.multi_hand_landmarks and result.multi_handedness:
        lm = result.multi_hand_landmarks[0]
        handed_label = result.multi_handedness[0].classification[0].label
        is_right = 1 if handed_label == "Right" else 0

        # --------------------------
        # Wrist for normalization
        # --------------------------
        wrist = lm.landmark[0]
        wx, wy, wz = wrist.x, wrist.y, wrist.z

        # --------------------------
        # Scale (max distance from wrist)
        # --------------------------
        distances = [np.linalg.norm([p.x - wx, p.y - wy, p.z - wz]) for p in lm.landmark]
        scale = max(distances) if max(distances) > 1e-6 else 1e-6

        # --------------------------
        # Feature vector in exact training format
        # x0,y0,z0,x1,y1,z1,...x20,y20,z20,is_right_hand
        # --------------------------
        coords = []
        for i, p in enumerate(lm.landmark):
            coords.extend([
                (p.x - wx) / scale,
                (p.y - wy) / scale,
                (p.z - wz) / scale
            ])

        # Append hand flag at the end
        coords.append(is_right)

        coords = np.array(coords, dtype=np.float32).reshape(1, -1)
        coords_scaled = scaler.transform(coords)  # same scaler as training

        X_tensor = torch.tensor(coords_scaled, dtype=torch.float32).to(device)

        # --------------------------
        # Model prediction
        # --------------------------
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            class_id = torch.argmax(probs).item()
            confidence = probs[class_id].item()

        gesture_name = le.inverse_transform([class_id])[0]

        prediction_text = f"{handed_label} | {gesture_name} ({confidence:.2f})"

        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # --------------------------
    # Display
    # --------------------------
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

# PUT NAME OF YOUR MODEL FILE 
MODEL_FILEPATH="models/gesture_lr.pkl"

with open(MODEL_FILEPATH, "rb") as f:
    obj = pickle.load(f)

clf = obj["model"]
le = obj["label_encoder"]

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

fps_history = deque(maxlen=10)
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    fps = 1.0 / (now - last_time)
    last_time = now
    fps_history.append(fps)
    smooth_fps = sum(fps_history) / len(fps_history)

    frame_mirrored = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    gesture_text = "No hand"
    conf_text = ""

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_landmarks = result.multi_hand_landmarks[0]
        handedness_obj = result.multi_handedness[0].classification[0]
        handedness = handedness_obj.label
        confidence = handedness_obj.score 
        conf_text = f"{confidence:.2f}"

        is_right = 1 if handedness == "Right" else 0

        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        wrist_z = hand_landmarks.landmark[0].z


        distances = [
            np.linalg.norm([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])

            for lm in hand_landmarks.landmark
        ]

        scale = max(distances)
        coords = []

        for lm in hand_landmarks.landmark:
            x = (lm.x - wrist_x) / scale
            y = (lm.y - wrist_y) / scale
            z = (lm.z - wrist_z) / scale
            coords.extend([x, y, z])



        input_vec = np.array(coords + [is_right]).reshape(1, -1)
        pred_class = clf.predict(input_vec)[0]
        gesture_text = le.inverse_transform([pred_class])[0]



    cv2.putText(frame_mirrored, f"Gesture: {gesture_text}",

                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 0), 3)

    cv2.putText(frame_mirrored, f"Conf: {conf_text}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 3)

    cv2.putText(frame_mirrored, f"FPS: {smooth_fps:.1f}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 3)


    cv2.imshow("Gesture Recognition", frame_mirrored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
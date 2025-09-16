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

### Success Criteria

1. Accuracy: ≥80% classification accuracy on validation set 
2. Robustness: works across different people, skin tones, lighting, and backgrounds 
3. Performance
   * Prediction delay ≤ 200 ms per frame
   * Target ≥ 25 FPS on a standard laptop webcam
4. Usability & Stability: recognizes gestures consistently without false positives
5. 

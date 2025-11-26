import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# -------------------------
# Load TFLITE Model
# -------------------------
interpreter = tf.lite.Interpreter(model_path="isl_gesture_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = input_details[0]['shape'][1]  # assuming model expects square images


# -------------------------
# Hand Detection (simple + stable)
# -------------------------
def detect_hand(frame):
    """Detects hand via skin mask + contour. Returns bounding box crop."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (works for most lighting)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)

        if w * h > 5000:  # ignore noise
            return frame[y:y+h, x:x+w]

    # fallback (center crop)
    h, w, _ = frame.shape
    size = min(h, w)
    startx = w//2 - size//2
    starty = h//2 - size//2
    return frame[starty:starty+size, startx:startx+size]


# -------------------------
# Run Model
# -------------------------
def predict(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    idx = int(np.argmax(output))
    confidence = float(np.max(output))

    return idx, confidence


# -------------------------
# UI
# -------------------------
st.title("ISL Gesture Detection - Real Time")
st.write("Uses your isl_gesture_model.tflite + OpenCV hand detection")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Camera error")
        break

    frame = cv2.flip(frame, 1)

    hand_roi = detect_hand(frame)

    gesture_id, conf = predict(hand_roi)

    # Display bounding box text
    cv2.putText(frame, f"Gesture: {gesture_id}  Conf: {conf:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    FRAME_WINDOW.image(frame, channels="BGR")

if cap:
    cap.release()

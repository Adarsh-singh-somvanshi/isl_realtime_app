import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import av
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import json

# -------------------------
# Load TFLite Model
# -------------------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="isl_gesture_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_size = input_details[0]["shape"][1]
    return interpreter, input_details, output_details, img_size

interpreter, input_details, output_details, IMG_SIZE = load_model()

# -------------------------
# Load labels (JSON)
# -------------------------
@st.cache_resource
def load_labels():
    with open("label_map.json", "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    return idx_to_label

idx_to_label = load_labels()

# -------------------------
# Safe crop for model
# -------------------------
def detect_hand(frame):
    h, w, _ = frame.shape
    size = min(h, w)
    cx = w // 2 - size // 2
    cy = h // 2 - size // 2
    return frame[cy:cy+size, cx:cx+size]

# -------------------------
# Prediction
# -------------------------
def predict(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = int(np.argmax(output))
    conf = float(np.max(output))
    label = idx_to_label.get(idx, "Unknown")
    return label, conf

# -------------------------
# Video Processing Class
# -------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        crop = detect_hand(img)
        label, conf = predict(crop)

        cv2.putText(
            img, f"{label} ({conf:.2f})", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# UI
# -------------------------
st.title("🖐️ ISL Gesture Detection – Real Time (TFLite)")

webrtc_streamer(
    key="isl-app",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

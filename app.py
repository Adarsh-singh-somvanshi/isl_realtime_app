import os
# Set env variables before importing tensorflow/tflite
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
# Helper: Check File Integrity
# -------------------------
def check_model_file(model_path):
    """
    Checks if the model file exists and is not a Git LFS pointer.
    """
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file '{model_path}' not found.")
        st.stop()
    
    file_size = os.path.getsize(model_path)
    # Git LFS pointers are usually around 130 bytes. Real models are MBs.
    if file_size < 2000: 
        st.error(f"❌ Error: Model file '{model_path}' is too small ({file_size} bytes).")
        st.warning("""
        **Diagnosis:** You are likely using Git LFS (Large File Storage). 
        Streamlit Cloud has downloaded the 'LFS Pointer' (a text file) instead of the actual model.
        
        **Solution:**
        1. If your model is < 100MB: Remove it from Git LFS and push it as a regular file.
        2. If your model is > 100MB: You must host it externally (Google Drive/S3) and download it in the code.
        """)
        st.stop()
    
    return True

# -------------------------
# Load TFLite Model
# -------------------------
@st.cache_resource
def load_model():
    model_path = "isl_gesture_model.tflite"
    
    # Run the check before loading
    check_model_file(model_path)
    
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img_size = input_details[0]["shape"][1]
        return interpreter, input_details, output_details, img_size
    except Exception as e:
        st.error(f"Failed to allocate tensors. Error: {e}")
        st.stop()

interpreter, input_details, output_details, IMG_SIZE = load_model()

# -------------------------
# Load labels (JSON)
# -------------------------
@st.cache_resource
def load_labels():
    if not os.path.exists("label_map.json"):
        st.error("label_map.json not found!")
        st.stop()
        
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

import os
# Configure environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import json
import pickle
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -------------------------
# FIXED: Streamlit Cloud SAFE WebRTC servers
# -------------------------
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {
                "urls": ["turn:relay1.expressturn.com:3478"],
                "username": "expressturn",
                "credential": "expressturn"
            }
        ]
    }
)

# -------------------------
# Helper: Check Model Integrity
# -------------------------
def check_model_file(model_path):
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found.")
        st.stop()
    file_size = os.path.getsize(model_path)
    if file_size < 2000:
        st.error(f"Error: Model file '{model_path}' is INVALID ({file_size} bytes).")
        st.warning("""
        **Diagnosis:** Git LFS pointer issue
        **Fix:**
        1. Delete file
        2. Run: git lfs untrack isl_gesture_model.tflite
        3. Add the real .tflite file again
        4. Push
        """)
        st.stop()

# -------------------------
# Load Model (Lazy Import)
# -------------------------
@st.cache_resource
def load_model():
    import tensorflow as tf
    
    model_path = "isl_gesture_model.tflite"
    check_model_file(model_path)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_size = input_details[0]["shape"][1]
    return interpreter, input_details, output_details, img_size

interpreter, input_details, output_details, IMG_SIZE = load_model()

# -------------------------
# Load Labels - FIXED to handle both .pkl and .json
# -------------------------
@st.cache_resource
def load_labels():
    # Try JSON first
    if os.path.exists("label_map.json"):
        try:
            with open("label_map.json", "r") as f:
                label_to_idx = json.load(f)
            return {int(v): k for k, v in label_to_idx.items()}
        except Exception as e:
            st.warning(f"Error loading label_map.json: {e}")
    
    # Try PKL next
    if os.path.exists("label_map.pkl"):
        try:
            with open("label_map.pkl", "rb") as f:
                idx_to_label = pickle.load(f)
            return idx_to_label if isinstance(idx_to_label, dict) else {int(k): str(v) for k, v in enumerate(idx_to_label)}
        except Exception as e:
            st.warning(f"Error loading label_map.pkl: {e}")
    
    st.warning("No label map found. Using dummy labels.")
    return {}

idx_to_label = load_labels()

# -------------------------
# Crop center square
# -------------------------
def detect_hand(frame):
    h, w, _ = frame.shape
    size = min(h, w)
    cx = w // 2 - size // 2
    cy = h // 2 - size // 2
    return frame[cy:cy + size, cx:cx + size]

# -------------------------
# Predict
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
# Video Processor
# -------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        crop = detect_hand(img)
        try:
            label, conf = predict(crop)
            cv2.rectangle(img, (10, 10), (320, 65), (0, 0, 0), -1)
            cv2.putText(img, f"{label} ({conf:.2f})", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="ISL Gesture Detection", layout="wide")
st.title("Hand Gesture Detection - Real Time")
st.write("Indian Sign Language gesture recognition powered by TensorFlow Lite & Streamlit Cloud")

col1, col2 = st.columns([3, 1])
with col2:
    st.info("App Status: Ready", icon="info")

webrtc_streamer(
    key="isl-gesture",
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("""
---
**How to use:**
1. Allow camera access when prompted
2. Show hand gestures to the camera
3. The model will detect and display the ISL gesture in real-time
4. Confidence score is shown above each prediction
""")

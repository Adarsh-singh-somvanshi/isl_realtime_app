import os
# Configure environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import json
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
        st.error(f"❌ Error: Model file '{model_path}' not found.")
        st.stop()

    file_size = os.path.getsize(model_path)
    if file_size < 2000:
        st.error(f"❌ Error: Model file '{model_path}' is INVALID ({file_size} bytes).")
        st.warning("""
        **Diagnosis:** Git LFS pointer issue  
        **Fix:**  
        1. Delete file  
        2. Run: `git lfs untrack isl_gesture_model.tflite`  
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
# Load Labels
# -------------------------
@st.cache_resource
def load_labels():
    if not os.path.exists("label_map.json"):
        st.warning("⚠️ label_map.json not found. Using dummy labels.")
        return {}
    with open("label_map.json", "r") as f:
        label_to_idx = json.load(f)
    return {int(v): k for k, v in label_to_idx.items()}

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
st.title("🖐️ ISL Gesture Detection – Real Time")
st.write("Using upgraded drivers & TURN servers for stable WebRTC on Streamlit Cloud.")

webrtc_streamer(
    key="isl-gesture",
    rtc_configuration=RTC_CONFIG,   # 🔥 FIXED: Stable WebRTC
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

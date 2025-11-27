import os
# Configure environment variables to disable GPU and reduce logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import av
import cv2
import numpy as np
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# NOTE: We do NOT import tensorflow globally here to save startup memory.
# It is imported inside load_model().

# -------------------------
# Helper: Check File Integrity
# -------------------------
def check_model_file(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file '{model_path}' not found.")
        st.stop()
    
    file_size = os.path.getsize(model_path)
    # Check if it's a Git LFS pointer (usually < 2KB)
    if file_size < 2000: 
        st.error(f"❌ Error: Model file '{model_path}' is too small ({file_size} bytes).")
        st.warning("""
        **Diagnosis:** You are likely using Git LFS, but Streamlit Cloud downloaded the pointer file.
        
        **Solution:**
        1. Untrack the file: `git lfs untrack isl_gesture_model.tflite`
        2. Delete the file, commit, and push.
        3. Add the actual file back as a regular file and push.
        """)
        st.stop()

# -------------------------
# Load Model (Lazy Import Strategy)
# -------------------------
@st.cache_resource
def load_model():
    # Import TensorFlow HERE to avoid "Segmentation Fault" at app startup
    import tensorflow as tf
    
    model_path = "isl_gesture_model.tflite"
    check_model_file(model_path)
    
    try:
        # Using tf.lite.Interpreter allows us to use Flex Ops
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        img_size = input_details[0]["shape"][1]
        return interpreter, input_details, output_details, img_size
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        st.stop()

# Load model (this triggers the function above)
interpreter, input_details, output_details, IMG_SIZE = load_model()

# -------------------------
# Load labels (JSON)
# -------------------------
@st.cache_resource
def load_labels():
    if not os.path.exists("label_map.json"):
        st.warning("Warning: label_map.json not found. Using dummy labels.")
        return {}
        
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
    # Ensure crop doesn't go out of bounds
    cx = max(0, cx)
    cy = max(0, cy)
    return frame[cy:cy+size, cx:cx+size]

# -------------------------
# Prediction Logic
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
        
        try:
            label, conf = predict(crop)
            
            # Draw result
            cv2.rectangle(img, (10, 10), (300, 60), (0, 0, 0), -1) # Background for text
            cv2.putText(
                img, f"{label} ({conf:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        except Exception as e:
            print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# UI
# -------------------------
st.title("🖐️ ISL Gesture Detection – Real Time")
st.write("Ensuring Tensorflow-CPU compatibility...")

webrtc_streamer(
    key="isl-app",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

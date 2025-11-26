import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- 1. CRITICAL MEMORY SETTINGS (Must be at the very top) ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # Force CPU mode (No GPU)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # Reduce logs
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" 

import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# ---- 2. IMPORT TENSORFLOW ----
# We use the full library because your model requires Flex Delegates
import tensorflow as tf
Interpreter = tf.lite.Interpreter

# ---- PAGE SETUP ----
st.set_page_config(page_title="ISL Translator", page_icon="🖐️", layout="wide")
st.title("🖐️ Indian Sign Language – Real-Time Translator")

# ... (The rest of your code remains the same as the optimized version I gave you)
# ---- CUSTOM DRAWING FUNCTION (Replaces mp_drawing to save RAM) ----
def draw_landmarks_fast(image, landmarks, connections, color=(0, 255, 0)):
    if not landmarks:
        return
    h, w, _ = image.shape
    
    # Convert landmarks to pixel coordinates
    points = {}
    for idx, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        # Draw Keypoints (Small circles)
        cv2.circle(image, (cx, cy), 3, color, -1)

    # Draw Connections (Lines)
    if connections:
        for start_idx, end_idx in connections:
            if start_idx in points and end_idx in points:
                cv2.line(image, points[start_idx], points[end_idx], color, 2)

# ---- LOAD RESOURCES ----
@st.cache_resource
def load_model():
    # Load Label Map
    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Load TFLite Model (Single thread to prevent CPU spikes)
    interpreter = Interpreter(model_path="isl_gesture_model.tflite", num_threads=1)
    interpreter.allocate_tensors()
    return idx_to_label, interpreter

try:
    idx_to_label, interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = ""

# ---- VIDEO PROCESSOR ----
class TFLiteProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # NOTE: We removed mp.solutions.drawing_utils to save memory

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0, # Critical for Free Tier performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.sequence = []
        self.last_pred = "Waiting..."
        self.last_conf = 0.0
        self.last_final = None

    def extract_features(self, results):
        vec = []
        # Shoulders
        if results.pose_landmarks:
            for idx in [11, 12]:
                lm = results.pose_landmarks.landmark[idx]
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 6)

        # Hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 63)
        
        # Palm base
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                lm = hand.landmark[0]
                vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 3)
                
        return np.array(vec, dtype=np.float32)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        # Fast Drawing (OpenCV instead of Matplotlib)
        if results.left_hand_landmarks:
            draw_landmarks_fast(img, results.left_hand_landmarks, 
                              self.mp_holistic.HAND_CONNECTIONS, (255, 0, 0)) # Blue for left
        if results.right_hand_landmarks:
            draw_landmarks_fast(img, results.right_hand_landmarks, 
                              self.mp_holistic.HAND_CONNECTIONS, (0, 255, 0)) # Green for right

        # Prediction Logic
        feat = self.extract_features(results)
        self.sequence.append(feat)
        self.sequence = self.sequence[-32:]

        if len(self.sequence) == 32:
            seq = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], seq)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            
            idx = int(np.argmax(preds))
            self.last_pred = idx_to_label.get(idx, "Unknown")
            self.last_conf = float(preds[idx]) * 100

            if self.last_conf > 80 and self.last_pred != "Unknown":
                if self.last_pred != self.last_final:
                    st.session_state.translated_sentence += " " + self.last_pred
                    self.last_final = self.last_pred

        # Draw UI
        cv2.rectangle(img, (0, 0), (800, 40), (0, 0, 0), -1)
        cv2.putText(img, f"{self.last_pred} ({self.last_conf:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- WEBRTC SETUP ----
rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Live Camera Feed")
    webrtc_streamer(
        key="isl-live",
        video_processor_factory=TFLiteProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📝 Translation Output")
    st.info("Text updates when you stop speaking/gesturing.")
    st.write(st.session_state.translated_sentence)
    if st.button("Clear Text"):
        st.session_state.translated_sentence = ""
        st.rerun()

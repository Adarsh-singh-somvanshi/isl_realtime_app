#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISL Real-Time Translation - Streamlit Web App with WebRTC"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# ===========================
# Configuration
# ===========================
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 138
CONFIDENCE_THRESHOLD = 0.5

# Fallback label map (5 ISL gestures)
DEFAULT_LABEL_MAP = {
    0: 'help_you',
    1: 'congratulation',
    2: 'hi_how_are_you',
    3: 'i_am_hungry',
    4: 'take_care_of_yourself'
}

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="ISL Live Translation",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Utility Functions
# ===========================
def load_label_map():
    """Load label map with robust fallback handling"""
    label_map = {}
    label_to_idx = {}
    
    # Try multiple file names and formats
    potential_files = ['label_map_updated.txt', 'label_map.txt']
    
    for filename in potential_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.rsplit(' ', 1)
                            if len(parts) == 2:
                                label, idx_str = parts
                                try:
                                    idx = int(idx_str)
                                    label_map[idx] = label
                                    label_to_idx[label] = idx
                                except ValueError:
                                    pass
                if label_map:
                    return label_map, label_to_idx
            except Exception as e:
                continue
    
    # Use fallback if no files found
    label_to_idx = {v: k for k, v in DEFAULT_LABEL_MAP.items()}
    return DEFAULT_LABEL_MAP, label_to_idx

# ===========================
# Load Resources
# ===========================
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        # Load MediaPipe
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False
        )
        
        # Load label map
        label_map, label_to_idx = load_label_map()
        
        # Load TensorFlow Lite model
        model_path = 'isl_gesture_model.tflite'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return holistic, mp_drawing, mp_holistic, label_map, label_to_idx, interpreter, input_details, output_details, True
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, None, None, None, None, None, None, False

# Load resources
holistic, mp_drawing, mp_holistic, label_map, label_to_idx, interpreter, input_details, output_details, resources_loaded = load_resources()

# ===========================
# Feature Extraction
# ===========================
def extract_features(results, FEATURE_SIZE=138):
    """Extract hand landmarks for gesture recognition"""
    vec = []
    
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 21 * 3)
    
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * 21 * 3)
    
    return np.array(vec, dtype=np.float32) if len(vec) == FEATURE_SIZE else np.zeros(FEATURE_SIZE, dtype=np.float32)

# ===========================
# UI
# ===========================
st.markdown("# 🖐️ ISL Live Real-Time Translation")
st.markdown("### Continuous Hand Gesture Recognition")

if not resources_loaded:
    st.error("❌ Failed to load resources. Check if model files exist.")
    st.stop()

st.success("✅ Resources loaded successfully!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    sequence_length = st.slider("Sequence Length (frames)", 10, 50, 30, step=5)
    
    st.markdown("---")
    st.markdown("### 📋 Supported Gestures")
    if label_map:
        for idx in sorted(label_map.keys()):
            gesture_name = label_map[idx].replace('_', ' ').title()
            st.write(f"**{idx}. {gesture_name}**")

# Main Content
st.subheader("📹 Live Camera Stream")
st.info("🔟 Position your hand in front of the camera. The app will detect and recognize gestures in real-time.")

# WebRTC Configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class GestureDetectionProcessor:
    def __init__(self):
        self.buffer = deque(maxlen=sequence_length)
        self.last_prediction = None
        self.last_confidence = 0.0
        self.frame_count = 0
        self.gesture_count = {}
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w, c = img.shape
            
            # Process with MediaPipe
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Extract features
            features = extract_features(results)
            self.buffer.append(features)
            self.frame_count += 1
            
            # Make prediction if buffer is full
            if len(self.buffer) == sequence_length:
                try:
                    input_data = np.array(list(self.buffer), dtype=np.float32)[np.newaxis, ...]
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                    top_idx = np.argmax(predictions)
                    conf = float(predictions[top_idx])
                    
                    if conf > confidence_threshold:
                        self.last_prediction = label_map.get(top_idx, "Unknown")
                        self.last_confidence = conf
                        
                        # Track gesture count
                        if self.last_prediction not in self.gesture_count:
                            self.gesture_count[self.last_prediction] = 0
                        self.gesture_count[self.last_prediction] += 1
                except Exception as e:
                    pass
            
            # Draw landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            # Display prediction on frame
            if self.last_prediction:
                text = f"{self.last_prediction.title()} ({self.last_confidence:.2f})"
                cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Display frame count
            cv2.putText(img, f"Frames: {self.frame_count}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame

webrtc_ctx = webrtc_streamer(
    key="ISL-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=GestureDetectionProcessor,
    media_stream_constraints={"audio": False, "video": True},
    async_processing=True,
)

# Status
if webrtc_ctx.state.playing:
    st.success("✅ **Camera is LIVE - Point your hand in front of camera**")
    st.info("🔏 The app is continuously analyzing your hand gestures in real-time.")
else:
    st.warning("⚠️ Start the camera to begin gesture detection")

st.markdown("---")
st.markdown("🚀 Real-Time ISL Translation | Powered by Streamlit, MediaPipe & TensorFlow Lite")

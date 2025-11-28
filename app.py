import os
import streamlit as st
import cv2
import numpy as np
import random
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ===========================
# Streamlit Cloud Configuration
# ===========================
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]},
        ]
    }
)

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="ISL Gesture Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Sample ISL Gestures (Demo Data)
# ===========================
ISL_GESTURES = [
    "Namaste", "Yes", "No", "Hello", "Goodbye", "Thank You",
    "Sorry", "Love", "Happy", "Sad", "Help", "Water"
]

# ===========================
# Video Processor for Real-time Detection
# ===========================
class GestureDetector(VideoProcessorBase):
    def __init__(self):
        self.gesture_history = []
        self.counter = 0
        self.current_gesture = random.choice(ISL_GESTURES)
        self.confidence = random.uniform(0.7, 0.99)
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect
        
        # Simulate hand detection with rectangle
        h, w = img.shape[:2]
        x_start, y_start = w // 4, h // 4
        x_end, y_end = 3 * w // 4, 3 * h // 4
        
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(img, "Hand Detection Area", (x_start + 10, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update gesture every 30 frames (simulating detection)
        self.counter += 1
        if self.counter % 30 == 0:
            self.current_gesture = random.choice(ISL_GESTURES)
            self.confidence = random.uniform(0.7, 0.99)
            self.gesture_history.append(self.current_gesture)
        
        # Display detected gesture
        if self.confidence > 0.75:
            cv2.rectangle(img, (10, 10), (400, 80), (0, 0, 0), -1)
            cv2.putText(img, f"Gesture: {self.current_gesture}",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(img, f"Confidence: {self.confidence:.2%}",
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===========================
# Main UI
# ===========================
st.title("🖐️ ISL Gesture Recognition - Demo")
st.markdown("### Real-time Indian Sign Language Gesture Detection")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.info(
        "**How it works:**\n"
        "- Point your hand towards the camera\n"
        "- The system will detect your gesture\n"
        "- Results appear with confidence score",
        icon="ℹ️"
    )

with col2:
    mode = st.radio("Mode:", ["Live", "Demo"], horizontal=True)

with col3:
    st.metric("Status", "🟢 Active")

st.markdown("---")

# Main video area
with st.container():
    webrtc_ctx = webrtc_streamer(
        key="ISL-Detection",
        mode="webrtc" if mode == "Live" else "recvonly",
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=GestureDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        =True,
    )

st.markdown("---")

# Info sections
tab1, tab2, tab3 = st.tabs(["📊 Supported Gestures", "ℹ️ About", "⚙️ Settings"])

with tab1:
    st.write("**Available ISL Gestures:**")
    cols = st.columns(4)
    for idx, gesture in enumerate(ISL_GESTURES):
        with cols[idx % 4]:
            st.write(f"🤟 {gesture}")

with tab2:
    st.markdown("""
    ### About ISL Gesture Recognition
    
    This application demonstrates real-time gesture recognition for Indian Sign Language (ISL).
    
    **Features:**
    - 🎥 Real-time video processing via WebRTC
    - 🤖 Hand gesture detection and classification
    - 📊 Confidence scoring for predictions
    - 🌐 Cloud-based deployment (Streamlit Cloud)
    
    **Technologies Used:**
    - Streamlit for web interface
    - OpenCV for image processing
    - WebRTC for real-time video streaming
    - TensorFlow Lite for gesture recognition (production version)
    """)

with tab3:
    st.write("**Application Settings:**")
    confidence_threshold = st.slider("Confidence Threshold:", 0.5, 0.99, 0.75)
    detection_speed = st.select_slider(
        "Detection Speed:",
        options=["Slow", "Medium", "Fast"],
        value="Medium"
    )
    st.write(f"✓ Threshold: {confidence_threshold:.2%}")
    st.write(f"✓ Speed: {detection_speed}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🚀 ISL Recognition | Powered by Streamlit | 2025"
    "</p>",
    unsafe_allow_html=True
)

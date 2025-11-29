import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import deque
import tensorflow as tf

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
# Load Resources
# ===========================
@st.cache_resource
def load_resources():
    try:
        # Load MediaPipe Holistic
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
        with open('label_map.pkl', 'rb') as f:
            label_map = pickle.load(f)
        
        idx_to_label = {v: k for k, v in label_map.items()}
        
        # Load LSTM model
        model = tf.keras.models.load_model('isl_gesture_model.h5')
        
        return holistic, mp_drawing, mp_holistic, idx_to_label, model, True
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None, False

holistic, mp_drawing, mp_holistic, idx_to_label, model, resources_loaded = load_resources()

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
    
    return np.array(vec, dtype=float) if len(vec) == FEATURE_SIZE else np.zeros(FEATURE_SIZE, dtype=float)

# ===========================
# Main UI
# ===========================
st.markdown("# 🖐️ ISL Live Real-Time Translation")
st.markdown("### Continuous Hand Gesture Recognition with Live Webcam")

if not resources_loaded:
    st.error("❌ Failed to load resources. Check if model files exist.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    sequence_length = st.slider("Sequence Length", 10, 50, 30, step=5)
    st.markdown("---")
    st.markdown("### 📋 Gesture Label Map")
    if idx_to_label:
        for idx, label in idx_to_label.items():
            st.write(f"**{idx}. {label.replace('_', ' ').title()}**")

# Main Content
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Use WebRTC for live streaming
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    class VideoProcessor:
        def __init__(self):
            self.buffer = deque(maxlen=sequence_length)
            self.last_prediction = None
            self.last_confidence = 0.0
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Process with MediaPipe
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Extract features
            features = extract_features(results)
            self.buffer.append(features)
            
            # Make prediction if buffer is full
            if len(self.buffer) == sequence_length:
                input_data = np.array(list(self.buffer))[np.newaxis, ...]
                predictions = model.predict(input_data, verbose=0)[0]
                top_idx = np.argmax(predictions)
                confidence = float(predictions[top_idx])
                
                if confidence > confidence_threshold:
                    self.last_prediction = idx_to_label.get(top_idx, "Unknown")
                    self.last_confidence = confidence
            
            # Draw landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
            
            # Add prediction text
            if self.last_prediction:
                text = f"{self.last_prediction} ({self.last_confidence:.2f})"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    webrtc_ctx = webrtc_streamer(
        key="ISL-live-translation",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        media_stream_constraints_options={"max_width": 1280},
    )

with col2:
    st.subheader("📊 Live Translation")
    
    # Real-time translation display
    if webrtc_ctx.state.playing:
        st.info("📹 Webcam is running - position your hand in frame")
        st.metric("Status", "🟊 Live", help="Camera streaming")
    else:
        st.warning("⚠️ Webcam not started")
        st.metric("Status", "🔴 Waiting", help="Start camera to begin")
    
    st.markdown("#### Latest Prediction")
    placeholder_text = st.empty()
    placeholder_conf = st.empty()
    
    # Update placeholders with real predictions from VideoProcessor
    if webrtc_ctx.state.playing:
        # This will be updated continuously by the VideoProcessor
        placeholder_text.success("Listening...") if webrtc_ctx else None

st.markdown("---")
st.markdown("🚀 Real-Time ISL Translation | Powered by Streamlit, MediaPipe & TensorFlow LSTM")

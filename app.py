import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ===========================
# ISL Gesture Labels
# ===========================
ISL_GESTURES = [
    "help_you",                # 0
    "congratulation",          # 1
    "hi_how_are_you",          # 2
    "i_am_hungry",             # 3
    "take_care_of_yourself"    # 4
]

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="ISL Real-Time Detection",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Load Model
# ===========================
@st.cache_resource
def load_model():
    """Load TensorFlow Lite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        return interpreter, True
    except Exception as e:
        return None, False

interpreter, model_loaded = load_model()

if model_loaded and interpreter is not None:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# ===========================
# Preprocess Function
# ===========================
def preprocess_frame(frame):
    """Preprocess frame for model inference - FIXED"""
    try:
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype('float32') / 255.0
        
        # Add batch dimension: (1, 224, 224, 3)
        frame_expanded = np.expand_dims(frame_normalized, axis=0)
        
        return frame_expanded
    except Exception as e:
        return None

def predict_gesture(frame):
    """Predict gesture from frame"""
    if not model_loaded or interpreter is None:
        return None, None, None
    
    try:
        # Preprocess
        img_input = preprocess_frame(frame)
        if img_input is None:
            return None, None, None
        
        # Set tensor and invoke
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        
        # Get output
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        
        return top_idx, confidence, predictions
    except Exception as e:
        return None, None, None

# ===========================
# Main UI
# ===========================
st.markdown("# 🖐️ ISL Real-Time Gesture Recognition")
st.markdown("### Live Translation of Indian Sign Language")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, step=0.05)
    show_all_predictions = st.checkbox("Show All Predictions", value=True)
    
    st.markdown("---")
    st.markdown("### 📋 Supported Gestures")
    for i, gesture in enumerate(ISL_GESTURES, 1):
        st.write(f"**{i}. {gesture.replace('_', ' ').title()}**")

# Main layout
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    if model_loaded:
        # Camera input
        camera_frame = st.camera_input("Point your hand towards camera")
        
        if camera_frame is not None:
            # Convert to numpy array
            image = Image.open(camera_frame)
            image_np = np.array(image)
            
            # Display frame
            st.image(image_np, use_column_width=True, channels="RGB", caption="Captured Frame")
            
            # Make prediction
            top_idx, confidence, predictions = predict_gesture(image_np)
            
            if top_idx is not None and confidence is not None:
                gesture_name = ISL_GESTURES[top_idx].replace('_', ' ').title()
                
                st.markdown("---")
                
                # Display result
                if confidence >= confidence_threshold:
                    st.success(f"✅ **DETECTED: {gesture_name}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta="Above threshold")
                else:
                    st.warning(f"⚠️ **LOW CONFIDENCE: {gesture_name}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%", delta="Below threshold")
                
                # All predictions
                if show_all_predictions and predictions is not None:
                    st.markdown("#### 📊 All Predictions")
                    for i, pred in enumerate(predictions):
                        pred_value = float(pred)
                        bar_label = f"{i+1}. {ISL_GESTURES[i].replace('_', ' ').title()}"
                        st.progress(min(pred_value, 1.0), text=bar_label)
            else:
                st.error("❌ Error in gesture detection")
        else:
            st.info("📷 Click camera button to capture and detect gesture")
    else:
        st.error("⚠️ Model not loaded. Ensure isl_gesture_model.tflite exists in the directory.")

with col2:
    st.subheader("📊 Status")
    
    if model_loaded:
        st.metric("Status", "🟢 Ready", help="Model loaded successfully")
        st.metric("Model", "✅ Yes", help="TensorFlow Lite model")
    else:
        st.metric("Status", "🔴 Error", help="Model failed to load")
        st.metric("Model", "❌ No", help="Check model file")
    
    st.metric("Gestures", len(ISL_GESTURES), help="Total supported gestures")
    
    st.markdown("#### ✨ Supported")
    for i, g in enumerate(ISL_GESTURES, 1):
        st.write(f"**{i}. {g.replace('_', ' ').title()}**")

st.markdown("---")
st.markdown("🚀 Real-Time ISL Recognition | Powered by Streamlit & TensorFlow Lite")

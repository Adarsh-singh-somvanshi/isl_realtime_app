import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import time

# ===========================
# ISL Gesture Names (Correct Labels)
# ===========================
ISL_GESTURES = {
    0: "Namaste",
    1: "Yes",
    2: "No",
    3: "Hello",
    4: "Goodbye"
}

# ===========================
# Page Configuration
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
    try:
        interpreter = tf.lite.Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

interpreter = load_model()

# ===========================
# Main UI
# ===========================
st.markdown("# 🖐️ ISL Real-Time Gesture Recognition")
st.markdown("### Live Translation of Indian Sign Language Gestures")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05)
    process_every = st.slider("Process Every N Frames", 1, 5, 1)
    display_fps = st.checkbox("Show FPS", value=True)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("#### 🎥 Live Camera Stream")
    FRAME_WINDOW = st.image([])
    status_text = st.empty()
    results_container = st.container()

with col2:
    st.markdown("#### 📊 Status")
    st.metric("Status", "🟢 Live" if interpreter else "⚠️ Error")
    st.metric("Model", "✅ Loaded" if interpreter else "❌ Failed")
    st.metric("Gestures", len(ISL_GESTURES))
    st.markdown("#### Supported Gestures")
    for i, gesture in ISL_GESTURES.items():
        st.write(f"**{i+1}.** {gesture}")

# ===========================
# Real-Time Processing
# ===========================
if interpreter:
    st.markdown("---")
    
    if st.button("▶️ Start Live Detection", key="start_btn"):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("⚠️ Cannot access camera. Please check permissions.")
        else:
            st.success("📹 Camera opened successfully!")
            frame_count = 0
            fps_time = time.time()
            fps = 0
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % process_every == 0:
                    # Preprocess frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    frame_normalized = frame_resized.astype('float32') / 255.0
                    frame_expanded = np.expand_dims(frame_normalized, axis=0)
                    
                    # Run inference
                    try:
                        interpreter.set_tensor(input_details[0]['index'], frame_expanded)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        # Get top prediction
                        top_idx = np.argmax(predictions)
                        confidence = predictions[top_idx]
                        gesture_name = ISL_GESTURES.get(top_idx, f"Unknown ({top_idx})")
                        
                        # Calculate FPS
                        if frame_count % 30 == 0:
                            fps = 30 / (time.time() - fps_time)
                            fps_time = time.time()
                        
                        # Add text to frame
                        if confidence >= confidence_threshold:
                            color = (0, 255, 0)  # Green
                            text = f"{gesture_name}: {confidence*100:.1f}%"
                        else:
                            color = (0, 165, 255)  # Orange
                            text = f"Low confidence: {confidence*100:.1f}%"
                        
                        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                        
                        if display_fps:
                            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    except Exception as e:
                        cv2.putText(frame, f"Error: {str(e)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, use_column_width=True)
                
                # Update status
                if confidence >= confidence_threshold:
                    with results_container:
                        st.success(f"✅ **{gesture_name}** - {confidence*100:.1f}%")
                
                # Check for stop button (optional: add stop button)
                time.sleep(0.01)  # Prevent CPU overload
else:
    st.error("🛠️ Model failed to load. Cannot proceed with detection.")

# Footer
st.markdown("---")
st.markdown("🚀 ISL Real-Time Recognition | Powered by Streamlit & TensorFlow")

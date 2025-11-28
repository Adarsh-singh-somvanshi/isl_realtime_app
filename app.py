import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# ===========================
# ISL Gesture Labels (CORRECT)
# ===========================
ISL_GESTURES = [
    "Namaste",    # 0
    "Yes",        # 1
    "No",         # 2
    "Hello",      # 3
    "Goodbye"     # 4
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
    try:
        interpreter = tf.lite.Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

interpreter = load_model()

# ===========================
# UI
# ===========================
st.markdown("# 🖐️ ISL Real-Time Gesture Recognition")
st.markdown("### Live Translation of Indian Sign Language")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)
    refresh_rate = st.slider("Refresh Rate (ms)", 100, 1000, 500)

# Main layout
col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("🎥 Capture & Detect")
    img_file = st.camera_input("Point your hand towards camera")
    
    if img_file is not None:
        # Read image
        image = Image.open(img_file)
        image_np = np.array(image)
        
        # Convert to BGR if needed
        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Preprocess
        img_resized = cv2.resize(image_np, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        # Inference
        if interpreter:
            try:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], img_expanded)
                interpreter.invoke()
                
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                top_idx = np.argmax(predictions)
                conf = predictions[top_idx]
                gesture = ISL_GESTURES[top_idx]
                
                # Display image
                st.image(image_np, use_column_width=True, channels="BGR", caption="Captured Frame")
                
                # Results
                if conf >= confidence:
                    st.success(f"✅ **DETECTED: {gesture}** ({conf*100:.1f}%)")
                else:
                    st.warning(f"⚠️ Low Confidence ({conf*100:.1f}%)")
                
                # All predictions
                st.markdown("#### All Predictions")
                for i, pred in enumerate(predictions):
                    bar_label = f"{i+1}. {ISL_GESTURES[i]}"
                    st.progress(min(pred, 1.0), text=bar_label)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error("⚠️ Model not loaded")
    else:
        st.info("📷 Click camera button to capture gesture")

with col2:
    st.subheader("📊 Status")
    st.metric("Status", "🟢 Ready" if interpreter else "⚠️ Error")
    st.metric("Model", "✅ Yes" if interpreter else "❌ No")
    st.metric("Gestures", len(ISL_GESTURES))
    
    st.markdown("#### Supported")
    for i, g in enumerate(ISL_GESTURES, 1):
        st.write(f"**{i}. {g}**")

st.markdown("---")
st.markdown("🚀 Real-Time ISL Recognition | Powered by Streamlit & TensorFlow")

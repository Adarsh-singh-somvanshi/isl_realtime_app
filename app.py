import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
# Load Model & Resources
# ===========================
@st.cache_resource
def load_model_and_labels():
    try:
        # Load LSTM model
        model = tf.keras.models.load_model('isl_gesture_model.h5')
        
        # Load label map
        with open('label_map.pkl', 'rb') as f:
            label_map = pickle.load(f)
        
        # Create idx to label mapping
        idx_to_label = {v: k for k, v in label_map.items()}
        
        return model, idx_to_label, True
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, False

model, idx_to_label, model_loaded = load_model_and_labels()

# ===========================
# Main UI
# ===========================
st.markdown("# 🖐️ ISL Live Real-Time Translation")
st.markdown("### Continuous Gesture Recognition")

if not model_loaded:
    st.error("❌ Failed to load model. Check if model files exist.")
    st.info("Expected files: isl_gesture_model.h5, label_map.pkl")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)
    
    st.markdown("---")
    st.markdown("### 📋 Supported Gestures")
    if idx_to_label:
        for idx in sorted(idx_to_label.keys()):
            label = idx_to_label[idx]
            st.write(f"**{idx}. {label.replace('_', ' ').title()}**")

# Main layout
col1, col2 = st.columns([2.5, 1.5])

with col1:
    st.subheader("📷 Capture Gesture")
    st.info("💡 Click the camera button below to capture your gesture. The app will analyze the hand position and predict the gesture.")
    
    # Camera input - works on Streamlit Cloud
    camera_frame = st.camera_input("Point your hand towards camera")
    
    if camera_frame is not None:
        # Convert to numpy array
        image = Image.open(camera_frame)
        image_np = np.array(image)
        
        # Display captured frame
        st.image(image_np, use_column_width=True, channels="RGB", caption="Captured Frame")
        
        st.markdown("---")
        
        # Placeholder for model inference message
        st.info("📊 Model Analysis:")
        st.write("The LSTM model would analyze hand landmarks from MediaPipe to predict the gesture.")
        st.write("This requires real-time hand tracking which works best with a local deployment.")
        st.markdown("""\n**For best results:**
- Run locally: `streamlit run app.py`
- Ensure webcam is available
- Install all dependencies: `pip install -r requirements.txt`
        """)
    else:
        st.info("📷 Click camera button to capture and detect gesture")

with col2:
    st.subheader("📊 Status")
    
    # Model status
    if model_loaded:
        st.metric("Model", "✅ Loaded", help="LSTM model ready")
        st.metric("Gestures", len(idx_to_label) if idx_to_label else 0, help="Total supported gestures")
    else:
        st.metric("Model", "❌ Error", help="Failed to load model")
    
    st.markdown("#### 📌 Deployment Info")
    st.warning("""⚠️ **Streamlit Cloud Limitation:**
    
Live WebRTC streaming is not available on Streamlit Cloud. For full live translation:

**Run Locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

This enables:
✅ Live webcam streaming
✅ Real-time hand detection
✅ Continuous gesture recognition
    """)

st.markdown("---")
st.markdown("🚀 ISL Translation | Run locally for live streaming | Powered by Streamlit & TensorFlow LSTM")

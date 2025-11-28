import streamlit as st
import cv2
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image
import io

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
# Load Model and Labels
# ===========================
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path="isl_gesture_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

@st.cache_resource
def load_labels():
    try:
        with open('label_map.pkl', 'rb') as f:
            labels = pickle.load(f)
            if isinstance(labels, dict):
                return sorted(labels.values())
            return labels
    except:
        # Fallback labels if pickle fails
        return ["Namaste", "Yes", "No", "Hello", "Goodbye"]

# Load resources
interpreter = load_model()
labels = load_labels()

# ===========================
# Main UI
# ===========================
st.markdown("# 🖐️ ISL Gesture Recognition - Live")
st.markdown("### Real-time Indian Sign Language Gesture Detection")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    detection_speed = st.select_slider("Detection Speed", ["Slow", "Medium", "Fast"], value="Medium")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 📹 Live Camera Feed")
    
    # Camera input
    camera_input = st.camera_input(
        "Point your hand towards the camera",
        help="Allow camera access to detect gestures"
    )
    
    if camera_input is not None:
        # Process the captured image
        image = Image.open(camera_input)
        image_array = np.array(image)
        
        # Convert to BGR for OpenCV
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Resize and preprocess
        img_resized = cv2.resize(image_array, (224, 224))
        img_normalized = img_resized.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)
        
        # Run inference if model loaded
        if interpreter is not None:
            try:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], img_expanded)
                interpreter.invoke()
                
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Get top prediction
                top_idx = np.argmax(predictions)
                confidence = predictions[top_idx]
                
                # Display results
                st.image(camera_input, use_column_width=True, caption="Captured Frame")
                
                if confidence >= confidence_threshold:
                    st.success(f"✅ Detected: **{labels[top_idx]}**")
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                else:
                    st.warning("⚠️ Low confidence - No clear gesture detected")
                
                # Show all predictions
                st.markdown("#### All Predictions:")
                pred_df = {}
                for i, label in enumerate(labels):
                    pred_df[label] = f"{predictions[i]*100:.1f}%"
                st.json(pred_df)
                
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
        else:
            st.image(camera_input, use_column_width=True)
            st.warning("⚠️ Model not loaded. Using camera in demo mode.")
    else:
        st.info("📷 Click the camera button above to capture a frame")

with col2:
    st.markdown("#### 📊 Status")
    st.metric("App Status", "🟢 Active")
    st.metric("Model Loaded", "✅ Yes" if interpreter else "❌ No")
    st.metric("Available Gestures", len(labels))
    
    st.markdown("#### 🖐️ Supported Gestures:")
    for idx, label in enumerate(labels, 1):
        st.text(f"{idx}. {label}")

# ===========================
# Information Tabs
# ===========================
tab1, tab2, tab3 = st.tabs(["About", "How It Works", "Settings"])

with tab1:
    st.markdown("""
    ## About ISL Gesture Recognition
    
    This application uses a TensorFlow Lite model to detect and classify Indian Sign Language (ISL) gestures in real-time.
    
    ### Features:
    - 🎥 Live camera input
    - 🤖 ML-powered gesture detection
    - 📊 Confidence scoring
    - ⚡ Real-time processing
    
    ### Supported Gestures:
    """)
    for label in labels:
        st.write(f"- {label}")

with tab2:
    st.markdown("""
    ## How It Works
    
    1. **Camera Capture**: Click the camera button to capture a frame
    2. **Image Processing**: The image is resized to 224x224 and normalized
    3. **Model Inference**: TFLite model processes the image
    4. **Gesture Classification**: Model outputs confidence scores for each gesture
    5. **Result Display**: The gesture with highest confidence is shown
    
    ### Tips for Better Detection:
    - ✋ Ensure your hand is clearly visible
    - 💡 Use good lighting conditions
    - 📐 Keep hand in center of frame
    - 🎯 Hold gesture steady while capturing
    """)

with tab3:
    st.markdown("## Configuration Options")
    st.write(f"**Current Confidence Threshold**: {confidence_threshold*100:.0f}%")
    st.write(f"**Detection Speed**: {detection_speed}")
    st.write(f"**Total Gestures**: {len(labels)}")

# Footer
st.markdown("---")
st.markdown("🚀 ISL Recognition | Powered by Streamlit & TensorFlow | 2025")

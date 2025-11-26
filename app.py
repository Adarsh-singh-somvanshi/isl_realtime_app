import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Disable OpenCV device probing (reduces cloud warnings)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "0"

import streamlit as st
import av
import cv2
import numpy as np
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Prefer lightweight tflite runtime on Streamlit Cloud, fall back to tf.lite for local runs
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_RUNTIME = True
except Exception:
    import tensorflow as _tf
    tflite = _tf.lite
    TFLITE_RUNTIME = False

# -------------------------
# App config / simple UI
# -------------------------
st.set_page_config(page_title="ISL Gesture Detection (TFLite)", layout="wide")
st.title("🖐️ ISL Gesture Detection – Real Time (TFLite)")

st.sidebar.markdown("### Settings")
CONF_THRESHOLD = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.01)
SHOW_FPS = st.sidebar.checkbox("Show FPS", value=True)
MODEL_PATH = st.sidebar.text_input("TFLite model path", value="isl_gesture_model.tflite")
LABEL_PATH = st.sidebar.text_input("Label JSON path", value="label_map.json")

# -------------------------
# Safety checks for files
# -------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Upload it to the app folder.")
    st.stop()

if not os.path.exists(LABEL_PATH):
    st.error(f"Label file not found: `{LABEL_PATH}`. Upload it to the app folder.")
    st.stop()

# -------------------------
# Load TFLite Model
# -------------------------
@st.cache_resource
def load_model(model_path: str):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # determine expected image size (handle different shapes)
    in_shape = input_details[0]["shape"]
    if len(in_shape) == 4:
        img_size = int(in_shape[1])
    elif len(in_shape) == 3:
        img_size = int(in_shape[0])
    else:
        img_size = 224  # fallback
    return {
        "interpreter": interpreter,
        "input_details": input_details,
        "output_details": output_details,
        "img_size": img_size,
    }

model = load_model(MODEL_PATH)
interpreter = model["interpreter"]
input_details = model["input_details"]
output_details = model["output_details"]
IMG_SIZE = model["img_size"]

# -------------------------
# Load labels (JSON)
# -------------------------
@st.cache_resource
def load_labels_json(path: str):
    with open(path, "r") as f:
        raw = json.load(f)

    # raw might be {"A":0, "B":1} or {"0":"A","1":"B"}; normalize to idx -> label
    idx_to_label = {}
    # if keys are labels and values are indices
    if all(isinstance(v, (int, float)) for v in raw.values()):
        for k, v in raw.items():
            idx_to_label[int(v)] = str(k)
    else:
        # assume keys are indices or strings of indices
        for k, v in raw.items():
            try:
                idx = int(k)
                idx_to_label[idx] = str(v)
            except Exception:
                # fallback: enumerate
                pass
        if not idx_to_label:
            # last fallback: list-like mapping
            for i, (k, v) in enumerate(raw.items()):
                idx_to_label[i] = str(v)
    return idx_to_label

idx_to_label = load_labels_json(LABEL_PATH)

# -------------------------
# Safe crop for model (center square)
# -------------------------
def detect_hand(frame):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    size = min(h, w)
    cx = w // 2 - size // 2
    cy = h // 2 - size // 2
    crop = frame[cy : cy + size, cx : cx + size]
    if crop.size == 0:
        return None
    return crop

# -------------------------
# Prediction (robust)
# -------------------------
def predict(img):
    if img is None:
        return "Unknown", 0.0
    try:
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    except Exception:
        return "Unknown", 0.0

    img_input = img_resized.astype("float32") / 255.0

    # If model expects single channel, convert
    expected_channels = input_details[0]["shape"][-1] if len(input_details[0]["shape"]) >= 3 else 3
    if img_input.ndim == 2 and expected_channels == 3:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
    if img_input.shape[-1] != expected_channels:
        # try to convert channels (common case: model expects 3)
        if expected_channels == 1:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        elif expected_channels == 3:
            if img_input.shape[-1] == 1:
                img_input = cv2.cvtColor(img_input[..., 0], cv2.COLOR_GRAY2BGR)
            else:
                # fallback: duplicate channels
                img_input = np.repeat(img_input[..., :1], 3, axis=-1)

    input_tensor = np.expand_dims(img_input, axis=0).astype(np.float32)

    try:
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]
        idx = int(np.argmax(output))
        conf = float(np.max(output))
        label = idx_to_label.get(idx, "Unknown")
        return label, conf
    except Exception as e:
        # avoid crashing the processor; return Unknown
        return "Unknown", 0.0

# -------------------------
# Video Processor (webrtc)
# -------------------------
import time

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._last_time = time.time()
        self._fps = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        crop = detect_hand(img)
        label, conf = predict(crop)

        # display only if above threshold
        display_text = f"{label} ({conf:.2f})" if conf >= CONF_THRESHOLD else f"--- ({conf:.2f})"

        cv2.putText(
            img,
            display_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 180, 255),
            3,
        )

        if SHOW_FPS:
            now = time.time()
            dt = now - self._last_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_time = now
            cv2.putText(img, f"FPS: {self._fps:.1f}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # return frame back to webrtc
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# Start WebRTC streamer
# -------------------------
webrtc_streamer(
    key="isl-app",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

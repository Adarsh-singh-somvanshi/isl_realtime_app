# Optimized ISL Live Translation - Faster Camera & Gesture Detection
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# Configuration - Optimized for Speed
SEQUENCE_LENGTH = 12
FEATURE_SIZE = 138
CONFIDENCE_THRESHOLD = 0.55

DEFAULT_LABEL_MAP = {
    0: 'help_you',
    1: 'congratulation',
    2: 'hi_how_are_you',
    3: 'i_am_hungry',
    4: 'take_care_of_yourself'
}

st.set_page_config(page_title="ISL Live", page_icon="🖐️", layout="wide")

def load_label_map():
    for f in ['label_map_updated.txt', 'label_map.txt']:
        if os.path.exists(f):
            try:
                lm = {}
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().rsplit(' ', 1)
                            if len(parts) == 2:
                                lm[int(parts[1])] = parts[0]
                if lm:
                    return lm, {v:k for k,v in lm.items()}
            except: pass
    return DEFAULT_LABEL_MAP, {v:k for k,v in DEFAULT_LABEL_MAP.items()}

@st.cache_resource
def load_model():
    try:
        mp_h = mp.solutions.holistic
        holistic = mp_h.Holistic(static_image_mode=False, model_complexity=0, min_detection_confidence=0.35, min_tracking_confidence=0.35)
        lm, li = load_label_map()
        interp = tf.lite.Interpreter('isl_gesture_model.tflite')
        interp.allocate_tensors()
        return holistic, mp.solutions.drawing_utils, mp_h, lm, li, interp, interp.get_input_details(), interp.get_output_details(), True
    except Exception as e:
        st.error(f"Load error: {e}")
        return (None,)*9

holistic, mp_draw, mp_h, label_map, l2i, interp, input_d, output_d, ok = load_model()

def extract_features(results):
    v = []
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                v.extend([lm.x, lm.y, lm.z])
        else:
            v.extend([0]*63)
    return np.array(v, dtype=np.float32) if len(v)==138 else np.zeros(138, dtype=np.float32)

st.title("🖐️ ISL Live Translation")
st.subheader("Gesture Recognition")

if not ok:
    st.error("Failed to load model")
    st.stop()

st.success("✅ Ready!")

with st.sidebar:
    st.header("⚙️ Settings")
    conf_th = st.slider("Confidence", 0.2, 1.0, 0.55, 0.05)
    seq_len = st.slider("Frames", 5, 25, 12, 1)
    st.divider()
    for i in sorted(label_map.keys()):
        st.write(f"{i}. {label_map[i].replace('_',' ').title()}")

st.info("📹 Click START to begin. Position hand in frame.")

rtc_cfg = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Proc:
    def __init__(self):
        self.buf = deque(maxlen=seq_len)
        self.pred = None
        self.conf = 0
        self.cnt = 0
    
    def recv(self, f):
        try:
            img = f.to_ndarray("bgr24")
            h, w, c = img.shape
            res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            feat = extract_features(res)
            self.buf.append(feat)
            self.cnt += 1
            
            if len(self.buf) == seq_len:
                try:
                    data = np.array(list(self.buf), dtype=np.float32)[np.newaxis, ...]
                    interp.set_tensor(input_d[0]['index'], data)
                    interp.invoke()
                    pred = interp.get_tensor(output_d[0]['index'])[0]
                    idx = np.argmax(pred)
                    c_val = float(pred[idx])
                    if c_val > conf_th:
                        self.pred = label_map.get(idx, "?")
                        self.conf = c_val
                except: pass
            
            if res.left_hand_landmarks:
                mp_draw.draw_landmarks(img, res.left_hand_landmarks, mp_h.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1),
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2))
            if res.right_hand_landmarks:
                mp_draw.draw_landmarks(img, res.right_hand_landmarks, mp_h.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                    mp_draw.DrawingSpec(color=(255,0,0), thickness=2))
            
            if self.pred:
                cv2.putText(img, f"{self.pred.title()} ({self.conf:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"F:{self.cnt}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            return av.VideoFrame.from_ndarray(img, "bgr24")
        except:
            return f

ctx = webrtc_streamer(
    key="isl",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_cfg,
    video_processor_factory=Proc,
    media_stream_constraints={"audio": False, "video": {"width": {"ideal": 480}}},
    async_processing=False
)

if ctx.state.playing:
    st.success("✅ Camera LIVE")
else:
    st.warning("⚠️ Start camera above")

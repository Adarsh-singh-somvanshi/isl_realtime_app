import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import tensorflow as tf

Interpreter = tf.lite.Interpreter

# ---- PAGE SETUP ----
st.set_page_config(page_title="ISL Translator", page_icon="🖐️", layout="wide")
st.title("🖐️ Indian Sign Language – Real-Time Translator")

# ---- CUSTOM FAST DRAWING ----
def draw_landmarks_fast(image, landmarks, connections, color=(0, 255, 0)):
    if not landmarks:
        return
    h, w, _ = image.shape

    points = {}
    for idx, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        cv2.circle(image, (cx, cy), 3, color, -1)

    if connections:
        for start_idx, end_idx in connections:
            if start_idx in points and end_idx in points:
                cv2.line(image, points[start_idx], points[end_idx], color, 2)

# ---- LOAD RESOURCES ----
@st.cache_resource
def load_model():

    # --- load label map ---
    if not os.path.exists("label_map.pkl"):
        st.error("label_map.pkl not found in repository.")
        st.stop()

    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)

    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # --- load tflite model ---
    model_path = "isl_gesture_model.tflite"

    if not os.path.exists(model_path):
        st.error(f"TFLite model not found: {model_path}")
        st.stop()

    interpreter = Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()

    return idx_to_label, interpreter

try:
    idx_to_label, interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Error loading model or label map: {e}")
    st.stop()

if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = ""

# ---- VIDEO PROCESSOR ----
class TFLiteProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.sequence = []
        self.last_pred = "Waiting..."
        self.last_conf = 0.0
        self.last_final = None

    def extract_features(self, results):
        vec = []

        # shoulder points
        if results.pose_landmarks:
            for idx in [11, 12]:
                lm = results.pose_landmarks.landmark[idx]
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 6)

        # hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 63)

        # palm base
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                lm = hand.landmark[0]
                vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 3)

        return np.array(vec, dtype=np.float32)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img.flags.writeable = False

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        img.flags.writeable = True

        # draw hands only
        if results.left_hand_landmarks:
            draw_landmarks_fast(
                img, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS, (255, 0, 0)
            )
        if results.right_hand_landmarks:
            draw_landmarks_fast(
                img, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS, (0, 255, 0)
            )

        # ---- prediction ----
        feat = self.extract_features(results)
        self.sequence.append(feat)
        self.sequence = self.sequence[-32:]

        if len(self.sequence) == 32:
            seq = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]["index"], seq)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            idx = int(np.argmax(preds))
            self.last_pred = idx_to_label.get(idx, "Unknown")
            self.last_conf = float(preds[idx]) * 100

            if self.last_conf > 80:
                if self.last_pred != self.last_final:
                    st.session_state.translated_sentence += " " + self.last_pred
                    self.last_final = self.last_pred

        # ui text
        cv2.rectangle(img, (0, 0), (900, 40), (0, 0, 0), -1)
        cv2.putText(
            img, f"{self.last_pred} ({self.last_conf:.1f}%)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- WEBRTC CONFIG ----
WEBRTC_CONFIG = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}
rtc_configuration = RTCConfiguration(WEBRTC_CONFIG)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Live Camera Feed")
    webrtc_streamer(
        key="isl-live",
        video_processor_factory=TFLiteProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("📝 Translation Output")
    st.write(st.session_state.translated_sentence)
    if st.button("Clear Text"):
        st.session_state.translated_sentence = ""
        st.rerun()

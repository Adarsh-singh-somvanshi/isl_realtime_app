import streamlit as st
import cv2
import numpy as np
import av
import mediapipe as mp
import pickle
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import os

# Fix for MediaPipe + WebRTC
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ---- LOAD TFLITE ----
try:
    import tensorflow.lite as tflite
    Interpreter = tflite.Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


# ---- PAGE SETUP ----
st.set_page_config(page_title="ISL Translator", page_icon="🖐️", layout="wide")
st.title("🖐️ Indian Sign Language – Real-Time Translator")

st.write("Live gesture translation using MediaPipe + TFLite LSTM Model.")


# ---- LOAD MODEL + LABELS ----
@st.cache_resource
def load_resources():
    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)
        idx_to_label = {v: k for k, v in label_to_idx.items()}

    interpreter = Interpreter(model_path="isl_gesture_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return idx_to_label, interpreter, input_details, output_details


idx_to_label, interpreter, input_details, output_details = load_resources()


# ---- TEXT HISTORY ----
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = ""


# ---- VIDEO PROCESSOR ----
class TFLiteProcessor(VideoProcessorBase):

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.sequence = []
        self.last_pred = "Waiting..."
        self.last_conf = 0.0
        self.last_final = None

    def extract_features(self, results):
        vec = []

        # Shoulders
        if results.pose_landmarks:
            for idx in [11, 12]:
                lm = results.pose_landmarks.landmark[idx]
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 6)

        # Hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                for lm in hand.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 63)

        # Palm base of both hands
        for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand:
                lm = hand.landmark[0]
                vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0] * 3)

        return np.array(vec, dtype=np.float32)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

        # Extract feature sequence
        feat = self.extract_features(results)
        self.sequence.append(feat)
        self.sequence = self.sequence[-32:]

        # Predict
        if len(self.sequence) == 32:
            seq = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)

            interpreter.set_tensor(input_details[0]['index'], seq)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            idx = int(np.argmax(preds))
            self.last_pred = idx_to_label.get(idx, "Unknown")
            self.last_conf = float(preds[idx]) * 100

            # Add to sentence when prediction stable
            if self.last_conf > 70 and self.last_pred != "Unknown":
                if self.last_pred != self.last_final:
                    st.session_state.translated_sentence += " " + self.last_pred
                    self.last_final = self.last_pred

        # Draw prediction bar
        cv2.rectangle(img, (0, 0), (800, 40), (0, 0, 0), -1)
        cv2.putText(img, f"{self.last_pred} ({self.last_conf:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---- WEBRTC ----
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                    "turn:openrelay.metered.ca:443?transport=tcp"
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

# ---- LAYOUT ----
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
        st.experimental_rerun()

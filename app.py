import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import av
import cv2
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---- Mediapipe Tasks ----
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

# ---- TFLITE Interpreter ----
import tensorflow as tf
Interpreter = tf.lite.Interpreter


st.set_page_config(page_title="ISL Translator", page_icon="🖐️", layout="wide")
st.title("🖐️ Indian Sign Language – Real-Time Translator")


# ------------------------- LOAD MODELS -------------------------
@st.cache_resource
def load_all_models():
    # Load label map
    with open("label_map.pkl", "rb") as f:
        label_to_idx = pickle.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Load your classifier model
    interpreter = Interpreter(model_path="isl_gesture_model.tflite", num_threads=1)
    interpreter.allocate_tensors()

    # Load TFLite MediaPipe Tasks models (local)
    base = python.BaseOptions

    # Hand Detector
    hand_det = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base(model_asset_path="hand_detector.tflite"),
            num_hands=2
        )
    )

    # Hand Landmark model
    hand_lm = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base(model_asset_path="hand_landmark_full.tflite"),
            num_hands=2
        )
    )

    # Pose Landmark model
    pose_lm = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path="pose_landmark_lite.tflite")
        )
    )

    return idx_to_label, interpreter, hand_det, hand_lm, pose_lm


idx_to_label, interpreter, hand_detector, hand_landmark, pose_landmark = load_all_models()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ---------------------- FEATURE EXTRACTOR ----------------------
def extract_features_from_tasks(frame):
    mp_img = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame)

    # Pose
    pose = pose_landmark.detect(mp_img)
    pose_vec = []
    if pose.pose_landmarks:
        for idx in [11, 12]:
            lm = pose.pose_landmarks[0].landmark[idx]
            pose_vec.extend([lm.x, lm.y, lm.z])
    else:
        pose_vec.extend([0.0] * 6)

    # Hand Detection
    hands = hand_detector.detect(mp_img)
    hand_vec = []

    if hands.handedness:
        # Run landmarks
        lm_res = hand_landmark.detect(mp_img)
        if lm_res.hand_landmarks:
            for hand in lm_res.hand_landmarks:
                for lm in hand:
                    hand_vec.extend([lm.x, lm.y, lm.z])
        else:
            hand_vec.extend([0.0] * 63 * 2)
    else:
        hand_vec.extend([0.0] * 63 * 2)

    # Final feature vector
    return np.array(pose_vec + hand_vec, dtype=np.float32)



# ---------------------- VIDEO PROCESSOR ------------------------
class Processor(VideoProcessorBase):
    def __init__(self):
        self.seq = []
        self.last_pred = "Waiting..."
        self.last_conf = 0
        self.last_final = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract features
        feat = extract_features_from_tasks(rgb)
        self.seq.append(feat)
        self.seq = self.seq[-32:]

        # Predict
        if len(self.seq) == 32:
            seq = np.expand_dims(np.array(self.seq, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]["index"], seq)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            idx = int(np.argmax(preds))
            self.last_pred = idx_to_label.get(idx, "Unknown")
            self.last_conf = float(preds[idx]) * 100

            if self.last_conf > 80:
                if self.last_pred != self.last_final:
                    st.session_state.text += " " + self.last_pred
                    self.last_final = self.last_pred

        # Display
        cv2.putText(img, f"{self.last_pred} ({self.last_conf:.1f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------- STREAMLIT UI ---------------------------
if "text" not in st.session_state:
    st.session_state.text = ""

cols = st.columns(2)

with cols[0]:
    st.subheader("📸 Live Feed")
    webrtc_streamer(
        key="live",
        video_processor_factory=Processor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

with cols[1]:
    st.subheader("📝 Translation")
    st.write(st.session_state.text)
    if st.button("Clear"):
        st.session_state.text = ""
        st.rerun()

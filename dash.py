# 🧠 YOLOv8 Streamlit Object Detection App
# Supports: Image Upload | Video Upload | Webcam Stream

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import time
from PIL import Image
import numpy as np
import os

# 🚀 1. Page Configuration

st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🎯 YOLOv8 Object Detection App")
st.markdown("### Upload an image, video, or use your webcam for detection")

# 📦 2. Load YOLOv8 Model

@st.cache_resource
def load_model():
    model_path = "D:\Projects\yolo_object_detection\yolov8n.pt"  # update if path differs
    model = YOLO(model_path)
    return model

model = load_model()
st.sidebar.success("✅ Model Loaded Successfully")

# ⚙️ 3. Sidebar - Input Options

st.sidebar.header("📸 Input Source")
source_option = st.sidebar.radio(
    "Select input type:",
    ("Image Upload", "Video Upload", "Webcam")
)

# 🖼️ 4. Image Upload Detection

if source_option == "Image Upload":
    st.subheader("📁 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Convert to OpenCV format
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        # Convert PIL to OpenCV
        img_array = np.array(image)
        results = model.predict(img_array, save=False, conf=0.5)

        # Draw boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption="🎯 Detected Image", use_container_width=True)

# 🎥 5. Video Upload Detection

elif source_option == "Video Upload":
    st.subheader("🎬 Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save temp video file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Open video with OpenCV
        vid_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        # Process each frame
        while True:
            success, frame = vid_cap.read()
            if not success:
                break

            results = model.predict(frame, save=False, conf=0.5)
            annotated_frame = results[0].plot()

            # Display video
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        vid_cap.release()
        st.success("✅ Video Processing Completed")

# 📷 6. Webcam Live Detection

elif source_option == "Webcam":
    st.subheader("📹 Live Webcam Detection")

    # Start webcam stream
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            success, frame = cap.read()
            if not success:
                st.warning("⚠️ Unable to access webcam")
                break

            # YOLO detection
            results = model.predict(frame, save=False, conf=0.5)
            annotated_frame = results[0].plot()

            # Display result
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            # Stop if checkbox is unchecked
            run = st.checkbox("Stop Webcam", value=True)

        cap.release()
        st.success("✅ Webcam Stream Stopped")

# 🧾 7. Footer

st.markdown("---")
st.markdown("Developed by 🧠 **EDISON XAVIER**")

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Neosight", layout="centered")
st.title("Neosight - AI Vision Assistance for Blind People")

# Load YOLO model (auto download)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Camera input
image = st.camera_input("Capture Image")

if image is not None:
    # Convert image
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Run detection
    results = model(frame)
    objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            objects.append(label)

            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Show result
    if objects:
        text = ", ".join(list(set(objects)))
        st.success("Detected Objects: " + text)

        # Text to speech (mobile friendly)
        tts = gTTS("I see " + text)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        tts.save(temp_file.name)

        st.audio(temp_file.name)

    st.image(frame, channels="BGR")
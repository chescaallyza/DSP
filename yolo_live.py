import cv2
import time
import streamlit as st
from ultralytics import YOLO
from draw_utils_live import plot_boxes_live, color_map_live

# Function to handle live webcam detection
def live_detection(plot_boxes, model_path="best.pt", webcam_resolution=(1280, 720)):
    frame_width, frame_height = webcam_resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO(model_path).to('cpu')
    frame_placeholder, object_description_placeholder = st.empty(), st.empty()

    # Queue to store the latest 5 object descriptions
    description_queue = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame, labels, descriptions = plot_boxes_live(results, frame, model, color_map_live)
        current_time = time.time()

        # Add new descriptions to the queue (FIFO: First In First Out)
        for label, desc in zip(labels, descriptions):
            description_text = f"<span class='object-label'>{label}</span>: <span class='object-definition'>{desc}</span>"
            if len(description_queue) >= 5:
                description_queue.pop(0)  # Remove the oldest description if queue is full
            description_queue.append((description_text, current_time))

        # Filter descriptions to show only recent ones (within the last 5 seconds)
        description_queue = [
            (desc, timestamp) for desc, timestamp in description_queue
            if current_time - timestamp <= 5
        ]

        # Display the descriptions in a list format
        description_display = "<div class='title-box'>Detected Objects:</div>"
        for description, _ in description_queue:
            description_display += f"<div class='description-box'>{description}</div>"

        # Update the frame and description list in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        object_description_placeholder.markdown(description_display, unsafe_allow_html=True)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

import streamlit as st
from yolo_live import live_detection
from yolo_upload import upload_image_detection
from draw_utils import plot_boxes
from draw_utils_live import plot_boxes_live
from utils import load_css


# Main function
def main():
    st.markdown("<h1 style='text-align: center; color: #e0f7fa;'>YOLOv8 Object Detection Streamlit UI</h1>", unsafe_allow_html=True)
    load_css()

    # Option to choose between "Live Detection" or "Upload Image"
    option = st.selectbox("Select Mode", ["Select Mode", "Live Detection", "Upload Image"], index=0, help="Select between live webcam detection or upload an image for detection.")

    if option == "Select Mode":
        pass
    elif option == "Live Detection":
        # Call the live detection function
        live_detection(plot_boxes_live)
    elif option == "Upload Image":
        # Call the upload image detection function
        upload_image_detection(plot_boxes)

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from draw_utils import plot_boxes, color_map # Import color_map here
from collections import Counter
from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageOps, ImageDraw
from num2words import num2words 
from image_generation import query, generate_image, count_to_words
from image_adjustment import apply_style, adjust_image
import pandas as pd

# Function to handle image upload and process it with YOLO
def upload_image_detection(plot_boxes):
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        image_height, image_width, _ = image.shape
        max_size = 640
        if max(image_height, image_width) > max_size:
            if image_height > image_width:
                new_height = max_size
                new_width = int(image_width * (max_size / image_height))
            else:
                new_width = max_size
                new_height = int(image_height * (max_size / image_width))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if st.button("Start Detecting Objects"):
            model = YOLO("best.pt").to('cpu')  # Load your YOLO model
            results = model(image)

            if not results or not any(result.boxes for result in results):
                st.error("No objects detected. Please try the following troubleshooting steps:")
                st.markdown("- Ensure the image is clear and well-lit.")
                st.markdown("- Try adjusting the brightness or contrast of the image.")
                st.markdown("- Use a different image with more distinct objects.")
                return
            
            image_with_detection, detection_data = plot_boxes(results, image.copy(), model, color_map)

            # Display results
            col1, col2 = st.columns([2, 1])  # Adjust the column widths
            with col1:
                st.image(image_with_detection, channels="BGR", caption="Detection Results", use_container_width=True)
            with col2:
                st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

            # Summarize object counts
            object_counts = Counter([item["Object"] for item in detection_data])
            df_summary = pd.DataFrame(object_counts.items(), columns=["Object", "Count"])

            # Detailed table with horizontal expansion
            st.markdown("<h3 style='text-align: center;'>Detailed Detection Information (Expanded View)</h3>", unsafe_allow_html=True)
            expanded_data = []
            for obj in df_summary["Object"].unique():
                details = [item for item in detection_data if item["Object"] == obj]
                row = {
                    "Object": obj,
                    "Count": len(details),
                    "Location": [f"X: {loc[0]}, Y: {loc[1]}" for loc in [item["Location (x, y)"] for item in details]],
                    "Size": [f"W: {size[0]} x H: {size[1]} px" for size in [item["Size (Width x Height)"] for item in details]],
                    "Confidence": [item["Confidence"] for item in details],
                }
                expanded_data.append(row)
            df_expanded = pd.DataFrame(expanded_data)
            st.dataframe(df_expanded, use_container_width=True)

            # Generate prompt
            prompt = ", ".join([count_to_words(count, label) for label, count in object_counts.items()])
            full_prompt = f"An image featuring {prompt}."
            st.markdown(f"<h3 style='text-align: center;'>Generated Prompt: {full_prompt}</h3>", unsafe_allow_html=True)
            st.session_state.generated_prompt = full_prompt
            st.session_state.image_generated = False  # Reset image generation state

    # Generate Image button (after detection)
    if "generated_prompt" in st.session_state and st.session_state.generated_prompt:
        if not st.session_state.get("image_generated", False):
            if st.button("Generate Image from Prompt"):
                prompt = st.session_state.generated_prompt
                with st.spinner("Generating image..."):
                    st.session_state.generated_image = generate_image(prompt, image_size=(640, 640))
                st.session_state.image_generated = True  # Set state to indicate image has been generated

        else:
            st.success("Image has been generated. You can adjust or view it.")

    # Real-time brightness and contrast adjustment in two-column layout
    if "generated_image" in st.session_state and st.session_state.generated_image:
        original_image = st.session_state.generated_image

        # Use Streamlit columns for the layout
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<h3 style='text-align: center;'>Final Image</h3>", unsafe_allow_html=True)
            st.image(original_image, caption="Generated Image", use_container_width=True)

        with col2:
            st.markdown("<h3 style='text-align: center;'>Adjustments</h3>", unsafe_allow_html=True)

            # Sliders for adjustments
            brightness = st.slider("Brightness", -100, 100, 0, key="brightness")
            contrast = st.slider("Contrast", -100, 100, 0, key="contrast")
            saturation = st.slider("Saturation", -100, 100, 0, key="saturation")
            warm = st.slider("Warm", -100, 100, 0, key="warm")
            vignette = st.slider("Vignette", -100, 100, 0, key="vignette")

            # Apply adjustments
            adjusted_image = adjust_image(original_image, brightness, contrast, saturation, vignette, warm)
            st.session_state.adjusted_image = adjusted_image

            # Option for Art Style
            st.markdown("### Apply Art Style")
            styles = ["None", "Sketch", "Black and White", "Sepia", "Vintage", "Cool"]
            selected_style = st.radio("Choose Art Style", styles, key="art_style")

            # Apply the selected style to the adjusted image
            if selected_style != "None":
                final_image = apply_style(adjusted_image, selected_style)
            else:
                final_image = adjusted_image  # No style applied

        # Display the final adjusted image in column 1
        with col1:
            st.image(final_image, caption="Final Adjusted Image", use_container_width=True)

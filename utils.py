import argparse
import requests
import streamlit as st

# Parse webcam resolution arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

# Generate object descriptions via API
def generate_description(object_name: str) -> str:
    api_url = "http://127.0.0.1:1234/v1/completions"
    payload = {
        "model": "gemma-2-2b-it",
        "prompt": f"Provide a brief description of a {object_name} in one sentence.",
        "temperature": 0.7,
        "max_tokens": 50
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        description = data.get('choices', [{}])[0].get('text', "No description available.")
        return description.split('.')[0].strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Load CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

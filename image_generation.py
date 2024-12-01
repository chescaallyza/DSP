import requests
from PIL import Image
import io
from num2words import num2words

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise ValueError(f"API Error: {response.status_code}, {response.text}")
    return response.content

def count_to_words(count, label):
    word_count = num2words(count)
    if count == 1:
        return f"{word_count} {label}"  # Singular
    else:
        return f"{word_count} {label}s"  # Plural

def generate_image(prompt, image_size=(640, 640)):
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "height": image_size[1],
                "width": image_size[0],
                "guidance_scale": 12,
                "num_inference_steps": 75
            }
        }
        image_bytes = query(payload)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        return None

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = (300, 300)
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "bird_species_efficientnet.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ----------------------------
# LOAD MODEL (IMPORTANT FIX)
# ----------------------------
model = load_model(
    MODEL_PATH,
    compile=False,          # 🔥 FIXES Functional deserialization issue
    safe_mode=False         # 🔥 Needed for keras v3+
)

# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# TOP-K PREDICTION
# ----------------------------
def predict_top_k(image_path, k=5):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]

    top_indices = predictions.argsort()[-k:][::-1]

    results = [
        {
            "species": class_names[idx],
            "confidence": float(predictions[idx])
        }
        for idx in top_indices
    ]
    return results
# =======================================================
# 🌾 Rice Nitrogen Grade Prediction API (Google Drive)
# =======================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity
import json, io, os, gdown

# =======================================================
# ⚙️ Flask Setup
# =======================================================
app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)
THRESHOLD = 0.85
GRADES = ['Grade 2', 'Grade 3', 'Grade 4']

# =======================================================
# 📁 Google Drive File IDs
# =======================================================
# 👉 Replace these with YOUR file IDs from Google Drive folder:
MODEL_ID = "PASTE_YOUR_TFLITE_FILE_ID_HERE"
EMB_ID = "PASTE_YOUR_JSON_FILE_ID_HERE"

# =======================================================
# 📦 Local Filenames
# =======================================================
MODEL_PATH = "lcc_ensemble.tflite"
EMB_PATH = "rice_reference_embeddings.json"

# =======================================================
# ⬇️ Download files from Google Drive if missing
# =======================================================
def download_from_drive():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
    
    if not os.path.exists(EMB_PATH):
        print("📥 Downloading embeddings from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={EMB_ID}", EMB_PATH, quiet=False)

download_from_drive()

# =======================================================
# 🧠 Load Model + Embeddings
# =======================================================
print("🧠 Loading model and embeddings...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

feature_model = MobileNetV2(include_top=False, pooling='avg', input_shape=IMG_SIZE+(3,))

with open(EMB_PATH, 'r') as f:
    reference_embeddings = np.array(json.load(f))

print("✅ Model & embeddings loaded successfully!")

# =======================================================
# 🔍 Helper Functions
# =======================================================
def compute_embedding(image):
    img = image.convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img), axis=0)
    arr = preprocess_input(arr)
    emb = feature_model.predict(arr, verbose=0)
    return emb

def predict_grade(img):
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    return GRADES[pred]

# =======================================================
# 🌍 API Routes
# =======================================================
@app.route('/')
def home():
    return jsonify({
        "message": "Rice Grade Prediction API is running ✅",
        "author": "napplcc",
        "status": "online"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))

    emb = compute_embedding(img)
    sims = cosine_similarity(emb, reference_embeddings)
    max_sim = np.max(sims)

    if max_sim < THRESHOLD:
        return jsonify({
            "status": "not_rice",
            "similarity": float(max_sim),
            "message": "⚠️ Not a rice image"
        })

    grade = predict_grade(img)

    return jsonify({
        "status": "ok",
        "similarity": float(max_sim),
        "predicted_grade": grade
    })

# =======================================================
# 🚀 Run (for local or Render deployment)
# =======================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

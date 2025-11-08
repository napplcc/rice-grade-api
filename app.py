from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np, io, json, os, gdown
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)
THRESHOLD = 0.85
GRADES = ['Grade 2', 'Grade 3', 'Grade 4']

# ==== Google Drive file IDs ====
MODEL_ID = "YOUR_TFLITE_FILE_ID_HERE"
EMB_ID = "YOUR_EMBED_FILE_ID_HERE"

# ==== Local paths ====
MODEL_PATH = "lcc_ensemble.tflite"
EMB_PATH = "rice_reference_embeddings.json"

# ==== Download files if not already present ====
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(EMB_PATH):
    print("📥 Downloading embeddings from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={EMB_ID}", EMB_PATH, quiet=False)

# ==== Load model + embeddings ====
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

feature_model = MobileNetV2(include_top=False, pooling='avg', input_shape=IMG_SIZE+(3,))

with open(EMB_PATH, "r") as f:
    reference_embeddings = np.array(json.load(f))

@app.route("/")
def home():
    return jsonify({"message": "Rice API running ✅"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img), axis=0)
    arr_pre = preprocess_input(arr)

    emb = feature_model.predict(arr_pre, verbose=0)
    sims = cosine_similarity(emb, reference_embeddings)
    max_sim = np.max(sims)

    if max_sim < THRESHOLD:
        return jsonify({"status": "not_rice", "similarity": float(max_sim)})

    arr2 = np.expand_dims(np.array(img)/255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], arr2)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)
    grade = GRADES[pred]

    return jsonify({
        "status": "ok",
        "similarity": float(max_sim),
        "predicted_grade": grade
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

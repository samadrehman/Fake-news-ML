from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# ====== Load model and tokenizer ======
MODEL_PATH = "model.h5"           # your saved model file
TOKENIZER_PATH = "tokenizer.pkl"  # your saved tokenizer

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 150  # same as used during training


# ====== Home route ======
@app.route("/")
def home():
    return render_template("index.html")


# ====== Prediction route ======
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided."}), 400

    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Prediction
    pred = model.predict(padded)[0][0]

    # ====== Classification logic ======
    # You can tune these thresholds for your model
    if pred >= 0.75:
        label = "Fake"
    elif pred <= 0.25:
        label = "True"
    else:
        label = "Real / Possibly True"

    response = {
        "prediction": label,
        "confidence": float(pred)
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)

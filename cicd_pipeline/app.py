from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH_SVM = "cicd_pipeline/svm_model.pkl"
MODEL_PATH_GB = "cicd_pipeline/gb_model.pkl"

def load_model(model_type):
    if model_type == "svm":
        return joblib.load(MODEL_PATH_SVM)
    elif model_type == "gb":
        return joblib.load(MODEL_PATH_GB)
    else:
        raise ValueError("Unknown model type")

@app.route("/")
def home():
    return jsonify({"message": "CI/CD ML API running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "model" not in data:
        return jsonify({"error": "Missing model parameter (svm/gb)"}), 400

    model_type = data["model"]
    features = data.get("features")

    if features is None:
        return jsonify({"error": "Missing 'features' field"}), 400

    try:
        model = load_model(model_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return jsonify({"model": model_type, "prediction": int(prediction)}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
import os

app = Flask(__name__)

# =======================================
# LOAD BOTH MODELS
# =======================================
svm_model_path = "manual_pipeline/svm_model.pkl"
gb_model_path = "manual_pipeline/gb_model.pkl"

models_available = {}

# Load SVM model
if os.path.exists(svm_model_path):
    try:
        models_available["svm"] = joblib.load(svm_model_path)
        print("Model SVM berhasil dimuat.")
    except:
        print("Gagal memuat model SVM.")

# Load Gradient Boosting model
if os.path.exists(gb_model_path):
    try:
        models_available["gb"] = joblib.load(gb_model_path)
        print("Model Gradient Boosting berhasil dimuat.")
    except:
        print("Gagal memuat model Gradient Boosting.")


# =======================================
# ROUTES
# =======================================

@app.route("/", methods=["GET"])
def home():
    return {
        "message": "Titanic Survival Prediction API - Manual Pipeline",
        "models_loaded": list(models_available.keys())
    }, 200


@app.route("/models", methods=["GET"])
def get_models():
    return {
        "available_models": list(models_available.keys())
    }, 200


@app.route("/predict", methods=["POST"])
def predict():
    # Determine model choice
    model_choice = request.args.get("model", "").lower()

    if model_choice not in models_available:
        return {
            "error": f"Model '{model_choice}' tidak ditemukan. Gunakan ?model=svm atau ?model=gb"
        }, 400

    model = models_available[model_choice]

    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

        return {
            "model_used": model_choice,
            "prediction": int(prediction)
        }, 200

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }, 500


if __name__ == "__main__":
    print("Menjalankan Flask server manual…")
    app.run(host="0.0.0.0", port=5000)

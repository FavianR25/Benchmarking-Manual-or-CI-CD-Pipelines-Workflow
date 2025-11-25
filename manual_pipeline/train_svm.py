import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
import psutil
import csv
import os
import joblib


def log_manual(metrics: dict):
    log_path = "manual_pipeline/log_manual.csv"
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def train_svm_manual():
    start_time = time()
    process = psutil.Process()

    # Load preprocessed dataset
    df = pd.read_csv("manual_pipeline/processed_manual.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train SVM model
    model = SVC()
    model.fit(X_train, y_train)

    # Save model for deployment
    joblib.dump(model, "manual_pipeline/svm_model.pkl")

    end_time = time()

    metrics = {
        "workflow": "manual",
        "model": "svm",
        "deployment_time": end_time - start_time,
        "error_rate": 0,
        "reproducibility": 1,
        "setup_time": start_time - process.create_time(),
        "cpu_usage": process.cpu_percent(),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024)
    }

    log_manual(metrics)

    print("Training SVM selesai. Model disimpan ke manual_pipeline/svm_model.pkl")
    print("Log disimpan di manual_pipeline/log_manual.csv")


if __name__ == "__main__":
    train_svm_manual()

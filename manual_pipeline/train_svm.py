import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time
import psutil
import os
from utils import log_manual

def train_svm_model():
    start_time = time.time()
    setup_start = time.perf_counter()

    # Load dataset
    try:
        df = pd.read_csv("manual_pipeline/processed_manual.csv")
    except Exception as e:
        log_manual({
            "workflow": "manual",
            "model": "svm",
            "deployment_time": 0,
            "error_rate": 1,
            "reproducibility": 0,
            "setup_time": 0,
            "cpu_usage": 0,
            "memory_usage_mb": 0
        })
        raise e

    # Split dataset
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup time
    setup_time = time.perf_counter() - setup_start

    # Measure CPU & Memory before training
    process = psutil.Process(os.getpid())
    cpu_usage = process.cpu_percent(interval=0.2)
    mem_usage = process.memory_info().rss / (1024 * 1024)

    error_rate = 0

    try:
        model = SVC(kernel='rbf', probability=True)
        model.fit(X_train, y_train)
    except:
        error_rate = 1

    # Save model
    joblib.dump(model, "manual_pipeline/svm_model.pkl")

    # Deployment time
    deployment_time = time.time() - start_time

    # Reproducibility flag
    reproducibility = 1 if error_rate == 0 else 0

    # Log
    log_manual({
        "workflow": "manual",
        "model": "svm",
        "deployment_time": deployment_time,
        "error_rate": error_rate,
        "reproducibility": reproducibility,
        "setup_time": setup_time,
        "cpu_usage": cpu_usage,
        "memory_usage_mb": mem_usage
    })

if __name__ == "__main__":
    train_svm_model()

# cicd_pipeline/train_gb.py
import os
import sys
sys.path.append(os.path.dirname(__file__))

import time
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import log_cicd, process_cpu_seconds, process_memory_mb

def train_gb():
    print("[CI/CD] Training Gradient Boosting...")
    start_all = time.time()

    processed_path = os.path.join(os.path.dirname(__file__), "processed_cicd.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    setup_start = time.perf_counter()
    df = pd.read_csv(processed_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    setup_time = time.perf_counter() - setup_start

    cpu_before = process_cpu_seconds()

    train_start = time.time()
    error_rate = 0.0
    reproducibility = 1
    try:
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
    except Exception as e:
        error_rate = 1.0
        reproducibility = 0
        deployment_time = time.time() - train_start
        cpu_after = process_cpu_seconds()
        cpu_seconds = max(cpu_after - cpu_before, 0.0)
        cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
        mem_mb = process_memory_mb()
        log_cicd({
            "workflow": "cicd",
            "model": "gradient_boosting",
            "deployment_time": deployment_time,
            "error_rate": error_rate,
            "reproducibility": reproducibility,
            "setup_time": setup_time,
            "cpu_usage": round(cpu_percent, 4),
            "memory_usage_mb": round(mem_mb, 4)
        })
        raise

    deployment_time = time.time() - train_start
    cpu_after = process_cpu_seconds()
    cpu_seconds = max(cpu_after - cpu_before, 0.0)
    cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
    mem_mb = process_memory_mb()

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1.0 - acc

    non_critical_failure = 0
    if error_rate > 0.40:
        non_critical_failure = 1
    if cpu_usage < 1:
        non_critical_failure = 1
    if memory_usage_mb > (2 * baseline_memory):  # baseline diambil dari file atau konstanta
        non_critical_failure = 1
    if reproducibility == 0:
        non_critical_failure = 1

    model_path = os.path.join(os.path.dirname(__file__), "gb_model.pkl")
    joblib.dump(model, model_path)

    log_cicd({
        "workflow": "cicd",
        "model": "gradient_boosting",
        "deployment_time": round(deployment_time, 6),
        "error_rate": round(error_rate, 6),
        "reproducibility": reproducibility,
        "setup_time": round(setup_time, 6),
        "cpu_usage": round(cpu_percent, 4),
        "memory_usage_mb": round(mem_mb, 4)
    })

    print("[CI/CD] Gradient Boosting training done. Model saved to", model_path)

if __name__ == "__main__":
    train_gb()

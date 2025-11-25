# manual_pipeline/train_svm.py
import os
import sys
import time
from time import perf_counter

# make sure utils can be imported when running as `python manual_pipeline/train_svm.py`
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import psutil

from utils import log_manual

def _process_cpu_seconds():
    """Return process CPU time (user+system) in seconds via psutil (portable)."""
    times = psutil.Process().cpu_times()
    return times.user + times.system

def _process_memory_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def train_svm_model():
    start_all = time.time()
    setup_start = perf_counter()

    # Load processed dataset
    try:
        df = pd.read_csv("manual_pipeline/processed_manual.csv")
    except Exception as e:
        # If preprocess missing or broken -> log failure
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
        raise

    # Features / label
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Split dataset (use same split as CI/CD)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Setup time measured until before training
    setup_time = perf_counter() - setup_start

    # CPU seconds before training
    cpu_before = _process_cpu_seconds()

    # start training timer
    train_start = time.time()
    error_rate = 0.0
    reproducibility = 1
    try:
        model = SVC(kernel='rbf', probability=False)
        model.fit(X_train, y_train)
    except Exception:
        # training failed
        error_rate = 1.0
        reproducibility = 0
        # compute deployment time & cpu usage so failure still logged
        deployment_time = time.time() - train_start
        cpu_after = _process_cpu_seconds()
        cpu_seconds = max(cpu_after - cpu_before, 0.0)
        cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
        mem_mb = _process_memory_mb()
        log_manual({
            "workflow": "manual",
            "model": "svm",
            "deployment_time": round(deployment_time, 6),
            "error_rate": error_rate,
            "reproducibility": reproducibility,
            "setup_time": round(setup_time, 6),
            "cpu_usage": round(cpu_percent, 4),
            "memory_usage_mb": round(mem_mb, 4)
        })
        raise

    deployment_time = time.time() - train_start

    # CPU after training
    cpu_after = _process_cpu_seconds()
    cpu_seconds = max(cpu_after - cpu_before, 0.0)
    cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0

    # Memory after training
    mem_mb = _process_memory_mb()

    # Evaluate on test set -> compute error_rate properly
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1.0 - acc

    # Save model
    joblib.dump(model, "manual_pipeline/svm_model.pkl")

    # Log metrics
    log_manual({
        "workflow": "manual",
        "model": "svm",
        "deployment_time": round(deployment_time, 6),
        "error_rate": round(error_rate, 6),
        "reproducibility": reproducibility,
        "setup_time": round(setup_time, 6),
        "cpu_usage": round(cpu_percent, 4),
        "memory_usage_mb": round(mem_mb, 4)
    })

    #NCF
    non_critical_failure = 0
    if error_rate > 0.40:
        non_critical_failure = 1
    if cpu_usage < 1:
        non_critical_failure = 1
    if memory_usage_mb > (2 * baseline_memory):  # baseline diambil dari file atau konstanta
        non_critical_failure = 1
    if reproducibility == 0:
        non_critical_failure = 1

    print("SVM training finished. Model saved to manual_pipeline/svm_model.pkl")

if __name__ == "__main__":
    train_svm_model()

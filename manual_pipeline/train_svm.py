# manual_pipeline/train_svm.py
import os
import sys
import time
from time import perf_counter

# ensure utils import works when run as `python manual_pipeline/train_svm.py`
sys.path.append(os.path.dirname(__file__))

import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import psutil

from utils import log_manual

MB = 1024 * 1024


def _proc_cpu_seconds():
    """Return process CPU seconds (user + system) using psutil."""
    p = psutil.Process()
    t = p.cpu_times()
    return t.user + t.system


def _proc_mem_mb():
    p = psutil.Process()
    return p.memory_info().rss / MB


def train_svm_model():
    start_all = time.time()
    setup_start = perf_counter()

    # processed file (created by manual preprocess)
    processed_path = os.path.join(os.path.dirname(__file__), "processed_manual.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    # load
    df = pd.read_csv(processed_path)

    # features & label
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # split (same seed as CI/CD)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    setup_time = perf_counter() - setup_start

    # baseline memory BEFORE training (automatic baseline)
    baseline_memory_mb = _proc_mem_mb()

    cpu_before = _proc_cpu_seconds()
    train_start = time.time()

    error_rate = 0.0
    reproducibility = 1
    try:
        model = SVC(kernel="rbf", probability=False)
        model.fit(X_train, y_train)
    except Exception:
        # training failed -> log failure but still attempt to record metrics
        reproducibility = 0
        error_rate = 1.0
        deployment_time = time.time() - train_start
        cpu_after = _proc_cpu_seconds()
        cpu_seconds = max(cpu_after - cpu_before, 0.0)
        cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
        mem_mb = _proc_mem_mb()

        log_manual({
            "workflow": "manual",
            "model": "svm",
            "deployment_time": round(deployment_time, 6),
            "error_rate": error_rate,
            "reproducibility": reproducibility,
            "setup_time": round(setup_time, 6),
            "cpu_usage": round(cpu_percent, 4),
            "memory_usage_mb": round(mem_mb, 4),
            "baseline_memory_mb": baseline_memory_mb
        })
        raise

    deployment_time = time.time() - train_start

    cpu_after = _proc_cpu_seconds()
    cpu_seconds = max(cpu_after - cpu_before, 0.0)
    cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0

    mem_mb = _proc_mem_mb()

    # evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1.0 - acc

    # save model
    joblib.dump(model, os.path.join(os.path.dirname(__file__), "svm_model.pkl"))

    # log
    log_manual({
        "workflow": "manual",
        "model": "svm",
        "deployment_time": round(deployment_time, 6),
        "error_rate": round(error_rate, 6),
        "reproducibility": reproducibility,
        "setup_time": round(setup_time, 6),
        "cpu_usage": round(cpu_percent, 4),
        "memory_usage_mb": round(mem_mb, 4),
        "baseline_memory_mb": baseline_memory_mb
    })

    print("SVM training finished. Model saved to manual_pipeline/svm_model.pkl")


if __name__ == "__main__":
    train_svm_model()

# manual_pipeline/train_gb.py
import os
import sys
import time
from time import perf_counter
sys.path.append(os.path.dirname(__file__))

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import psutil

from utils import log_manual

def _process_cpu_seconds():
    times = psutil.Process().cpu_times()
    return times.user + times.system

def _process_memory_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def train_gb_model():
    start_all = time.time()
    setup_start = perf_counter()

    try:
        df = pd.read_csv("manual_pipeline/processed_manual.csv")
    except Exception:
        log_manual({
            "workflow": "manual",
            "model": "gradient_boosting",
            "deployment_time": 0,
            "error_rate": 1,
            "reproducibility": 0,
            "setup_time": 0,
            "cpu_usage": 0,
            "memory_usage_mb": 0
        })
        raise

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    setup_time = perf_counter() - setup_start

    cpu_before = _process_cpu_seconds()
    train_start = time.time()

    error_rate = 0.0
    reproducibility = 1
    try:
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
    except Exception:
        error_rate = 1.0
        reproducibility = 0
        deployment_time = time.time() - train_start
        cpu_after = _process_cpu_seconds()
        cpu_seconds = max(cpu_after - cpu_before, 0.0)
        cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
        mem_mb = _process_memory_mb()
        log_manual({
            "workflow": "manual",
            "model": "gradient_boosting",
            "deployment_time": round(deployment_time, 6),
            "error_rate": error_rate,
            "reproducibility": reproducibility,
            "setup_time": round(setup_time, 6),
            "cpu_usage": round(cpu_percent, 4),
            "memory_usage_mb": round(mem_mb, 4)
        })
        raise

    deployment_time = time.time() - train_start
    cpu_after = _process_cpu_seconds()
    cpu_seconds = max(cpu_after - cpu_before, 0.0)
    cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
    mem_mb = _process_memory_mb()

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1.0 - acc

    joblib.dump(model, "manual_pipeline/gb_model.pkl")

    log_manual({
        "workflow": "manual",
        "model": "gradient_boosting",
        "deployment_time": round(deployment_time, 6),
        "error_rate": round(error_rate, 6),
        "reproducibility": reproducibility,
        "setup_time": round(setup_time, 6),
        "cpu_usage": round(cpu_percent, 4),
        "memory_usage_mb": round(mem_mb, 4)
    })

    print("Gradient Boosting training finished. Model saved to manual_pipeline/gb_model.pkl")

if __name__ == "__main__":
    train_gb_model()

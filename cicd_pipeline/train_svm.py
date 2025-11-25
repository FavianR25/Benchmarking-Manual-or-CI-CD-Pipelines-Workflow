# cicd_pipeline/train_svm.py
import os
import sys
# ensure local folder in path so "from utils import log_cicd" works
sys.path.append(os.path.dirname(__file__))

import time
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import log_cicd, process_cpu_seconds, process_memory_mb

def train_svm():
    print("[CI/CD] Training SVM...")
    start_all = time.time()

    # ===== PREP: load processed dataset produced by cicd_pipeline/preprocess.py =====
    processed_path = os.path.join(os.path.dirname(__file__), "processed_cicd.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    # setup timer (pre-training)
    setup_start = time.perf_counter()
    df = pd.read_csv(processed_path)

    # features/label
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    setup_time = time.perf_counter() - setup_start

    # measure cpu time before training (proc CPU seconds)
    cpu_before = process_cpu_seconds()

    # training
    train_start = time.time()
    error_rate = 0.0
    reproducibility = 1
    try:
        model = SVC(kernel="rbf", probability=False)
        model.fit(X_train, y_train)
    except Exception as e:
        # if training fails, log error_rate = 1 and re-raise to make CI show failure
        error_rate = 1.0
        reproducibility = 0
        # still attempt to log (with error) then raise
        deployment_time = time.time() - train_start
        cpu_after = process_cpu_seconds()
        cpu_seconds = max(cpu_after - cpu_before, 0.0)
        cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0
        mem_mb = process_memory_mb()
        log_cicd({
            "workflow": "cicd",
            "model": "svm",
            "deployment_time": deployment_time,
            "error_rate": error_rate,
            "reproducibility": reproducibility,
            "setup_time": setup_time,
            "cpu_usage": round(cpu_percent, 4),
            "memory_usage_mb": round(mem_mb, 4)
        })
        raise

    deployment_time = time.time() - train_start

    # cpu/time after training
    cpu_after = process_cpu_seconds()
    cpu_seconds = max(cpu_after - cpu_before, 0.0)
    # compute cpu percent during training (= cpu_seconds / wall_seconds * 100)
    cpu_percent = (cpu_seconds / max(deployment_time, 1e-9)) * 100.0

    # memory after training
    mem_mb = process_memory_mb()

    # evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1.0 - acc

    # save model artifact
    model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
    joblib.dump(model, model_path)

    # log metrics
    log_cicd({
        "workflow": "cicd",
        "model": "svm",
        "deployment_time": round(deployment_time, 6),
        "error_rate": round(error_rate, 6),
        "reproducibility": reproducibility,
        "setup_time": round(setup_time, 6),
        "cpu_usage": round(cpu_percent, 4),
        "memory_usage_mb": round(mem_mb, 4)
    })

    print("[CI/CD] SVM training done. Model saved to", model_path)

if __name__ == "__main__":
    train_svm()

# cicd_pipeline/train_svm.py

import time
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import log_cicd


def read_cpu():
    """Read total CPU jiffies from /proc/stat."""
    with open("/proc/stat", "r") as f:
        fields = f.readline().split()[1:]
        fields = list(map(int, fields))
        return sum(fields)


def read_memory_mb():
    """Read RSS memory usage for this process in megabytes."""
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def train_svm():

    # ---------- SETUP TIME ----------
    setup_start = time.time()

    df = pd.read_csv("cicd_pipeline/processed_cicd.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    setup_time = time.time() - setup_start

    # ---------- TRAINING ----------
    cpu_before = read_cpu()
    t0 = time.time()

    try:
        model = SVC(kernel="rbf")
        model.fit(X_train, y_train)
        reproducibility = 1
    except Exception:
        reproducibility = 0

    deployment_time = time.time() - t0
    cpu_after = read_cpu()

    # CPU usage calculation
    cpu_usage = ((cpu_after - cpu_before) / max(deployment_time, 1e-9)) / 100000

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1 - acc

    # Memory use
    memory_mb = read_memory_mb()

    # Save model
    joblib.dump(model, "cicd_pipeline/svm_model.pkl")

    # Logging
    log_cicd({
        "workflow": "cicd",
        "model": "svm",
        "deployment_time": deployment_time,
        "error_rate": error_rate,
        "reproducibility": reproducibility,
        "setup_time": setup_time,
        "cpu_usage": cpu_usage,
        "memory_usage_mb": memory_mb,
    })


if __name__ == "__main__":
    train_svm()

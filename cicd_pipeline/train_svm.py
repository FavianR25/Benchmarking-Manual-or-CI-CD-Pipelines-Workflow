# cicd_pipeline/train_svm.py
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from utils import log_cicd

def read_cpu():
    with open("/proc/stat", "r") as f:
        parts = f.readline().split()[1:]
        parts = list(map(int, parts))
        return sum(parts)

def read_memory_mb():
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0

def train_svm():
    setup_start = time.time()

    df = pd.read_csv("train.csv")

    df = df.drop(columns=["Name", "Ticket", "Cabin", "Embarked"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    setup_time = time.time() - setup_start

    cpu_before = read_cpu()
    start = time.time()

    try:
        model = SVC(kernel="rbf")
        model.fit(X_train, y_train)
        reproducibility = 1
    except Exception:
        reproducibility = 0

    deployment_time = time.time() - start
    cpu_after = read_cpu()

    cpu_usage = ((cpu_after - cpu_before) / max(deployment_time, 1e-9)) / 100000

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    error_rate = 1 - acc

    memory_mb = read_memory_mb()

    joblib.dump(model, "cicd_pipeline/svm_model.pkl")

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

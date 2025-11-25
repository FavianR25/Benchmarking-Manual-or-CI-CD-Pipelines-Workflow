import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from time import time
import psutil
import csv
import os

def log_manual(metrics: dict):
    log_path = "manual_pipeline/log_manual.csv"
    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def train_gb_manual():
    start_time = time()
    process = psutil.Process()

    # Load dataset
    df = pd.read_csv("manual_pipeline/processed_manual.csv")

    # Split data
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    end_time = time()

    metrics = {
        "workflow": "manual",
        "model": "gradient_boosting",
        "deployment_time": end_time - start_time,
        "error_rate": 0,
        "reproducibility": 1,
        "setup_time": start_time - process.create_time(),
        "cpu_usage": process.cpu_percent(),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024)
    }

    log_manual(metrics)
    print("Training GB manual selesai. Log disimpan di manual_pipeline/log_manual.csv")

if __name__ == "__main__":
    train_gb_manual()

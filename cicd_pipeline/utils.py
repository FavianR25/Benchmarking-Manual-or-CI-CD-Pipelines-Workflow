# cicd_pipeline/utils.py
import os
import csv
import time

LOG_PATH = "cicd_pipeline/log_cicd.csv"

HEADER = [
    "workflow",
    "model",
    "deployment_time",
    "error_rate",
    "reproducibility",
    "setup_time",
    "cpu_usage",
    "memory_usage_mb",
    "non_critical_failure"
]

BASELINE_MEMORY_MB = 150  # Kamu bisa adjust ini berdasarkan median awal.

def ensure_log_exists():
    """Ensures the CSV file exists with a header."""
    folder = os.path.dirname(LOG_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def compute_non_critical_failure(metrics):
    """Calculate non-critical failure based on thresholds."""
    error_rate = metrics["error_rate"]
    cpu = metrics["cpu_usage"]
    mem = metrics["memory_usage_mb"]
    reproducibility = metrics["reproducibility"]

    if error_rate > 0.40:
        return 1
    if cpu < 1:
        return 1
    if mem > 2 * BASELINE_MEMORY_MB:
        return 1
    if reproducibility == 0:
        return 1

    return 0


def log_cicd(metrics: dict):
    """Append metrics including non-critical failure."""
    ensure_log_exists()

    metrics["non_critical_failure"] = compute_non_critical_failure(metrics)

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            metrics["workflow"],
            metrics["model"],
            metrics["deployment_time"],
            metrics["error_rate"],
            metrics["reproducibility"],
            metrics["setup_time"],
            metrics["cpu_usage"],
            metrics["memory_usage_mb"],
            metrics["non_critical_failure"]
        ])

    print("Logged CI/CD metrics:", metrics)

# cicd_pipeline/utils.py

import os
import csv
import psutil
import time
import pandas as pd

LOG_PATH = "cicd_pipeline/log_cicd.csv"

HEADER = [
    "workflow",
    "model",
    "deployment_time",
    "error_rate",
    "reproducibility",
    "setup_time",
    "cpu_usage",
    "memory_usage_mb"
]

def ensure_log_exists():
    """Create log file with header if not exists."""
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADER)

def log_cicd(data: dict):
    """Append one row of metrics to log file."""
    ensure_log_exists()

    with open(LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            data["workflow"],
            data["model"],
            data["deployment_time"],
            data["error_rate"],
            data["reproducibility"],
            data["setup_time"],
            data["cpu_usage"],
            data["memory_usage_mb"]
        ])

def measure_cpu():
    # small delay to get a stable reading
    psutil.cpu_percent(interval=0.2)
    return psutil.cpu_percent(interval=0.5)

def measure_memory():
    return psutil.Process().memory_info().rss / (1024 * 1024)

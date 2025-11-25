# manual_pipeline/utils.py
import os
import csv

LOG_PATH = "manual_pipeline/log_manual.csv"

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
    """Create folder + CSV with header if not exists."""
    folder = os.path.dirname(LOG_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def log_manual(metrics: dict):
    """
    Append metrics dict to manual log CSV.
    Expected keys: workflow, model, deployment_time, error_rate, reproducibility,
                   setup_time, cpu_usage, memory_usage_mb
    """
    ensure_log_exists()
    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            metrics.get("workflow"),
            metrics.get("model"),
            metrics.get("deployment_time"),
            metrics.get("error_rate"),
            metrics.get("reproducibility"),
            metrics.get("setup_time"),
            metrics.get("cpu_usage"),
            metrics.get("memory_usage_mb")
        ])

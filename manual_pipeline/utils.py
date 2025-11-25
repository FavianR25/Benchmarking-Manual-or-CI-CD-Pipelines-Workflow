# manual_pipeline/utils.py
import os
import csv

LOG_PATH = os.path.join("manual_pipeline", "log_manual.csv")

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

# default baseline if caller doesn't provide one (MB)
DEFAULT_BASELINE_MEMORY_MB = 150


def ensure_log_exists():
    folder = os.path.dirname(LOG_PATH)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def compute_non_critical_failure(metrics: dict) -> int:
    """
    Rules (same as CI/CD):
      - error_rate > 0.40 -> failure
      - cpu_usage < 1 -> failure
      - memory_usage_mb > 2 * baseline_memory -> failure
      - reproducibility == 0 -> failure
    """
    error_rate = metrics.get("error_rate", 0.0)
    cpu = metrics.get("cpu_usage", 0.0)
    mem = metrics.get("memory_usage_mb", 0.0)
    reproducibility = metrics.get("reproducibility", 1)

    baseline = metrics.get("baseline_memory_mb", DEFAULT_BASELINE_MEMORY_MB)

    if error_rate is None:
        error_rate = 0.0
    try:
        if error_rate > 0.40:
            return 1
    except Exception:
        pass

    try:
        if cpu < 1:
            return 1
    except Exception:
        pass

    try:
        if mem > 2 * baseline:
            return 1
    except Exception:
        pass

    if reproducibility == 0:
        return 1

    return 0


def log_manual(metrics: dict):
    """
    Append metrics row to manual_pipeline/log_manual.csv.
    metrics should at least contain:
      workflow, model, deployment_time, error_rate, reproducibility, setup_time,
      cpu_usage, memory_usage_mb
    Optionally metrics can include 'baseline_memory_mb' used to compute NCF.
    """
    ensure_log_exists()

    # compute non-critical failure (do not persist baseline to CSV)
    ncf = compute_non_critical_failure(metrics)
    row = [
        metrics.get("workflow"),
        metrics.get("model"),
        metrics.get("deployment_time"),
        metrics.get("error_rate"),
        metrics.get("reproducibility"),
        metrics.get("setup_time"),
        metrics.get("cpu_usage"),
        metrics.get("memory_usage_mb"),
        ncf
    ]

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

import csv
import os

LOG_FILE = "cicd_pipeline/log_cicd.csv"

def log_cicd(data: dict):
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "workflow",
            "model",
            "deployment_time",
            "error_rate",
            "reproducibility",
            "setup_time",
            "cpu_usage",
            "memory_usage_mb"
        ])

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

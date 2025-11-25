import csv
import os

def log_manual(metrics):
    log_path = "manual_pipeline/log_manual.csv"

    file_exists = os.path.isfile(log_path)

    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow([
                "workflow",
                "model",
                "deployment_time",
                "error_rate",
                "reproducibility",
                "setup_time",
                "cpu_usage",
                "memory_usage_mb"
            ])

        # Write data row
        writer.writerow([
            metrics["workflow"],
            metrics["model"],
            metrics["deployment_time"],
            metrics["error_rate"],
            metrics["reproducibility"],
            metrics["setup_time"],
            metrics["cpu_usage"],
            metrics["memory_usage_mb"]
        ])

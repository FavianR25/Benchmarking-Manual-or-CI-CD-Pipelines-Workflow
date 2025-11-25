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
    "cpu_usage",           # percent during training (0-100)
    "memory_usage_mb"      # VmRSS after training (MB)
]

def ensure_log_exists():
    if not os.path.exists(LOG_PATH):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def log_cicd(entry: dict):
    """
    entry keys should include:
    workflow, model, deployment_time, error_rate, reproducibility, setup_time, cpu_usage, memory_usage_mb
    """
    ensure_log_exists()
    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            entry.get("workflow"),
            entry.get("model"),
            entry.get("deployment_time"),
            entry.get("error_rate"),
            entry.get("reproducibility"),
            entry.get("setup_time"),
            entry.get("cpu_usage"),
            entry.get("memory_usage_mb")
        ])

# -----------------
# helper functions for /proc reads (Linux)
# -----------------
def _read_proc_stat():
    """Return (utime_ticks, stime_ticks) as ints for current process."""
    with open("/proc/self/stat", "r") as f:
        parts = f.read().split()
        # according to procfs, utime is 14th (index 13), stime is 15th (index 14)
        utime = int(parts[13])
        stime = int(parts[14])
    return utime, stime

def process_cpu_seconds():
    """
    Return process CPU time (user+system) in seconds.
    Uses system clock ticks (/proc/sys/kernel/osrelease CLK_TCK via os.sysconf).
    """
    utime, stime = _read_proc_stat()
    clk_tck = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
    return (utime + stime) / clk_tck

def process_memory_mb():
    """
    Return VmRSS in MB (resident set size). If VmRSS not found return 0.0
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # parts[1] is value, in kB
                    kb = float(parts[1])
                    return kb / 1024.0
    except Exception:
        pass
    return 0.0

# analysis/compute_mi_and_plots.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG: update paths if your CSVs live elsewhere ---
manual_path = "manual_pipeline/log_manual.csv"
cicd_path   = "cicd_pipeline/log_cicd_Metric.csv"
out_dir = "analysis_outputs"
os.makedirs(out_dir, exist_ok=True)

# --- LOAD ---
df_manual = pd.read_csv(manual_path)
df_cicd   = pd.read_csv(cicd_path)

# normalize column names
df_manual.columns = [c.strip() for c in df_manual.columns]
df_cicd.columns = [c.strip() for c in df_cicd.columns]

# add pipeline id if missing
if 'pipeline' not in df_manual.columns:
    df_manual['pipeline'] = 'manual'
if 'pipeline' not in df_cicd.columns:
    df_cicd['pipeline'] = 'cicd'

# combine
df = pd.concat([df_manual, df_cicd], ignore_index=True, sort=False)

# required columns -- change names if yours differ slightly
required = ["deployment_time","setup_time","error_rate","reproducibility",
            "cpu_usage","memory_usage_mb","non_critical_failure","model","pipeline"]
for r in required:
    if r not in df.columns:
        raise SystemExit(f"Missing required column: {r} -- available: {df.columns.tolist()}")

# Ensure numeric
numcols = ["deployment_time","setup_time","error_rate","reproducibility",
           "cpu_usage","memory_usage_mb","non_critical_failure"]
for c in numcols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

# MIN-MAX normalization (combined dataset)
def minmax(s):
    lo = s.min()
    hi = s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)

df['T_a_norm'] = minmax(df['deployment_time'])
df['S_t_norm'] = minmax(df['setup_time'])
df['cpu_norm'] = minmax(df['cpu_usage'])
df['mem_norm'] = minmax(df['memory_usage_mb'])
df['RU_norm'] = (df['cpu_norm'] + df['mem_norm']) / 2.0
df['NCF_norm'] = df['non_critical_failure'].clip(0,1)
df['E_r'] = df['error_rate'].clip(0,1)
df['R_s'] = df['reproducibility'].clip(0,1)

# MI formula as provided by you:
# MI = ((1 - E_r) + R_s + (1 - T_a_norm) + (1 - S_t_norm) + (1 - RU_norm) + (1 - NCF_norm)) / 6
df['MI'] = ((1 - df['E_r']) + df['R_s'] + (1 - df['T_a_norm']) +
            (1 - df['S_t_norm']) + (1 - df['RU_norm']) + (1 - df['NCF_norm'])) / 6.0

# Save outputs
df.to_csv(os.path.join(out_dir, "mi_per_run.csv"), index=False)

summary = df.groupby(['pipeline','model']).agg(
    runs=('MI','count'),
    MI_mean=('MI','mean'),
    MI_std=('MI','std'),
    NCF_rate=('non_critical_failure','mean'),
    error_rate_mean=('error_rate','mean'),
    deployment_time_mean=('deployment_time','mean'),
    cpu_mean=('cpu_usage','mean'),
    mem_mean=('memory_usage_mb','mean')
).reset_index()
summary.to_csv(os.path.join(out_dir, "mi_summary_by_pipeline_model.csv"), index=False)

# --- PLOTS ---
# 1) Mean deployment time by pipeline & model
plt.figure(figsize=(8,5))
pivot = df.groupby(['pipeline','model'])['deployment_time'].mean().unstack()
pivot.plot(kind='bar')
plt.title("Mean Deployment Time by Pipeline & Model")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "deployment_time_by_pipeline_model.png"))
plt.close()

# 2) NCF rate by pipeline & model
plt.figure(figsize=(7,5))
ncf = df.groupby(['pipeline','model'])['non_critical_failure'].mean().unstack()
ncf.plot(kind='bar')
plt.title("NCF Rate by Pipeline & Model")
plt.ylabel("NCF Rate (0..1)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ncf_rate_by_pipeline_model.png"))
plt.close()

# 3) MI distribution by pipeline & model (boxplot)
plt.figure(figsize=(8,5))
df.boxplot(column='MI', by=['pipeline','model'], rot=45)
plt.title("MI Distribution by Pipeline & Model")
plt.suptitle("")
plt.ylabel("MI (0..1)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mi_boxplot_by_pipeline_model.png"))
plt.close()

# 4) Mean MI bar
plt.figure(figsize=(8,5))
mean_mi = summary.pivot(index='pipeline', columns='model', values='MI_mean')
mean_mi.plot(kind='bar')
plt.title("Mean MI by Pipeline & Model")
plt.ylabel("Mean MI (0..1)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mi_mean_by_pipeline_model.png"))
plt.close()

print("Outputs saved to", out_dir)
print("Files: mi_per_run.csv, mi_summary_by_pipeline_model.csv, and PNG plots.")

#!/usr/bin/env python3
"""Green AI metrics — carbon footprint, energy, latency, model size.

Objective:
  Quantify the environmental and deployment costs of each model family
  used in the paper (LR, RF, XGB locally; LGBM measured on the Kaggle
  kernel water-audit-lgbm-green-metrics and imported from its output CSV),
  answering three questions for resource-constrained water agencies:
    (Q1) What is the CO2 cost of training one model from scratch?
    (Q2) What is the inference latency (ms/prediction) on a CPU?
    (Q3) What is the serialized model size (MB)?

Models and protocol:
  Same leakage-free H_rig protocol as §2.2 (80/20 stratified, median
  imputation + standard scaling on train only, SMOTE on train only).
  Fixed config: LR (C=1.0), RF (300 trees, max_depth=12),
  XGBoost (300 trees, lr 0.05, max_depth=5, as in the D1 overfitting
  diagnostics). LGBM is measured on Kaggle (scripts/_lgbm_green_metrics.py)
  with the benchmark config (300 trees, lr 0.05, depth 5).

Carbon model:
  - Energy: wall_time x avg_power, where avg_power is interpolated from
    system-wide CPU utilization (psutil) sampled every 50 ms by a
    background thread during the fit (0% -> P_BASE=15 W, 100% -> P_MAX=65 W).
    Fast fits are repeated (LR x10, RF/XGB x3) and divided down so the
    monitor obtains stable samples; all reported values are per single fit.
  - Carbon intensity: India grid 0.708 kg CO2e/kWh, US grid
    0.475 kg CO2e/kWh, global avg 0.475 kg CO2e/kWh.

Outputs:
  results/tables/green_metrics.csv         — raw per-model metrics
  results/figures/fig10_green_carbon.png  — CO2 emissions bar chart
  results/figures/fig11_green_latency.png — inference latency bar chart
"""
import threading
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# Carbon intensities (kg CO2e per kWh)
CI_INDIA = 0.708
CI_US = 0.475
CI_GLOBAL = 0.475

# Estimated system power during CPU-bound training (W)
P_BASE = 15.0    # W idle baseline
P_MAX = 65.0     # W max during training

df = pd.read_csv(ROOT / "data/processed/ds1/ds1_potability.csv")
X = df[FEATS].values
y = df["Potability"].values

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
imp = SimpleImputer(strategy="median").fit(Xtr)
scl = StandardScaler().fit(imp.transform(Xtr))
Xtr_s = scl.transform(imp.transform(Xtr))
Xte_s = scl.transform(imp.transform(Xte))
Xtr_s, ytr_s = SMOTE(random_state=42).fit_resample(Xtr_s, ytr)

# ---- Ambient-load guard --------------------------------------------------
# System-wide power sampling attributes ALL CPU load to the fit under
# measurement; a busy machine contaminates power, CO2, and latency.
# Abort if ambient load exceeds 15%.
_idle = [psutil.cpu_percent(interval=0.05) for _ in range(20)]
idle_avg = float(np.mean(_idle))
print(f"Ambient CPU baseline (1 s, 20 samples): {idle_avg:.1f}%")
if idle_avg > 15.0:
    raise SystemExit(
        f"ABORT: ambient CPU load {idle_avg:.1f}% exceeds 15% - "
        "power/latency measurements would be contaminated. "
        "Close background processes and re-run.")

results = []

# Configs match the study protocol: LR C=1.0; RF 300 trees, max_depth=12
# (as in §2.2/Table 1); XGB 300 trees, lr 0.05, depth 5 (as in the overfitting
# diagnostics of §3.2). Training uses n_jobs=-1 (as in the benchmark);
# inference latency is timed single-threaded.
models = {
    "LR":  LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "RF":  RandomForestClassifier(n_estimators=300, max_depth=12,
                                 min_samples_leaf=4, random_state=42, n_jobs=-1),
    "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         n_jobs=-1, eval_metric="logloss"),
}

# Timing / sampling constants
N_INFER = 300      # timed predict() calls (one 10-sample batch each)
BATCH = 10         # samples per inference call
# Fast fits are repeated so the 50 ms power monitor gets stable samples;
# all reported metrics are divided down to per-single-fit values.
REPEATS = {"LR": 10, "RF": 3, "XGB": 3}

def timed_fit(model, repeats):
    """Fit `repeats` times; return (wall s per fit, system-wide CPU% samples)."""
    samples, stop = [], threading.Event()

    def monitor():
        psutil.cpu_percent(interval=None)              # prime; first call is 0
        while not stop.is_set():
            samples.append(psutil.cpu_percent(interval=0.05))

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.fit(Xtr_s, ytr_s)
    wall_total = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1.0)
    if len(samples) > 1:
        samples = samples[:-1]                          # drop window straddling fit end
    return wall_total / repeats, samples


for name, model in models.items():
    train_wall, cpu_samples = timed_fit(model, REPEATS[name])
    avg_cpu_pct = float(np.clip(np.mean(cpu_samples), 0.0, 100.0)) if cpu_samples else 0.0
    avg_power = P_BASE + (P_MAX - P_BASE) * (avg_cpu_pct / 100.0)
    train_energy_wh = avg_power * (train_wall / 3600.0)
    train_co2_india = train_energy_wh / 1000 * CI_INDIA
    train_co2_us = train_energy_wh / 1000 * CI_US
    train_co2_global = train_energy_wh / 1000 * CI_GLOBAL

    # Inference latency — timed single-threaded (edge-deployment realistic)
    try:
        model.set_params(n_jobs=1)
    except (ValueError, TypeError):
        pass
    batch = Xte_s[:BATCH]
    for _ in range(20):                                # warm-up
        model.predict(batch)
    t0 = time.perf_counter()
    for _ in range(N_INFER):
        model.predict(batch)
    infer_wall = time.perf_counter() - t0
    latency_ms = (infer_wall / N_INFER) * 1000         # per 10-sample batch
    latency_per_sample_ms = latency_ms / BATCH

    # Serialized model size
    tmp_file = OUT / "_tmp_model.joblib"
    joblib.dump(model, tmp_file)
    size_mb = tmp_file.stat().st_size / (1e6)
    tmp_file.unlink()

    # Parameter count: LR -> learned coefficients; tree ensembles -> total nodes
    if name == "LR":
        n_params = int(model.coef_.size + model.intercept_.size)
    elif name == "RF":
        n_params = int(sum(t.tree_.node_count for t in model.estimators_))
    else:
        n_params = int(model.get_booster().trees_to_dataframe().shape[0])

    f1 = f1_score(yte, model.predict(Xte_s), zero_division=0)
    n_cores = psutil.cpu_count(logical=True)

    results.append({
        "model": name,
        "train_wall_s": round(train_wall, 3),
        "n_repeats": REPEATS[name],
        "avg_cpu_pct": round(avg_cpu_pct, 1),
        "avg_power_w": round(avg_power, 1),
        "train_energy_wh": round(train_energy_wh, 4),
        "train_co2_india_g": round(train_co2_india * 1000, 4),
        "train_co2_us_g": round(train_co2_us * 1000, 4),
        "train_co2_global_g": round(train_co2_global * 1000, 4),
        "infer_latency_batch10_ms": round(latency_ms, 3),
        "infer_latency_per_sample_ms": round(latency_per_sample_ms, 3),
        "model_size_mb": round(size_mb, 3),
        "n_params": n_params,
        "f1": round(f1, 4),
        "n_cores": n_cores,
        "source": "local (measured, this run)",
    })
    print(f"{name} (n={REPEATS[name]}): train={train_wall:.2f}s  cpu={avg_cpu_pct:.0f}%  "
          f"pwr={avg_power:.0f}W  co2_india={train_co2_india*1000:.3f}g  "
          f"latency={latency_per_sample_ms:.3f}ms  size={size_mb:.3f}MB  "
          f"params={n_params}  f1={f1:.4f}")


# LGBM — MEASURED on Kaggle (kernel water-audit-lgbm-green-metrics,
# scripts/_lgbm_green_metrics.py; identical protocol: 3 timed repeats with
# system-wide CPU sampling, single-threaded latency, node-count params).
# Local execution is blocked by an xarray/dask/pandas version conflict
# (Section 4.3), so the kernel output CSV is imported as the artifact.
_lgbm_csv = ROOT / "kaggle" / "green_out" / "lgbm_green_metrics.csv"
if not _lgbm_csv.exists():
    raise SystemExit(f"missing {_lgbm_csv} — run the Kaggle kernel "
                     "water-audit-lgbm-green-metrics and download its output")
_lgbm_row = pd.read_csv(_lgbm_csv).iloc[0].to_dict()
_lgbm_row["model"] = "LGBM (K)"
_lgbm_row["source"] = "Kaggle water-audit-lgbm-green-metrics (measured)"
results.append(_lgbm_row)

rdf = pd.DataFrame(results)
rdf.to_csv(OUT / "tables" / "green_metrics.csv", index=False)

# --- Charts ---
models_order = ["LR", "RF", "XGB", "LGBM (K)"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# Carbon chart (log scale: LR's ~3e-5 g is invisible on a linear axis
# beside RF's 0.0104 g; log keeps every family visible and honestly
# proportioned. Labels get explicit headroom below the title.)
fig, ax = plt.subplots(figsize=(8, 5))
co2 = [rdf.loc[rdf.model == m, "train_co2_india_g"].values[0] for m in models_order]
bars = ax.bar(models_order, co2, color=colors)
ax.set_ylabel("Training CO2 (India grid, 0.708 kg CO2e/kWh, log scale)")
ax.set_title("Carbon cost of training one model on DS1 (3,276 samples)")
ax.set_yscale("log")
ax.set_ylim(min(v for v in co2 if v > 0) / 3, max(co2) * 20)
for bar, val in zip(bars, co2):
    label = "<0.0001" if val < 0.0001 else f"{val:.4f}"
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.35,
            f"{label} g", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig10_green_carbon.png", dpi=300)
plt.close(fig)

# Latency chart (log scale; explicit headroom so the RF label cannot
# collide with the title)
fig, ax = plt.subplots(figsize=(8, 5))
lat = [rdf.loc[rdf.model == m, "infer_latency_per_sample_ms"].values[0]
       for m in models_order]
bars = ax.bar(models_order, lat, color=colors)
ax.set_ylabel("Inference latency (ms / sample, CPU, log scale)")
ax.set_title("Inference speed (lower = better for edge deployment)")
ax.set_yscale("log")
ax.set_ylim(min(lat) / 3, max(lat) * 8)
for bar, val in zip(bars, lat):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.35,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig11_green_latency.png", dpi=300)
plt.close(fig)

print("\n=== Green metrics summary ===")
print(rdf.to_string(index=False))
print("DONE")

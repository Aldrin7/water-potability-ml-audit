# -*- coding: utf-8 -*-
"""Regenerate ONLY the green-metric charts from the existing measured CSV.

Does NOT re-run any measurement: reads results/tables/green_metrics.csv
and redraws fig10/fig11 with the corrected log-scale styling from
green_metrics.py so the published artifacts match the committed code
without perturbing the measured values in Table 8.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results"

rdf = pd.read_csv(OUT / "tables" / "green_metrics.csv")

models_order = ["LR", "RF", "XGB", "LGBM"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# Carbon chart
fig, ax = plt.subplots(figsize=(8, 5))
co2 = [rdf.loc[rdf.model == m, "train_co2_india_g"].values[0] for m in models_order]
bars = ax.bar(models_order, co2, color=colors)
ax.set_ylabel("Training CO2 (India grid, 0.708 kg CO2e/kWh, log scale)")
ax.set_title("Carbon cost of training one model on DS1 (3,276 samples)")
ax.set_yscale("log")
ax.set_ylim(min(v for v in co2 if v > 0) / 3, max(co2) * 20)
for bar, val in zip(bars, co2):
    if val < 0.0001:
        y = ax.get_ylim()[0] * 1.6
    else:
        y = val * 1.35
    label = "<0.0001" if val < 0.0001 else f"{val:.4f}"
    ax.text(bar.get_x() + bar.get_width() / 2, y,
            f"{label} g", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig10_green_carbon.png", dpi=300)
plt.close(fig)

# Latency chart
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

print("charts regenerated from", OUT / "tables" / "green_metrics.csv")
print("DONE")
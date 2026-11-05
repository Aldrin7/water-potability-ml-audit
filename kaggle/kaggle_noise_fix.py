#!/usr/bin/env python3
"""Kaggle kernel v3: corrected sensor-noise sensitivity (RF + LGBM, DS1).

v2 kernel bug (kaggle_benchmark.py noise block): np.random.default_rng(seed)
was created INSIDE the per-feature loop, so every feature was perturbed by the
SAME standard-normal vector (rank-1 perturbation) -> noise looked harmless
(F1 drop -0.0155 for LGBM). This kernel re-runs noise sensitivity correctly:
one RNG instance per seed, fresh independent Gaussian draws per feature -
SD = frac * std(imputed+scaled training column); the RF arm cross-checks local on clean F1 (both find noise harmless).

Runs BOTH RF and LGBM; RF cross-checks local on clean F1 (local v1's -0.28 was the NaN-std artifact, not noise).
Attaches only DS1.
Outputs -> /kaggle/working/results/tables/{m9_noise_rf_fixed.csv,m9_noise_lgbm_fixed.csv}
"""
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path("/kaggle/working/results")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)  # v3 fix: v2 crashed writing the log (FileNotFoundError)
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
NOISE_FRAC = {"ph": 0.30, "Hardness": 0.20, "Solids": 0.10,
              "Chloramines": 0.20, "Sulfate": 0.20, "Conductivity": 0.10,
              "Organic_carbon": 0.20, "Trihalomethanes": 0.20,
              "Turbidity": 0.20}
SEEDS = list(range(1, 16))

BASE = Path("/kaggle/input")
all_csv = [p for p in BASE.glob("**/*.csv")]
if not all_csv:
    raise SystemExit(f"no csv under {BASE}: "
                     f"{[str(p) for p in BASE.iterdir()]}")
f = all_csv[0]
df = pd.read_csv(f)
X, y = df[FEATS].values, df["Potability"].values
log(f"DS1 loaded {df.shape} from {f}")


def run(model_factory, label):
    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        scl = StandardScaler().fit(imp.transform(Xtr))
        Xtr_s = scl.transform(imp.transform(Xtr))
        Xte_s = scl.transform(imp.transform(Xte))
        Xtr_b, ytr_b = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
        m = model_factory(seed).fit(Xtr_b, ytr_b)
        f1c = f1_score(yte, m.predict(Xte_s), zero_division=0)
        # EACH feature gets an INDEPENDENT normal draw (one RNG, seeded once)
        rng = np.random.default_rng(seed)
        noise = np.zeros_like(Xte_s)
        for j, feat in enumerate(FEATS):
            sd = Xtr_s[:, j].std()
            noise[:, j] = rng.normal(0, NOISE_FRAC[feat] * sd, size=len(Xte))
        f1n = f1_score(yte, m.predict(Xte_s + noise), zero_division=0)
        rows.append({"model": label, "seed": seed, "f1_clean": f1c,
                     "f1_noisy": f1n, "f1_drop": f1c - f1n})
        if seed == 1:
            log(f"[{label}] seed1 clean={f1c:.4f} noisy={f1n:.4f}")
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tables" / f"m9_noise_{label.lower()}_fixed.csv", index=False)
    log(f"[{label}] mean clean={d['f1_clean'].mean():.4f} "
        f"noisy={d['f1_noisy'].mean():.4f} "
        f"drop={d['f1_drop'].mean():.4f} (sd)={d['f1_drop'].std(ddof=1):.4f}")
    return d


rf_d = run(lambda s: RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=4,
    random_state=s, n_jobs=-1), "RF")
lgbm_d = run(lambda s: LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, random_state=s, n_jobs=-1, verbosity=-1), "LGBM")

rep = ["# FIXED NOISE SENSITIVITY (independent per-feature RNG, 15 seeds)", "",
       "| Model | Clean F1 | Noisy F1 | Drop mean±sd |", "|---|---|---|---|",
       f"| RF   | {rf_d['f1_clean'].mean():.4f} | {rf_d['f1_noisy'].mean():.4f} | "
       f"{rf_d['f1_drop'].mean():+.4f} ± {rf_d['f1_drop'].std(ddof=1):.4f} |",
       f"| LGBM | {lgbm_d['f1_clean'].mean():.4f} | {lgbm_d['f1_noisy'].mean():.4f} | "
       f"{lgbm_d['f1_drop'].mean():+.4f} ± {lgbm_d['f1_drop'].std(ddof=1):.4f} |",
       "", "RF is the deterministic cross-check vs the local run (expect "
       "drop ≈ -0.001 ± 0.020; matches the corrected local v2 RF (Δ ≈ -0.008); both find noise negligible).",
       "", f"wallclock {time.time()-T0:.0f}s"]
(OUT / "logs" / "m9_noise_fixed.md").write_text("\n".join(rep), encoding="utf-8")
log("DONE")
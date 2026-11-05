#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expanded honest panels — EXACT mirror of src/ds2_benchmark.py protocol.

Spatial CV: GroupKFold(5) by station_id, on the 2018-2019 training YEARS only,
single split (models seed 42) — identical to the paper's Table 3 rows.
Temporal: train 2018-19 -> test 2020 (F1, bal acc, plain accuracy).
Secondary (robustness, NOT for Table 3): 10 permutation seeds x 5 folds.

Panels: non_bis / non_bis+EC / non_bis+EC+gwl  (models LR, RF, XGB, LGBM).
Output -> results/tables/ds2_panels_benchmark.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "ds2" / "ds2_groundwater_panel.csv"

NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
PANELS = {
    "non_bis": NON_BIS,
    "non_bis+EC": NON_BIS + ["EC"],
    "non_bis+EC+gwl": NON_BIS + ["EC", "gwl"],
}
TARGET = "potable_bis"


def models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=4, random_state=seed,
                                     n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=seed, n_jobs=-1, verbosity=-1),
    }


df = pd.read_csv(PANEL)
train, test = df[df.year <= 2019], df[df.year == 2020]
rows = []
t0 = time.time()

for panel, cols in PANELS.items():
    imp = SimpleImputer(strategy="median").fit(train[cols])
    scl = StandardScaler().fit(imp.transform(train[cols]))
    Xtr = scl.transform(imp.transform(train[cols]))
    Xte = scl.transform(imp.transform(test[cols]))
    ytr, yte = train[TARGET].values, test[TARGET].values
    groups = train["station_id"].values

    # ---- exact benchmark protocol: fixed GroupKFold(5), seed 42
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, groups))
    for name, m in models(42).items():
        fs = []
        for a, b in folds:
            mm = m.fit(Xtr[a], ytr[a])
            fs.append(f1_score(ytr[b], mm.predict(Xtr[b]), zero_division=0))
        full = m  # already fitted on full training data? refit:
        full.fit(Xtr, ytr)
        yp = full.predict(Xte)
        rows.append({"panel": panel, "model": name, "protocol": "bench_exact",
                     "spatial_cv_f1_mean": float(np.mean(fs)),
                     "spatial_cv_f1_sd": float(np.std(fs, ddof=1)),
                     "spatial_cv_f1_folds": len(fs),
                     "temporal_2020_f1": f1_score(yte, yp, zero_division=0),
                     "temporal_2020_bal_acc": balanced_accuracy_score(yte, yp),
                     "temporal_2020_accuracy": accuracy_score(yte, yp)})
        print(f"[{panel}] {name:5s} EXACT spatialF1={np.mean(fs):.4f} "
              f"temporalF1={rows[-1]['temporal_2020_f1']:.4f} "
              f"acc={rows[-1]['temporal_2020_accuracy']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- secondary: 10 permutation seeds x 5 station-group folds
    uniq = np.unique(groups)
    per_seed = {n: [] for n in models(42)}
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        order_split = np.array_split(rng.permutation(uniq), 5)
        for name in per_seed:
            f1s = []
            for idx in order_split:
                a = ~np.isin(groups, idx)
                b = np.isin(groups, idx)
                if a.sum() < 50 or b.sum() < 5:
                    continue
                m2 = models(seed)[name].fit(Xtr[a], ytr[a])
                f1s.append(f1_score(ytr[b], m2.predict(Xtr[b]), zero_division=0))
            if f1s:
                per_seed[name].append(np.mean(f1s))
    for name, vals in per_seed.items():
        rows.append({"panel": panel, "model": name, "protocol": "perm10seed",
                     "spatial_cv_f1_mean": float(np.mean(vals)),
                     "spatial_cv_f1_sd": float(np.std(vals, ddof=1)),
                     "spatial_cv_f1_folds": len(vals),
                     "temporal_2020_f1": np.nan,
                     "temporal_2020_bal_acc": np.nan,
                     "temporal_2020_accuracy": np.nan})
        print(f"[{panel}] {name:5s} PERM10 spatialF1={np.mean(vals):.4f} "
              f"± {np.std(vals, ddof=1):.4f} [{time.time()-t0:.0f}s]", flush=True)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "tables" / "ds2_panels_benchmark.csv",
           index=False)
print("DONE")

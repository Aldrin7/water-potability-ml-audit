#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DS3 deployment LGBM re-run (local, lightgbm 4.6.0) — mirrors
src/ds3_deployment.py exactly (GroupShuffleSplit by State, 10 iterations,
SMOTE per split, train-only preprocessing) for the three cost tiers.
Output -> results/tables/ds3_lgbm_local.csv"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TARGET = "classB_compliant"
ARMS = {
    "full_params_circular": ["DO_Min", "BOD_Max", "Fecal_Max",
                             "TotalCol_Max", "NOx_Max", "pH_Max",
                             "Temp_Max", "Cond_Max"],
    "do_probe_T1": ["DO_Min"],
    "cheap_sensor_T0": ["Temp_Max", "pH_Max", "Cond_Max"],
}

def lgbm():
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1, verbosity=-1)

df = pd.read_csv(ROOT / "data" / "processed" / "ds3" / "ds3_cpcb_surface.csv")
print(f"DS3 rows={len(df)} compliant_rate={df[TARGET].mean():.3f}", flush=True)
rows = []
t0 = time.time()
for arm, feats in ARMS.items():
    X, y, groups = df[feats].values, df[TARGET].values, df["State"].values
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    f1s, accs, recs = [], [], []
    for i, (a, b) in enumerate(gss.split(X, y, groups)):
        imp = SimpleImputer(strategy="median").fit(X[a])
        scl = StandardScaler().fit(imp.transform(X[a]))
        Xa = scl.transform(imp.transform(X[a]))
        Xb = scl.transform(imp.transform(X[b]))
        ya, yb = y[a], y[b]
        Xa, ya = SMOTE(random_state=i).fit_resample(Xa, ya)
        yp = lgbm().fit(Xa, ya).predict(Xb)
        f1s.append(f1_score(yb, yp, zero_division=0))
        accs.append(balanced_accuracy_score(yb, yp))
        recs.append(recall_score(yb, yp, pos_label=0, zero_division=0))
    rows.append({"tier": arm, "model": "LGBM",
                 "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1)),
                 "bal_acc": float(np.mean(accs)),
                 "recall_noncompliant": float(np.mean(recs))})
    print(f"{arm}: F1={np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f} "
          f"balAcc={np.mean(accs):.3f} rec0={np.mean(recs):.3f} [{time.time()-t0:.0f}s]",
          flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "ds3_lgbm_local.csv",
                          index=False)
print("DONE")

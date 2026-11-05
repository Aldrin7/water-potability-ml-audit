#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LGBM overfitting-diagnostics re-run (Table 2 row): train vs test F1 over 30
seeds under the H_rig leakage-free holdout, all local (lightgbm 4.6.0).
Output -> results/tables/overfitting_lgbm_local.csv"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ds1 = pd.read_csv(ROOT / "data" / "processed" / "ds1" / "ds1_potability.csv")
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
X, y = ds1[FEATS].values, ds1["Potability"].values
rows = []
t0 = time.time()
for seed in range(1, 31):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20,
                                          random_state=seed, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    Xtr_b, ytr_b = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                       subsample=0.8, colsample_bytree=0.8, random_state=seed,
                       n_jobs=-1, verbosity=-1).fit(Xtr_b, ytr_b)
    rows.append({"seed": seed,
                 "train_f1": f1_score(ytr, m.predict(Xtr_s), zero_division=0),
                 "test_f1": f1_score(yte, m.predict(Xte_s), zero_division=0),
                 "test_acc": accuracy_score(yte, m.predict(Xte_s))})
    print(f"seed {seed} done [{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "tables" / "overfitting_lgbm_local.csv",
          index=False)
print(df.agg(["mean", "std"]).round(4).to_string())
EOF = None

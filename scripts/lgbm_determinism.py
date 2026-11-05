#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Determinism check: LGBM DS2 non_bis spatial CV, n_jobs=1 vs n_jobs=-1,
repeated. Explains the 0.573 (Kaggle) vs 0.555 (local) spatial-F1 delta."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "ds2" / "ds2_groundwater_panel.csv")
NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
train = df[df.year <= 2019]
imp = SimpleImputer(strategy="median").fit(train[NON_BIS])
Xs = StandardScaler().fit_transform(imp.transform(train[NON_BIS]))
y = train["potable_bis"].values
g = train["station_id"].values
folds = list(GroupKFold(5).split(Xs, y, g))


def run(n_jobs, seed):
    fs = []
    for a, b in folds:
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=seed, n_jobs=n_jobs,
                           verbosity=-1).fit(Xs[a], y[a])
        fs.append(f1_score(y[b], m.predict(Xs[b]), zero_division=0))
    return float(np.mean(fs)), [round(v, 3) for v in fs]


print("n_jobs=1 seed42 :", run(1, 42))
print("n_jobs=-1 seed42:", run(-1, 42))
print("n_jobs=-1 seed42 again:", run(-1, 42))
print("per-fold Kaggle-era: 0.573 ± 0.063")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute + persist PR/DCA curves (same protocol as decision_curve_analysis.py) and
report the deployment-relevant DCA statistics:
  - net benefit at the deployed safety-first threshold (mean per model)
  - threshold interval where the model's NB >= flag-all NB (and NB >= 0)
Outputs -> results/tables/pr_dca_curves.csv (+ summary update)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUTT = ROOT / "results" / "tables"
ds1 = pd.read_csv(ROOT / "data" / "processed" / "ds1" / "ds1_potability.csv")
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
X, y = ds1[FEATS].values, ds1["Potability"].values

MODELS = {
    "LR": lambda s: LogisticRegression(max_iter=2000, random_state=s),
    "RF": lambda s: RandomForestC(s),
}

def RandomForestC(s):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, max_depth=12,
                                  min_samples_leaf=4, random_state=s, n_jobs=-1)

MODELS = {
    "LR": lambda s: LogisticRegression(max_iter=2000, random_state=s),
    "RF": RandomForestC,
    "XGB": lambda s: XGBClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8,
                                   colsample_bytree=0.8, random_state=s,
                                   n_jobs=-1, eval_metric="logloss"),
    "LGBM": lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                     max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, random_state=s,
                                     n_jobs=-1, verbosity=-1),
}
# also fetch each model's deployed policy threshold from the calibration run
cal = pd.read_csv(OUTT / "calibration_threshold_local.csv")
thr_deploy = cal.groupby("model").thr_policy.mean().to_dict()

probs = {name: [] for name in MODELS}
ys = []
for seed in range(1, 31):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20,
                                          random_state=seed, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    ys.append(yte)
    for name, ctor in MODELS.items():
        probs[name].append(ctor(seed).fit(Xtr_s, ytr).predict_proba(Xte_s)[:, 1])
    print("seed", seed, flush=True)
y_pool = np.concatenate(ys)

ts = np.round(np.arange(0.02, 0.991, 0.01), 2)
prev = (1 - y_pool).mean()
nb_all = prev - (1 - prev) * ts / (1 - ts)
rows = []
extra = []
for name in MODELS:
    p = np.concatenate(probs[name])
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_pool, p)
    for i in range(len(prec)):
        rows.append({"model": name, "kind": "pr", "threshold": np.nan,
                     "x": rec[i], "y": prec[i]})
    flag = p < ts[:, None]
    y_unsafe = 1 - y_pool
    tp = (flag & (y_unsafe == 1)).sum(axis=1)
    fp = (flag & (y_unsafe == 0)).sum(axis=1)
    nb = (tp - fp * ts / (1 - ts)) / len(y_pool)
    for j, t in enumerate(ts):
        rows.append({"model": name, "kind": "dca", "threshold": t,
                     "x": t, "y": nb[j]})
    for j, t in enumerate(ts):
        rows.append({"model": "flag-all", "kind": "dca", "threshold": t,
                     "x": t, "y": nb_all[j]})
    # deployment stats
    tdep = thr_deploy[name]
    flagd = p < tdep; tpd = (flagd & (y_unsafe == 1)).sum()
    fpd = (flagd & (y_unsafe == 0)).sum()
    nbdep = (tpd - fpd * tdep / (1 - tdep)) / len(y_pool)
    ok = ts[np.where(nb >= nb_all)[0]]
    ok = ok[(ok > 0.05) & (ok < 0.9)]
    nz = ts[(nb > 0) & (ts < 0.9)]
    extra.append({"model": name, "nb_at_deployed_thr": round(float(nbdep), 4),
                  "deployed_thr": round(float(tdep), 3),
                  "nb_beats_flagall_interval": (f"{ok.min():.2f}-{ok.max():.2f}"
                                                if len(ok) > 5 else "none"),
                  "nb_positive_interval": (f"{nz.min():.2f}-{nz.max():.2f}"
                                           if len(nz) else "none"),
                  "auprc": round(float(average_precision_score(y_pool, p)), 4)})
pd.DataFrame(rows).to_csv(OUTT / "pr_dca_curves.csv", index=False)
ex = pd.DataFrame(extra)
ex.to_csv(OUTT / "pr_dca_summary_v2.csv", index=False)
print(ex.to_string(index=False))

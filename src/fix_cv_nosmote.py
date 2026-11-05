#!/usr/bin/env python3
"""Fix CV_nosmote: full-data impute+scale (leaky prep), NO SMOTE anywhere.

Only this arm was bugged (previous code applied SMOTE unconditionally on
the leaky path). Runs 10 seeds x (5-fold x 2 repeats), RF and GB, then
replaces the CV_nosmote rows in protocol_audit.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
T = ROOT / "results" / "tables"
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
SEEDS = list(range(1, 11))


def make_model(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=12,
                                      min_samples_leaf=4, random_state=seed,
                                      n_jobs=-1)
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, subsample=0.8,
                                      random_state=seed)


def run(df, model_name):
    """Leaky prep (impute+scale on full data), NO SMOTE, CV model-only."""
    rows = []
    X, y = df[FEATS].values, df["Potability"].values
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    for seed in SEEDS:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                       random_state=seed)
        for tr, te in rskf.split(X, y):
            m = make_model(model_name, seed).fit(X[tr], y[tr])
            yp = m.predict(X[te])
            rows.append({"seed": seed,
                         "accuracy": accuracy_score(y[te], yp),
                         "f1": f1_score(y[te], yp, zero_division=0)})
    d = pd.DataFrame(rows)
    return {"arm": "CV_nosmote", "model": model_name, "n_fits": len(SEEDS) * 10,
            "acc_mean": d.accuracy.mean(), "acc_sd": d.accuracy.std(ddof=1),
            "f1_mean": d.f1.mean(), "f1_sd": d.f1.std(ddof=1)}


def main():
    df = pd.read_csv(PROC / "ds1" / "ds1_potability.csv")
    out = []
    for name in ("RF", "GB"):
        t0 = time.time()
        rec = run(df, name)
        rec_n = {k: round(v, 4) for k, v in rec.items()} | {"n_fits": rec["n_fits"]}
        out.append(rec)
        print(f"CV_nosmote {name} acc={rec['acc_mean']:.4f}+-{rec['acc_sd']:.4f} "
              f"f1={rec['f1_mean']:.4f}+-{rec['f1_sd']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    full = pd.read_csv(T / "protocol_audit.csv")
    full = full[full.arm != "CV_nosmote"]
    full = pd.concat([full, pd.DataFrame(out)], ignore_index=True)
    full.to_csv(T / "protocol_audit.csv", index=False)
    print("DONE — merged corrected CV_nosmote into protocol_audit.csv")


if __name__ == "__main__":
    main()
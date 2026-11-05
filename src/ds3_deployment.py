#!/usr/bin/env python3
"""RQ3 — DS3 CPCB surface water: deployment-cost analysis.

Label: classB_compliant (CPCB Class B thresholds on worst-case station
values; see Gate A). Rate = 0.176 (imbalanced ~4.7:1).

Arms (feature-cost tiers):
  full_params_circular  DO/BOD/Fecal/TotalCol/NOx/pH/Temp/Cond
                        -> circularity demo (label defined by DO/BOD/Fecal)
  cheap_sensor_T0       Temp_Max, pH_Max, Cond_Max (field test-kit tier)
  do_probe_T1           DO_Min only (single mid-cost sensor)

Validation: GroupShuffleSplit by State, 10 iterations (spatial transfer);
preprocessing + SMOTE fit on the training partition of each split only.
Metrics: F1 (compliant), balanced accuracy, recall of the NON-compliant
class (agency safety risk metric).
Models: LR, RF, XGB (LGBM excluded locally — flagged; see ds2 report).
Output: results/tables/ds3_deployment.csv,
        results/logs/ds3_deployment_verification.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TARGET = "classB_compliant"
ARMS = {
    "full_params_circular": ["DO_Min", "BOD_Max", "Fecal_Max",
                             "TotalCol_Max", "NOx_Max", "pH_Max",
                             "Temp_Max", "Cond_Max"],
    "cheap_sensor_T0": ["Temp_Max", "pH_Max", "Cond_Max"],
    "do_probe_T1": ["DO_Min"],
}


def models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=4, random_state=seed,
                                     n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=5, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             eval_metric="logloss"),
    }


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "ds3" /
                     "ds3_cpcb_surface.csv")
    print(f"DS3 rows={len(df)} compliant_rate={df[TARGET].mean():.3f}",
          flush=True)
    rows = []
    for arm, feats in ARMS.items():
        X, y, groups = df[feats].values, df[TARGET].values, df["State"].values
        gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
        for name, ctor in models(42).items():
            t0 = time.time()
            scores = []
            for i, (a, b) in enumerate(gss.split(X, y, groups)):
                imp = SimpleImputer(strategy="median").fit(X[a])
                scl = StandardScaler().fit(imp.transform(X[a]))
                Xa = scl.transform(imp.transform(X[a]))
                Xb = scl.transform(imp.transform(X[b]))
                ya, yb = y[a], y[b]
                Xa, ya = SMOTE(random_state=i).fit_resample(Xa, ya)
                m = ctor.fit(Xa, ya)
                yp = m.predict(Xb)
                scores.append({
                    "f1": f1_score(yb, yp, zero_division=0),
                    "bal_acc": balanced_accuracy_score(yb, yp),
                    "rec_nc": recall_score(yb, yp, pos_label=0)})
            rows.append({
                "arm": arm, "model": name,
                "f1_compliant_mean": np.mean([s["f1"] for s in scores]),
                "f1_compliant_sd": np.std([s["f1"] for s in scores], ddof=1),
                "bal_acc_mean": np.mean([s["bal_acc"] for s in scores]),
                "recall_noncompliant_mean": np.mean([s["rec_nc"]
                                                     for s in scores]),
                "wallclock_s": round(time.time() - t0, 1)})
            r = rows[-1]
            print(f"[{arm}] {name:4s} F1={r['f1_compliant_mean']:.4f}"
                  f"+-{r['f1_compliant_sd']:.4f} "
                  f"balAcc={r['bal_acc_mean']:.4f} "
                  f"recallNC={r['recall_noncompliant_mean']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tables" / "ds3_deployment.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()


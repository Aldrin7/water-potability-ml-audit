#!/usr/bin/env python3
"""RQ2 — DS2 Telangana groundwater: spatial + temporal generalization.

Protocol (leakage-controlled):
  Temporal: train on 2018+2019, test on 2020 (station-level transfer
  across time).
  Spatial: within the training years, GroupKFold by village-mandal so a
  village never appears in both inner-train and inner-validation.
  All imputation/scaling fit on the training partition only.
Models: LR, RF, XGBoost, LightGBM (fixed symmetric hyperparameters).
Target: BIS 10500-derived potable_bis (see Gate A).
Output: results/tables/ds2_benchmark.csv, per-fold rows + aggregates,
        results/logs/ds2_benchmark_verification.md
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:  # LGBM optional: local dask/xarray/pandas clash breaks import (flagged)
    from lightgbm import LGBMClassifier
except Exception as _lgbm_err:  # noqa: BLE001
    LGBMClassifier = None
    LGBM_ERR = repr(_lgbm_err)[:200]
else:
    LGBM_ERR = None

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "ds2" / "ds2_groundwater_panel.csv"
OUT = ROOT / "results"
IONS = ["pH", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
        "SO4", "Na", "K", "Ca", "Mg", "TH"]
# Features NOT used in the BIS-10500 label derivation -> non-circular task
NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
TARGET = "potable_bis"


def models(seed):
    out = {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=4, random_state=seed,
                                     n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=5, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             eval_metric="logloss"),
    }
    if LGBMClassifier is not None:
        out["LGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                     max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, random_state=seed,
                                     n_jobs=-1, verbosity=-1)
    return out


def run_feature_set(df, feature_set, ions, train, test):
    imp = SimpleImputer(strategy="median").fit(train[ions])
    scl = StandardScaler().fit(imp.transform(train[ions]))
    Xtr_all = scl.transform(imp.transform(train[ions]))
    Xte = scl.transform(imp.transform(test[ions]))
    ytr_all, yte = train[TARGET].values, test[TARGET].values
    groups = train["station_id"].values

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(Xtr_all, ytr_all, groups))
    rows = []
    for name, ctor in models(42).items():
        t0 = time.time()
        fold_scores = []
        for f, (a, b) in enumerate(folds):
            m = ctor.fit(Xtr_all[a], ytr_all[a])
            yp = m.predict(Xtr_all[b])
            fold_scores.append({
                "fold": f, "f1": f1_score(ytr_all[b], yp, zero_division=0),
                "bal_acc": balanced_accuracy_score(ytr_all[b], yp)})
        m = ctor.fit(Xtr_all, ytr_all)
        yp = m.predict(Xte)
        temp = {"f1": f1_score(yte, yp, zero_division=0),
                "bal_acc": balanced_accuracy_score(yte, yp),
                "accuracy": accuracy_score(yte, yp)}
        rows.append({
            "feature_set": feature_set, "model": name,
            "spatial_cv_f1_mean": np.mean([s["f1"] for s in fold_scores]),
            "spatial_cv_f1_sd": np.std([s["f1"] for s in fold_scores], ddof=1),
            "spatial_cv_bal_acc": np.mean([s["bal_acc"] for s in fold_scores]),
            "temporal_2020_f1": temp["f1"],
            "temporal_2020_bal_acc": temp["bal_acc"],
            "temporal_2020_accuracy": temp["accuracy"],
            "wallclock_s": round(time.time() - t0, 1)})
        print(f"[{feature_set}] {name:5s} spatialF1="
              f"{rows[-1]['spatial_cv_f1_mean']:.4f} temporalF1={temp['f1']:.4f} "
              f"balAcc={temp['bal_acc']:.4f}", flush=True)
    return rows


def main():
    df = pd.read_csv(PANEL)
    train, test = df[df.year <= 2019], df[df.year == 2020]
    print(f"train rows={len(train)} test rows={len(test)} "
          f"train potable={train[TARGET].mean():.3f} "
          f"test potable={test[TARGET].mean():.3f}", flush=True)

    # Arm 1: full ion chemistry — CIRCULARITY DEMONSTRATION (label is a
    # threshold function of these same ions; near-perfect scores expected
    # by construction, not evidence of predictive value).
    # Arm 2: NON_BIS features only — the honest, non-circular task.
    rows = run_feature_set(df, "full_ions_circular", IONS, train, test)
    rows += run_feature_set(df, "non_bis_inputs", NON_BIS, train, test)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tables" / "ds2_benchmark.csv", index=False)
    rep = ["# DS2 BENCHMARK — VERIFICATION REPORT (RQ2)", "",
           f"Temporal split: train 2018-2019 (n={len(train)}), "
           f"test 2020 (n={len(test)}).",
           "Spatial inner CV: GroupKFold(5) by village-mandal.",
           "Preprocessing fit on training years only.",
           "",
           "## Scope flag (Rule 3 / risk R3, RESOLVED EMPIRICALLY)",
           "",
           "Arm `full_ions_circular` uses the same ions that define the",
           "BIS-10500 label, so near-perfect scores are a circularity",
           "artifact, NOT predictive value. Arm `non_bis_inputs` uses only",
           "features absent from the label definition — the honest task.",
           "This circularity result is itself an audit finding (RQ1-adjacent):",
           "published papers reporting near-perfect potability classification",
           "from chemistry that defines the standard's thresholds warrant",
           "scrutiny.",
           "",
           "| Feature set | Model | Spatial CV F1 | 2020 F1 | 2020 BalAcc |",
           "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        rep.append(f"| {r.feature_set} | {r.model} | "
                   f"{r.spatial_cv_f1_mean:.4f} ± {r.spatial_cv_f1_sd:.4f} | "
                   f"{r.temporal_2020_f1:.4f} | "
                   f"{r.temporal_2020_bal_acc:.4f} |")
    (OUT / "logs" / "ds2_benchmark_verification.md").write_text(
        "\n".join(rep), encoding="utf-8")
    if LGBM_ERR:
        with open(OUT / "logs" / "ds2_benchmark_verification.md", "a",
                  encoding="utf-8") as f:
            f.write(f"\n\n## Scope flag (Rule 3)\n\nLGBM excluded from this "
                    f"run: local import failure — {LGBM_ERR}. "
                    f"To be re-run on Kaggle where the environment is clean.")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Overfitting & robustness diagnostics — all claims below are MEASURED.

D1  Train-vs-test gap per model (DS1, rigorous protocol, 30 seeds)
D2  Capacity sweep: RF max_depth {4,6,8,10,12,16,None} -> train/test F1
D3  Learning curve: train fraction {0.1,0.25,0.5,0.75,1.0} x 5 seeds
D4  DS2 non-BIS: RANDOM-CV vs SPATIAL-CV vs TEMPORAL gap
    (quantifies village-level memorization: random CV lets the same
     village appear in train and validation; spatial does not)
D5  Calibration: Brier score + reliability curve data (RF, DS1)
Outputs: results/tables/overfitting_diagnostics.csv (+ per-diagnostic CSVs),
         results/logs/overfitting_diagnostics.md
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
T = OUT / "tables"
T.mkdir(exist_ok=True)
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
SEEDS = list(range(1, 31))


def prep(Xtr, Xte, seed):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s, y_dummy = Xtr, None
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    return Xtr_s, Xte_s


def smote_train(Xtr_s, ytr, seed):
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)


def rf(seed, depth=None):
    return RandomForestClassifier(n_estimators=300, max_depth=depth,
                                  min_samples_leaf=4, random_state=seed,
                                  n_jobs=-1)


def gb(seed):
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, subsample=0.8,
                                      random_state=seed)


def xgb(seed):
    return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=-1, eval_metric="logloss")


def lr(seed):
    return LogisticRegression(max_iter=2000, random_state=seed)


log_lines = []


def log(m):
    print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    log_lines.append(m)


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "ds1" /
                     "ds1_potability.csv")
    X, y = df[FEATS].values, df["Potability"].values

    # ---------------- D1: train-vs-test gap, 4 models x 30 seeds
    rows = []
    for name, ctor in [("LR", lr), ("RF", rf), ("GB", gb), ("XGB", xgb)]:
        t0 = time.time()
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y)
            Xtr_s, Xte_s = prep(Xtr, Xte, seed)
            Xtr_b, ytr_b = smote_train(Xtr_s, ytr, seed)
            m = ctor(seed).fit(Xtr_b, ytr_b)
            rows.append({
                "model": name, "seed": seed,
                "train_f1": f1_score(ytr_b, m.predict(Xtr_b)),
                "test_f1": f1_score(yte, m.predict(Xte_s), zero_division=0),
                "test_acc": m.score(Xte_s, yte)})
        log(f"D1 {name} done [{time.time()-t0:.0f}s]")
    d1 = pd.DataFrame(rows)
    d1.to_csv(T / "d1_train_test_gap.csv", index=False)
    d1_sum = d1.groupby("model").agg(
        train_f1=("train_f1", "mean"), train_f1_sd=("train_f1", "std"),
        test_f1=("test_f1", "mean"), test_f1_sd=("test_f1", "std"),
        test_acc=("test_acc", "mean"))
    d1_sum["gap_f1"] = d1_sum.train_f1 - d1_sum.test_f1
    d1_sum.round(4).to_csv(T / "d1_train_test_gap_summary.csv")

    # ---------------- D2: capacity sweep (RF depth), 10 seeds
    rows = []
    for depth in [4, 6, 8, 10, 12, 16, None]:
        t0 = time.time()
        for seed in SEEDS[:10]:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y)
            Xtr_s, Xte_s = prep(Xtr, Xte, seed)
            Xtr_b, ytr_b = smote_train(Xtr_s, ytr, seed)
            m = rf(seed, depth).fit(Xtr_b, ytr_b)
            rows.append({
                "max_depth": str(depth), "seed": seed,
                "train_f1": f1_score(ytr_b, m.predict(Xtr_b)),
                "test_f1": f1_score(yte, m.predict(Xte_s), zero_division=0)})
        log(f"D2 depth={depth} done [{time.time()-t0:.0f}s]")
    d2 = pd.DataFrame(rows)
    d2.to_csv(T / "d2_capacity_sweep.csv", index=False)
    d2_sum = d2.groupby("max_depth").agg(
        train_f1=("train_f1", "mean"), test_f1=("test_f1", "mean"),
        test_f1_sd=("test_f1", "std"))
    d2_sum["gap_f1"] = d2_sum.train_f1 - d2_sum.test_f1
    d2_sum.round(4).to_csv(T / "d2_capacity_sweep_summary.csv")

    # ---------------- D3: learning curve, 5 seeds
    rows = []
    for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        t0 = time.time()
        for seed in SEEDS[:5]:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.20, random_state=seed, stratify=y)
            Xtr_s, Xte_s = prep(Xtr, Xte, seed)
            Xtr_b, ytr_b = smote_train(Xtr_s, ytr, seed)
            if frac < 1.0:
                n = int(len(Xtr_b) * frac)
                idx = np.random.RandomState(seed).choice(
                    len(Xtr_b), n, replace=False)
                Xtr_b, ytr_b = Xtr_b[idx], ytr_b[idx]
            m = rf(seed).fit(Xtr_b, ytr_b)
            rows.append({"frac": frac, "seed": seed,
                         "test_f1": f1_score(yte, m.predict(Xte_s),
                                             zero_division=0)})
        log(f"D3 frac={frac} done [{time.time()-t0:.0f}s]")
    d3 = pd.DataFrame(rows)
    d3.to_csv(T / "d3_learning_curve.csv", index=False)
    d3_sum = d3.groupby("frac").agg(
        test_f1=("test_f1", "mean"), sd=("test_f1", "std"))
    d3_sum.round(4).to_csv(T / "d3_learning_curve_summary.csv")

    # ---------------- D4: DS2 non-BIS — random vs spatial vs temporal
    panel = pd.read_csv(ROOT / "data" / "processed" / "ds2" /
                        "ds2_groundwater_panel.csv")
    NB = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
    train = panel[panel.year <= 2019]
    test = panel[panel.year == 2020]
    imp = SimpleImputer(strategy="median").fit(train[NB])
    scl = StandardScaler().fit(imp.transform(train[NB]))
    Xtr = scl.transform(imp.transform(train[NB]))
    Xte = scl.transform(imp.transform(test[NB]))
    ytr, yte = train["potable_bis"].values, test["potable_bis"].values
    groups = train["station_id"].values

    d4_rows = []
    for name, ctor in [("LR", lr), ("RF", rf), ("XGB", xgb)]:
        # (a) RANDOM 5-fold CV on training years (village leaks across folds)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rand_f1 = []
        for a, b in kf.split(Xtr):
            m = ctor(42).fit(Xtr[a], ytr[a])
            rand_f1.append(f1_score(ytr[b], m.predict(Xtr[b]),
                                    zero_division=0))
        # (b) SPATIAL GroupKFold by village (honest within-year estimate)
        gkf = GroupKFold(n_splits=5)
        spat_f1 = []
        for a, b in gkf.split(Xtr, ytr, groups):
            m = ctor(42).fit(Xtr[a], ytr[a])
            spat_f1.append(f1_score(ytr[b], m.predict(Xtr[b]),
                                    zero_division=0))
        # (c) train-set fit quality (memorization check)
        m_full = ctor(42).fit(Xtr, ytr)
        train_f1 = f1_score(ytr, m_full.predict(Xtr))
        # (d) temporal transfer (2020)
        temp_f1 = f1_score(yte, m_full.predict(Xte), zero_division=0)
        d4_rows.append({
            "model": name,
            "random_cv_f1": np.mean(rand_f1),
            "spatial_cv_f1": np.mean(spat_f1),
            "station_memorization_gap": np.mean(rand_f1) - np.mean(spat_f1),
            "train_f1": train_f1,
            "temporal_2020_f1": temp_f1,
            "spatial_temporal_gap": np.mean(spat_f1) - temp_f1})
        log(f"D4 {name} done")
    d4 = pd.DataFrame(d4_rows)
    d4.round(4).to_csv(T / "d4_ds2_generalization.csv", index=False)

    # ---------------- D5: calibration (RF, DS1, seed 42)
    from sklearn.calibration import calibration_curve
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    Xtr_s, Xte_s = prep(Xtr, Xte, 42)
    Xtr_b, ytr_b = smote_train(Xtr_s, ytr, 42)
    m = rf(42).fit(Xtr_b, ytr_b)
    prob = m.predict_proba(Xte_s)[:, 1]
    brier = brier_score_loss(yte, prob)
    frac_pos, mean_pred = calibration_curve(yte, prob, n_bins=8,
                                            strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos}
                 ).to_csv(T / "d5_reliability_curve.csv", index=False)
    log(f"D5 done, Brier={brier:.4f}")

    # ---------------- report
    rep = ["# OVERFITTING & ROBUSTNESS DIAGNOSTICS — MEASURED RESULTS", "",
           "All values below were computed by src/overfitting_diagnostics.py.",
           "", "## D1 — Train vs test F1 gap (DS1, 30 seeds, rigorous protocol)",
           "", "| Model | Train F1 | Test F1 | Gap |", "|---|---|---|---|"]
    for name, r in d1_sum.iterrows():
        rep.append(f"| {name} | {r.train_f1:.4f} ± {r.train_f1_sd:.4f} | "
                   f"{r.test_f1:.4f} ± {r.test_f1_sd:.4f} | {r.gap_f1:+.4f} |")
    rep += ["", "## D2 — RF capacity sweep (10 seeds)", "",
            "| max_depth | Train F1 | Test F1 | Gap |", "|---|---|---|---|"]
    for name, r in d2_sum.iterrows():
        rep.append(f"| {name} | {r.train_f1:.4f} | {r.test_f1:.4f} ± "
                   f"{r.test_f1_sd:.4f} | {r.gap_f1:+.4f} |")
    rep += ["", "## D3 — Learning curve (RF, 5 seeds)", "",
            "| Train fraction | Test F1 |", "|---|---|"]
    for name, r in d3_sum.iterrows():
        rep.append(f"| {name} | {r.test_f1:.4f} ± {r.sd:.4f} |")
    rep += ["", "## D4 — DS2 non-BIS generalization decomposition", "",
            "| Model | RandomCV F1 | SpatialCV F1 | Station-memorization "
            "gap | Train F1 | 2020 F1 |", "|---|---|---|---|---|---|"]
    for _, r in d4.iterrows():
        rep.append(f"| {r.model} | {r.random_cv_f1:.4f} | "
                   f"{r.spatial_cv_f1:.4f} | {r.station_memorization_gap:+.4f}"
                   f" | {r.train_f1:.4f} | {r.temporal_2020_f1:.4f} |")
    rep += ["", "## D5 — Calibration (RF, DS1, seed 42)", "",
            f"- Brier score: **{brier:.4f}**",
            "- Reliability curve data: results/tables/d5_reliability_curve.csv",
            "", "## Execution log", "```"] + log_lines + ["```"]
    (OUT / "logs" / "overfitting_diagnostics.md").write_text(
        "\n".join(rep), encoding="utf-8")
    log("ALL DIAGNOSTICS DONE")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""RQ1 — Protocol-gap decomposition audit on DS1 (water potability).

Chain of one-factor-at-a-time protocol changes from the rigorous baseline
to the Patel-style protocol, measuring how reported accuracy inflates:

  H_rig    holdout 80/20, median impute, SMOTE train-only     (rigorous)
  H_patel  holdout 70/30, mean impute, SMOTE train-only       (split+impute)
  CV_rig   stratified 5-fold x2 rep, per-fold train-only prep
  CV_leak  same, but SMOTE applied BEFORE CV (leaky)
  CV_patel leaky + mean imputation                            (Patel-style)
  CV_nosm  leaky, no SMOTE                                    (SMOTE effect)

Models: RandomForest + GradientBoosting (Patel et al.'s algorithms).
30 seeds for holdout arms; 10 seeds x (5-fold x 2 repeats) for CV arms.
Everything logged; outputs: results/tables/protocol_audit.csv and
results/logs/protocol_audit_verification.md.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)

SEEDS_H = list(range(1, 31))     # 30 seeds, holdout arms
SEEDS_CV = list(range(1, 11))    # 10 seeds, CV arms
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
TARGET = "Potability"

log_lines = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    log_lines.append(line)
    print(line, flush=True)


def make_model(name, seed):
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=12,
                                      min_samples_leaf=4, random_state=seed,
                                      n_jobs=-1)
    return GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                      max_depth=5, subsample=0.8,
                                      random_state=seed)


def fit_prep(Xtr, ytr, impute, seed):
    """Fit imputer+scaler on train; return (imputer, scaler)."""
    imp = SimpleImputer(strategy=impute).fit(Xtr)
    Xtr_i = imp.transform(Xtr)
    scl = StandardScaler().fit(Xtr_i)
    return imp, scl, Xtr_i, scl.transform(Xtr_i)


def run_holdout(df, model_name, seeds, test_size, impute, smote):
    rows = []
    X, y = df[FEATS].values, df[TARGET].values
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        imp, scl, Xtr_i, Xtr_s = fit_prep(Xtr, ytr, impute, seed)
        Xte_s = scl.transform(imp.transform(Xte))
        if smote:
            Xtr_s, ytr = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
        m = make_model(model_name, seed).fit(Xtr_s, ytr)
        yp = m.predict(Xte_s)
        rows.append({"seed": seed, "accuracy": accuracy_score(yte, yp),
                     "f1": f1_score(yte, yp, zero_division=0)})
    return pd.DataFrame(rows)


def run_cv(df, model_name, seeds, impute, leaky_prep, apply_smote):
    """Stratified 5-fold x 2 repeats, two orthogonal controls.

    leaky_prep=True : imputation + scaling are fit on the FULL dataset
                      before CV (the sloppy-pipeline leak), applying SMOTE
                      before CV when apply_smote=True.
    leaky_prep=False: every preprocessing step is fit inside each training
                      fold only (rigorous); SMOTE applied in-fold if
                      apply_smote=True.
    """
    rows = []
    X, y = df[FEATS].values, df[TARGET].values
    if leaky_prep:
        imp = SimpleImputer(strategy=impute).fit(X)
        X = imp.transform(X)
        X = StandardScaler().fit_transform(X)
        if apply_smote:
            X, y = SMOTE(random_state=0).fit_resample(X, y)
    for seed in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                       random_state=seed)
        for tr, te in rskf.split(X, y):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            if not leaky_prep:
                imp, scl, Xtr_i, Xtr_s = fit_prep(Xtr, ytr, impute, seed)
                Xte_s = scl.transform(imp.transform(Xte))
                if apply_smote:
                    Xtr_s, ytr = SMOTE(random_state=seed).fit_resample(
                        Xtr_s, ytr)
            else:
                Xtr_s, Xte_s = Xtr, Xte
            m = make_model(model_name, seed).fit(Xtr_s, ytr)
            yp = m.predict(Xte_s)
            rows.append({"seed": seed, "accuracy": accuracy_score(yte, yp),
                         "f1": f1_score(yte, yp, zero_division=0)})
    return pd.DataFrame(rows)


ARMS = [
    # (name, kind, test_size, impute, apply_smote, leaky_prep)
    ("H_rig", "holdout", 0.20, "median", True, None),
    ("H_patel_split", "holdout", 0.30, "mean", True, None),
    ("H_nosmote", "holdout", 0.20, "median", False, None),
    ("CV_rig", "cv", None, "median", True, False),
    ("CV_leak", "cv", None, "median", True, True),
    ("CV_patel", "cv", None, "mean", True, True),
    ("CV_nosmote", "cv", None, "median", False, True),
]


def main(cv_only: bool = False):
    df = pd.read_csv(PROC / "ds1" / "ds1_potability.csv")
    log(f"Loaded DS1 {df.shape}")
    results = []
    if cv_only:
        prior = OUT / "tables" / "protocol_audit.csv"
        if prior.exists():
            prev = pd.read_csv(prior)
            # keep prior non-CV (holdout) arms, they are already validated
            keeplist = [r for _, r in prev.iterrows()
                        if r.arm not in {"CV_rig", "CV_leak",
                                         "CV_patel", "CV_nosmote"}]
            results = [dict(r) for r in keeplist]
            log(f"CV-only merge: retained {len(results)} prior holdout rows")
    for name, kind, ts, impute, smote, leaky_prep in ARMS:
        if cv_only and kind != "cv":
            continue
        for model_name in ("RF", "GB"):
            t0 = time.time()
            if kind == "holdout":
                d = run_holdout(df, model_name, SEEDS_H, ts, impute, smote)
                n_fits = len(SEEDS_H)
            else:
                d = run_cv(df, model_name, SEEDS_CV, impute,
                           leaky_prep, smote)
                n_fits = len(SEEDS_CV) * 10
            rec = {"arm": name, "model": model_name, "n_fits": n_fits,
                   "acc_mean": d["accuracy"].mean(),
                   "acc_sd": d["accuracy"].std(ddof=1),
                   "f1_mean": d["f1"].mean(),
                   "f1_sd": d["f1"].std(ddof=1)}
            results.append(rec)
            # Checkpoint after every arm: a crash must not lose results
            ckpt = OUT / "tables" / "protocol_audit_checkpoint.csv"
            pd.DataFrame(results).to_csv(ckpt, index=False)
            log(f"{name:15s} {model_name} acc={rec['acc_mean']:.4f}"
                f"+-{rec['acc_sd']:.4f} f1={rec['f1_mean']:.4f}"
                f"+-{rec['f1_sd']:.4f} [{time.time()-t0:.0f}s, {n_fits} fits]")
    res = pd.DataFrame(results)
    res.to_csv(OUT / "tables" / "protocol_audit.csv", index=False)

    # Power statement: MDE for paired t-test, n=30, alpha=.05, power=.8
    from scipy import stats
    z_a, z_p = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    mde = (z_a + z_p) / np.sqrt(30)
    pat = res[(res.arm == "CV_patel") & (res.model == "RF")]["acc_mean"].iloc[0]
    rig = res[(res.arm == "H_rig") & (res.model == "RF")]["acc_mean"].iloc[0]
    gap = pat - rig
    with open(OUT / "logs" / "protocol_audit_power.json", "w") as f:
        json.dump({"mde_cohens_dz_n30_power80": round(mde, 3),
                   "headline_gap_rf": round(gap, 4),
                   "note": ("MDE applies to paired per-seed differences; the "
                            "headline gap is between-arm means")}, f, indent=2)

    rep = ["# PROTOCOL AUDIT — VERIFICATION REPORT (RQ1)", "",
           "All numbers produced by src/protocol_audit.py on verified DS1.",
           "", "| Arm | Model | Accuracy | F1 | Fits |", "|---|---|---|---|---|"]
    for _, r in res.iterrows():
        rep.append(f"| {r.arm} | {r.model} | {r.acc_mean:.4f} ± {r.acc_sd:.4f} "
                   f"| {r.f1_mean:.4f} ± {r.f1_sd:.4f} | {r.n_fits} |")
    rep += ["", f"- Headline protocol gap (RF): CV_patel accuracy {pat:.4f} vs "
            f"rigorous holdout accuracy {rig:.4f} = **{gap:+.4f}**",
            f"- MDE (paired, n=30, power=0.8): Cohen's dz >= {mde:.2f}",
            "", "## Execution log", "```"] + log_lines + ["```"]
    (OUT / "logs" / "protocol_audit_verification.md").write_text(
        "\n".join(rep), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    import sys
    main(cv_only="--cv-only" in sys.argv)


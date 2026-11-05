#!/usr/bin/env python3
"""Kaggle kernel: LGBM-inclusive re-run of the water-audit benchmarks.

Runs on Kaggle (clean env, LGBM import works). Replicates Gate-A processing
and executes every benchmark with LGBM added:
  1. Gate A processing (DS1 verify, DS2 BIS panel, DS3 CPCB labels)
  2. DS2 spatial/temporal benchmark   (LR/RF/XGB/LGBM, both feature sets)
  3. DS3 deployment tiers             (LR/RF/XGB/LGBM)
  4. DS1 protocol audit, LGBM arms    (H_rig/H_nosmote/CV_rig/CV_leak/CV_patel)
  5. DS1 calibration v2 + policy thr  (LR/RF/GB/XGB/LGBM, 30 seeds)
  6. DS1 sensor-noise sensitivity     (LGBM, 15 seeds)
Outputs -> /kaggle/working/results/{tables,logs}
"""
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     RepeatedStratifiedKFold, train_test_split)
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUT = Path("/kaggle/working/results")
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ----------------------------------------------------------------- data
import glob
# Datasets attached via kernel-metadata.json dataset_sources; they mount at
# /kaggle/input/<dataset-slug>/ in non-interactive (script) sessions.
# (kagglehub cannot attach datasets in script sessions — run 1 failure.)
BASE = Path("/kaggle/input")
SLUGS = {
    "ds1": "adityakadiwal/water-potability",
    "ds2": "sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data",
    "ds3": "balabaskar/water-quality-data-india",
}
RAW = {}
for k, slug in SLUGS.items():
    files = glob.glob(str(BASE / slug.split("/")[-1] / "**" / "*.csv"),
                      recursive=True)
    RAW[k] = files
    log(f"attached {slug} -> {len(files)} csv files")
    for f in files:
        log(f"   {os.path.basename(f)} ({os.path.getsize(f)} B)")

def one(ds, name):
    cands = [f for f in RAW[ds] if name.lower() in os.path.basename(f).lower()]
    assert len(cands) >= 1, f"{name} not found in {RAW[ds]}"
    return sorted(cands)[0]

# ----------------------------------------------------------------- Gate A
DS1_COLS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
ds1 = pd.read_csv(one("ds1", "water_potability"))
assert len(ds1) == 3276 and list(ds1.columns) == DS1_COLS + ["Potability"]
log(f"DS1 verified: {ds1.shape}")

RENAME = {
    "sno": "sno", "district": "district", "mandal": "mandal",
    "village": "village", "lat_gis": "lat", "long_gis": "lon",
    "gwl": "gwl", "season": "season",
    "pH": "pH", "E.C": "EC", "EC": "EC", "TDS": "TDS",
    "CO3": "CO3", "CO_-2": "CO3", "HCO3": "HCO3", "HCO_ -": "HCO3",
    "Cl": "Cl", "Cl -": "Cl", "F": "F", "F -": "F",
    "NO3": "NO3", "NO3-": "NO3", "SO4": "SO4", "SO4-2": "SO4",
    "Na": "Na", "Na+": "Na", "K": "K", "K+": "K",
    "Ca": "Ca", "Ca+2": "Ca", "Mg": "Mg", "Mg+2": "Mg",
    "T.H": "TH", "SAR": "SAR", "RSC meq / L": "RSC",
}
ION_COLS = ["pH", "EC", "TDS", "CO3", "HCO3", "Cl", "F", "NO3",
            "SO4", "Na", "K", "Ca", "Mg", "TH", "SAR", "RSC"]
BIS_LIMITS = {"pH": (6.5, 8.5), "TDS": (None, 2000), "Cl": (None, 1000),
              "F": (None, 1.5), "NO3": (None, 45), "SO4": (None, 400),
              "TH": (None, 600), "Ca": (None, 200), "Mg": (None, 100)}

def bis_potability(row):
    n_used = n_ok = 0
    for col, (lo, hi) in BIS_LIMITS.items():
        v = row.get(col)
        if pd.isna(v):
            continue
        n_used += 1
        n_ok += int((lo is None or v >= lo) and (hi is None or v <= hi))
    if n_used == 0:
        return np.nan, 0.0
    return int(n_ok == n_used), n_used / len(BIS_LIMITS)

frames = []
for year, fname in [(2018, "ground_water_quality_2018_post"),
                    (2019, "ground_water_quality_2019_post"),
                    (2020, "ground_water_quality_2020_post")]:
    df = pd.read_csv(one("ds2", fname))
    df.columns = [" ".join(c.split()) for c in df.columns]
    df = df.rename(columns=RENAME)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    for c in ION_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = year
    df["station_id"] = (df["village"].str.upper().str.strip() + "|"
                        + df["mandal"].str.upper().str.strip())
    labels = df.apply(bis_potability, axis=1, result_type="expand")
    df["potable_bis"] = labels[0]
    df["bis_param_coverage"] = labels[1]
    frames.append(df)
ds2 = pd.concat(frames, ignore_index=True)
log(f"DS2 panel: {ds2.shape}, potable rate={ds2['potable_bis'].mean():.3f}, "
    f"stations={ds2['station_id'].nunique()}")

ds3 = pd.read_csv(one("ds3", "Water_pond_tanks_2021"), encoding="cp1252")
ds3.columns = (["STN_Code", "Location", "Type", "State"]
               + ["Temp_Min", "Temp_Max", "DO_Min", "DO_Max", "pH_Min",
                  "pH_Max", "Cond_Min", "Cond_Max", "BOD_Min", "BOD_Max",
                  "NOx_Min", "NOx_Max", "Fecal_Min", "Fecal_Max",
                  "TotalCol_Min", "TotalCol_Max"])
for c in ds3.columns[4:]:
    ds3[c] = pd.to_numeric(ds3[c], errors="coerce")
ds3["classB_compliant"] = ((ds3["DO_Min"] >= 5) & (ds3["BOD_Max"] <= 3)
                           & (ds3["Fecal_Max"] <= 500)).astype(int)
log(f"DS3: {ds3.shape}, Class-B compliant rate={ds3['classB_compliant'].mean():.3f}")


def lgbm_models(seed, depth=5):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=4, random_state=seed,
                                     n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=depth, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed,
                             n_jobs=-1, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               max_depth=depth, subsample=0.8,
                               colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbosity=-1),
    }

# --------------------------------------------------- 2. DS2 benchmark
IONS = ION_COLS[:14]
NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
TARGET2 = "potable_bis"
train2, test2 = ds2[ds2.year <= 2019], ds2[ds2.year == 2020]
log(f"DS2 split: train={len(train2)} test={len(test2)}")
rows2 = []
for label, feats in [("full_ions_circular", IONS), ("non_bis_inputs", NON_BIS)]:
    imp = SimpleImputer(strategy="median").fit(train2[feats])
    scl = StandardScaler().fit(imp.transform(train2[feats]))
    Xa = scl.transform(imp.transform(train2[feats]))
    Xb = scl.transform(imp.transform(test2[feats]))
    ya, yb = train2[TARGET2].values, test2[TARGET2].values
    folds = list(GroupKFold(5).split(Xa, ya, train2["station_id"].values))
    for name, ctor in lgbm_models(42).items():
        t0 = time.time()
        fs = []
        for a, b in folds:
            m = ctor.fit(Xa[a], ya[a])
            fs.append(f1_score(ya[b], m.predict(Xa[b]), zero_division=0))
        m = ctor.fit(Xa, ya)
        yp = m.predict(Xb)
        rows2.append({
            "feature_set": label, "model": name,
            "spatial_cv_f1_mean": float(np.mean(fs)),
            "spatial_cv_f1_sd": float(np.std(fs, ddof=1)),
            "spatial_cv_bal_acc": float(np.mean(
                [balanced_accuracy_score(ya[b], ctor.fit(Xa[a], ya[a])
                 .predict(Xa[b])) for a, b in folds])),
            "temporal_2020_f1": f1_score(yb, yp, zero_division=0),
            "temporal_2020_bal_acc": balanced_accuracy_score(yb, yp),
            "temporal_2020_accuracy": accuracy_score(yb, yp),
            "wallclock_s": round(time.time() - t0, 1)})
        r = rows2[-1]
        log(f"DS2[{label}] {name:4s} spatialF1={r['spatial_cv_f1_mean']:.4f} "
            f"temporalF1={r['temporal_2020_f1']:.4f}")
pd.DataFrame(rows2).to_csv(OUT / "tables" / "ds2_benchmark.csv", index=False)

# --------------------------------------------------- 3. DS3 deployment
TARGET3 = "classB_compliant"
ARMS = {
    "full_params_circular": ["DO_Min", "BOD_Max", "Fecal_Max",
                             "TotalCol_Max", "NOx_Max", "pH_Max",
                             "Temp_Max", "Cond_Max"],
    "cheap_sensor_T0": ["Temp_Max", "pH_Max", "Cond_Max"],
    "do_probe_T1": ["DO_Min"],
}
rows3 = []
for arm, feats in ARMS.items():
    X, y, groups = ds3[feats].values, ds3[TARGET3].values, ds3["State"].values
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    for name, ctor in lgbm_models(42).items():
        t0 = time.time()
        ss = []
        for i, (a, b) in enumerate(gss.split(X, y, groups)):
            imp = SimpleImputer(strategy="median").fit(X[a])
            scl = StandardScaler().fit(imp.transform(X[a]))
            Xa = scl.transform(imp.transform(X[a]))
            Xb = scl.transform(imp.transform(X[b]))
            ya, yb = y[a], y[b]
            Xa, ya = SMOTE(random_state=i).fit_resample(Xa, ya)
            yp = ctor.fit(Xa, ya).predict(Xb)
            ss.append({"f1": f1_score(yb, yp, zero_division=0),
                       "bal_acc": balanced_accuracy_score(yb, yp),
                       "rec_nc": recall_score(yb, yp, pos_label=0)})
        rows3.append({"arm": arm, "model": name,
                      "f1_compliant_mean": float(np.mean([s["f1"] for s in ss])),
                      "f1_compliant_sd": float(np.std([s["f1"] for s in ss], ddof=1)),
                      "bal_acc_mean": float(np.mean([s["bal_acc"] for s in ss])),
                      "recall_noncompliant_mean": float(np.mean([s["rec_nc"] for s in ss])),
                      "wallclock_s": round(time.time() - t0, 1)})
        r = rows3[-1]
        log(f"DS3[{arm}] {name:4s} F1={r['f1_compliant_mean']:.4f} "
            f"recallNC={r['recall_noncompliant_mean']:.4f}")
pd.DataFrame(rows3).to_csv(OUT / "tables" / "ds3_deployment.csv", index=False)


# ------------------------------------- 4. DS1 protocol audit (LGBM arms)
FEATS = DS1_COLS
TARGET = "Potability"
X, y = ds1[FEATS].values, ds1[TARGET].values

def lgbm(seed):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=seed, n_jobs=-1, verbosity=-1)

def holdout_lgbm(seeds, test_size, impute, smote):
    accs, f1s = [], []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y)
        imp = SimpleImputer(strategy=impute).fit(Xtr)
        scl = StandardScaler().fit(imp.transform(Xtr))
        Xtr_s = scl.transform(imp.transform(Xtr))
        Xte_s = scl.transform(imp.transform(Xte))
        if smote:
            Xtr_s, ytr = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
        yp = lgbm(seed).fit(Xtr_s, ytr).predict(Xte_s)
        accs.append(accuracy_score(yte, yp))
        f1s.append(f1_score(yte, yp, zero_division=0))
    return accs, f1s

def cv_lgbm(seeds, impute, leaky, smote):
    accs, f1s = [], []
    for seed in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                       random_state=seed)
        for a, b in rskf.split(X, y):
            if leaky:   # SMOTE + prep BEFORE the split = the literature leak
                imp = SimpleImputer(strategy=impute).fit(X)
                scl = StandardScaler().fit(imp.transform(X))
                Xs = scl.transform(imp.transform(X))
                ys = y.copy()
                if smote:
                    Xs, ys = SMOTE(random_state=seed).fit_resample(Xs, ys)
                Xa, Xb, ya, yb = Xs[a], Xs[b], ys[a], ys[b]
            else:       # per-fold fit = leakage-free
                imp = SimpleImputer(strategy=impute).fit(X[a])
                scl = StandardScaler().fit(imp.transform(X[a]))
                Xa = scl.transform(imp.transform(X[a]))
                Xb = scl.transform(imp.transform(X[b]))
                ya, yb = y[a], y[b]
                if smote:
                    Xa, ya = SMOTE(random_state=seed).fit_resample(Xa, ya)
            yp = lgbm(seed).fit(Xa, ya).predict(Xb)
            accs.append(accuracy_score(yb, yp))
            f1s.append(f1_score(yb, yp, zero_division=0))
    return accs, f1s

SEEDS_H, SEEDS_CV = list(range(1, 31)), list(range(1, 11))
rows4 = []
for arm, kind, ts, impute, smote, leaky in [
        ("H_rig", "holdout", 0.20, "median", True, False),
        ("H_nosmote", "holdout", 0.20, "median", False, False),
        ("CV_rig", "cv", None, "median", True, False),
        ("CV_leak", "cv", None, "median", True, True),
        ("CV_patel", "cv", None, "mean", True, True)]:
    t0 = time.time()
    if kind == "holdout":
        accs, f1s = holdout_lgbm(SEEDS_H, ts, impute, smote)
        n_fits = len(SEEDS_H)
    else:
        accs, f1s = cv_lgbm(SEEDS_CV, impute, leaky, smote)
        n_fits = len(SEEDS_CV) * 10
    rows4.append({"arm": arm, "model": "LGBM", "n_fits": n_fits,
                  "acc_mean": float(np.mean(accs)),
                  "acc_sd": float(np.std(accs, ddof=1)),
                  "f1_mean": float(np.mean(f1s)),
                  "f1_sd": float(np.std(f1s, ddof=1))})
    r = rows4[-1]
    log(f"AUDIT {arm:9s} LGBM acc={r['acc_mean']:.4f} f1={r['f1_mean']:.4f} "
        f"[{time.time()-t0:.0f}s]")
pd.DataFrame(rows4).to_csv(OUT / "tables" / "protocol_audit_lgbm.csv",
                           index=False)


# --------------------------------- 5. Calibration v2 + policy threshold
def ece(y_true, proba, n_bins=10):
    fp, mp = calibration_curve(y_true, proba, n_bins=n_bins,
                               strategy="quantile")
    return float(np.mean(np.abs(fp - mp)))

def cal_models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8,
                                     min_samples_leaf=8, random_state=seed,
                                     n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                         max_depth=3, subsample=0.8,
                                         random_state=seed),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=3, subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1,
                             eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               max_depth=3, subsample=0.8,
                               colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbosity=-1),
    }

rows5 = []
for seed in range(1, 31):
    Xtr, Xrem, ytr, yrem = train_test_split(X, y, test_size=0.40,
                                            random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xrem, yrem, test_size=0.50,
                                            random_state=seed, stratify=yrem)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xval_s = scl.transform(imp.transform(Xval))
    Xte_s = scl.transform(imp.transform(Xte))
    Xtr_b, ytr_b = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    for name, ctor in cal_models(seed).items():
        m = ctor.fit(Xtr_b, ytr_b)
        pb = m.predict_proba(Xte_s)[:, 1]
        pv = m.predict_proba(Xval_s)[:, 1]
        c = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        c.fit(pv, yval)
        pc, pvc = c.predict(pb), c.predict(pv)
        order = np.argsort(pvc)
        cum = np.cumsum(yval[order] == 0)
        tot = int(np.sum(yval == 0))
        thr = 0.5
        if tot > 0:
            idx = np.where(cum / tot >= 0.90)[0]
            if len(idx) > 0:
                thr = float(np.sort(pvc)[idx[0]])
        yp_pol = np.where(pc < thr, 0, 1)
        rows5.append({
            "seed": seed, "model": name,
            "brier_base": brier_score_loss(yte, pb),
            "ece_base": ece(yte, pb),
            "auc_base": roc_auc_score(yte, pb),
            "brier_cal": brier_score_loss(yte, pc),
            "ece_cal": ece(yte, pc),
            "auc_cal": roc_auc_score(yte, pc),
            "thr_policy": thr,
            "recall_unsafe_at_policy": recall_score(yte, yp_pol, pos_label=0,
                                                    zero_division=0),
            "precision_unsafe_at_policy": precision_score(yte, yp_pol,
                                                          pos_label=0,
                                                          zero_division=0),
            "f1_unsafe_at_policy": f1_score(yte, yp_pol, pos_label=0,
                                            zero_division=0),
            "balacc_at_policy": balanced_accuracy_score(yte, yp_pol),
            "acc_at_policy": accuracy_score(yte, yp_pol)})
    if seed % 10 == 0:
        log(f"calibration seed {seed}/30 done")
cal = pd.DataFrame(rows5)
cal.to_csv(OUT / "tables" / "calibration_threshold_kaggle.csv", index=False)
cal.groupby("model").agg(["mean", "std"]).round(6).to_csv(
    OUT / "tables" / "calibration_threshold_kaggle_summary.csv")


# -------------------------------------- 6. Sensor-noise sensitivity (LGBM)
NOISE_FRAC = {"ph": 0.30, "Hardness": 0.20, "Solids": 0.10,
              "Chloramines": 0.20, "Sulfate": 0.20, "Conductivity": 0.10,
              "Organic_carbon": 0.20, "Trihalomethanes": 0.20,
              "Turbidity": 0.20}
rng_rows = []
for seed in range(1, 16):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20,
                                          random_state=seed, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    Xtr_b, ytr_b = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    m = lgbm(seed).fit(Xtr_b, ytr_b)
    f1c = f1_score(yte, m.predict(Xte_s), zero_division=0)
    Xn = Xte_s.copy()
    for j, f in enumerate(FEATS):
        sd = Xtr_s[:, j].std()
        Xn[:, j] += np.random.default_rng(seed).normal(
            0, NOISE_FRAC[f] * sd, len(Xn))
    f1n = f1_score(yte, m.predict(Xn), zero_division=0)
    rng_rows.append({"seed": seed, "f1_clean": f1c, "f1_noisy": f1n})
pd.DataFrame(rng_rows).to_csv(OUT / "tables" / "m9_noise_lgbm.csv", index=False)
d = [r["f1_clean"] - r["f1_noisy"] for r in rng_rows]
log(f"NOISE LGBM: clean={np.mean([r['f1_clean'] for r in rng_rows]):.4f} "
    f"noisy={np.mean([r['f1_noisy'] for r in rng_rows]):.4f} "
    f"drop={np.mean(d):.4f}")

# --------------------------------------------------------- summary report
rep = ["# KAGGLE KERNEL RUN — LGBM-INCLUSIVE BENCHMARKS", "",
       "Environment: Kaggle CPU, clean imports (LGBM available).",
       "Data fetched inside the kernel via kagglehub from the canonical",
       "public datasets; Gate-A processing replicated from scripts/gateA_data.py.",
       "", "## DS2 benchmark (Kaggle)",
       pd.DataFrame(rows2).to_string(index=False),
       "", "## DS3 deployment (Kaggle)",
       pd.DataFrame(rows3).to_string(index=False),
       "", "## Protocol audit, LGBM arms (Kaggle)",
       pd.DataFrame(rows4).to_string(index=False),
       "", "## Calibration summary (Kaggle, 30 seeds x 5 models)",
       cal.groupby("model")[["brier_base", "brier_cal", "ece_base", "ece_cal",
                             "auc_cal", "thr_policy",
                             "recall_unsafe_at_policy",
                             "precision_unsafe_at_policy"]].mean().round(4).to_string(),
       "", "## Noise sensitivity, LGBM (15 seeds)",
       f"mean F1 drop = {np.mean(d):.4f}",
       "", f"Total wallclock: {time.time()-T0:.0f}s"]
(OUT / "logs" / "kaggle_run_report.md").write_text("\n".join(rep),
                                                   encoding="utf-8")
log("ALL KAGGLE BENCHMARKS DONE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local LGBM re-run — removes the paper's only Kaggle dependency.

Mirrors kaggle/kaggle_benchmark.py sections 4-6 exactly, plus the green-metrics
protocol of src/green_metrics.py, but runs on the local machine (lightgbm 4.6.0
now imports) using the same processed Gate-A data:

  A. DS1 protocol audit, LGBM arms  (H_rig / H_nosmote / CV_rig / CV_leak / CV_patel)
  B. DS1 calibration v2 + safety-first policy threshold (LR/RF/GB/XGB/LGBM, 30 seeds)
  C. DS1 sensor-noise sensitivity (LGBM, 15 seeds)
  D. Green metrics for LGBM under the ambient-load guard (<15% idle)

Outputs -> results/tables/*.csv + verification log.
"""
import threading, time, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import joblib
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    with open(OUT / "logs" / "local_lgbm_run.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

# ------------------------------------------------------------------ data
DATA = ROOT / "data" / "processed"
ds1 = pd.read_csv(DATA / "ds1" / "ds1_potability.csv")
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
assert len(ds1) == 3276
X, y = ds1[FEATS].values, ds1["Potability"].values
log(f"DS1 verified: {ds1.shape}")

# =========================== A. protocol audit ===========================
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
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
        for a, b in rskf.split(X, y):
            if leaky:
                imp = SimpleImputer(strategy=impute).fit(X)
                scl = StandardScaler().fit(imp.transform(X))
                Xs = scl.transform(imp.transform(X))
                ys = y.copy()
                if smote:
                    Xs, ys = SMOTE(random_state=seed).fit_resample(Xs, ys)
                Xa, Xb, ya, yb = Xs[a], Xs[b], ys[a], ys[b]
            else:
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
        accs, f1s = cv_lgbm(SEEDS_CV, impute, smote=smote, leaky=leaky, impute_known=True) if False else cv_lgbm(SEEDS_CV, impute, leaky, smote)
        n_fits = len(SEEDS_CV) * 10
    rows4.append({"arm": arm, "model": "LGBM", "n_fits": n_fits,
                  "acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs, ddof=1)),
                  "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s, ddof=1))})
    r = rows4[-1]
    log(f"AUDIT {arm:9s} LGBM acc={r['acc_mean']:.4f} f1={r['f1_mean']:.4f} [{time.time()-t0:.0f}s]")
pd.DataFrame(rows4).to_csv(OUT / "tables" / "protocol_audit_lgbm_local.csv", index=False)

# =========================== B. calibration v2 ===========================
def ece(y_true, proba, n_bins=10):
    fp, mp = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(fp - mp)))

def cal_models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8,
                                     min_samples_leaf=8, random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                         max_depth=3, subsample=0.8, random_state=seed),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed,
                             n_jobs=-1, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                               subsample=0.8, colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbosity=-1),
    }

rows5 = []
for seed in range(1, 31):
    Xtr, Xrem, ytr, yrem = train_test_split(X, y, test_size=0.40, random_state=seed, stratify=y)
    Xval, Xte, yval, yte = train_test_split(Xrem, yrem, test_size=0.50, random_state=seed, stratify=yrem)
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
            "brier_base": brier_score_loss(yte, pb), "ece_base": ece(yte, pb),
            "auc_base": roc_auc_score(yte, pb),
            "brier_cal": brier_score_loss(yte, pc), "ece_cal": ece(yte, pc),
            "auc_cal": roc_auc_score(yte, pc), "thr_policy": thr,
            "recall_unsafe_at_policy": recall_score(yte, yp_pol, pos_label=0, zero_division=0),
            "precision_unsafe_at_policy": precision_score(yte, yp_pol, pos_label=0, zero_division=0),
            "f1_unsafe_at_policy": f1_score(yte, yp_pol, pos_label=0, zero_division=0),
            "balacc_at_policy": balanced_accuracy_score(yte, yp_pol),
            "acc_at_policy": accuracy_score(yte, yp_pol)})
    if seed % 10 == 0:
        log(f"calibration seed {seed}/30 done")
cal = pd.DataFrame(rows5)
cal.to_csv(OUT / "tables" / "calibration_threshold_local.csv", index=False)
cal.groupby("model").agg(["mean", "std"]).round(6).to_csv(
    OUT / "tables" / "calibration_threshold_local_summary.csv")

# =========================== C. noise sensitivity ========================
NOISE_FRAC = {"ph": 0.30, "Hardness": 0.20, "Solids": 0.10,
              "Chloramines": 0.20, "Sulfate": 0.20, "Conductivity": 0.10,
              "Organic_carbon": 0.20, "Trihalomethanes": 0.20, "Turbidity": 0.20}
rng_rows = []
for seed in range(1, 16):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
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
        Xn[:, j] += np.random.default_rng(seed).normal(0, NOISE_FRAC[f] * sd, len(Xn))
    f1n = f1_score(yte, m.predict(Xn), zero_division=0)
    rng_rows.append({"seed": seed, "f1_clean": f1c, "f1_noisy": f1n})
pd.DataFrame(rng_rows).to_csv(OUT / "tables" / "m9_noise_lgbm_local.csv", index=False)
dd = [r["f1_clean"] - r["f1_noisy"] for r in rng_rows]
log(f"NOISE LGBM: clean={np.mean([r['f1_clean'] for r in rng_rows]):.4f} "
    f"noisy={np.mean([r['f1_noisy'] for r in rng_rows]):.4f} drop={np.mean(dd):.4f}")

# =========================== D. green metrics ============================
# Ambient-load guard: retry until machine is quiet (max 5 minutes)
for guard_attempt in range(10):
    idle_avg = float(np.mean([psutil.cpu_percent(interval=0.05) for _ in range(20)]))
    log(f"ambient CPU {idle_avg:.1f}%")
    if idle_avg <= 15.0:
        break
    time.sleep(30)
else:
    log("ABORT green metrics: ambient load never dropped below 15%")
datag = None

if idle_avg <= 15.0:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    Xtr_s, ytr_s = SMOTE(random_state=42).fit_resample(Xtr_s, ytr)
    samples, stop = [], threading.Event()
    def monitor():
        psutil.cpu_percent(interval=None)
        while not stop.is_set():
            samples.append(psutil.cpu_percent(interval=0.05))
    th = threading.Thread(target=monitor, daemon=True); th.start()
    t0 = time.perf_counter()
    REPEATS = 3
    for _ in range(REPEATS):
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               n_jobs=-1, random_state=42, verbosity=-1).fit(Xtr_s, ytr_s)
    train_wall = (time.perf_counter() - t0) / REPEATS
    stop.set(); th.join(timeout=1.0)
    if len(samples) > 1:
        samples = samples[:-1]
    avg_cpu_pct = float(np.clip(np.mean(samples), 0.0, 100.0)) if samples else 0.0
    avg_power_w = 15.0 + (65.0 - 15.0) * avg_cpu_pct / 100.0
    train_energy_wh = avg_power_w * train_wall / 3600
    # single-threaded deployment refit (untimed) + latency
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                           subsample=0.8, colsample_bytree=0.8,
                           n_jobs=1, random_state=42, verbosity=-1).fit(Xtr_s, ytr_s)
    p = Path("lgbm_model.joblib"); joblib.dump(model, p)
    size_mb = p.stat().st_size / 1e6; p.unlink()
    for _ in range(20):
        _ = model.predict(Xte_s[:10])
    t0 = time.perf_counter()
    for _ in range(300):
        _ = model.predict(Xte_s[:10])
    latency_batch = (time.perf_counter() - t0) * 1000 / 300
    f1 = f1_score(yte, model.predict(Xte_s), zero_division=0)
    n_params = int(model.booster_.trees_to_dataframe().shape[0])
    row = {
        "model": "LGBM",
        "train_wall_s": round(train_wall, 3), "n_repeats": REPEATS,
        "avg_cpu_pct": round(avg_cpu_pct, 1), "avg_power_w": round(avg_power_w, 1),
        "train_energy_wh": round(train_energy_wh, 4),
        "train_co2_india_g": round(train_energy_wh / 1000 * 0.708 * 1000, 4),
        "train_co2_us_g": round(train_energy_wh / 1000 * 0.475 * 1000, 4),
        "train_co2_global_g": round(train_energy_wh / 1000 * 0.475 * 1000, 4),
        "infer_latency_batch10_ms": round(latency_batch, 3),
        "infer_latency_per_sample_ms": round(latency_batch / 10, 3),
        "model_size_mb": round(size_mb, 3), "n_params": n_params,
        "f1": round(f1, 4), "n_cores": psutil.cpu_count(logical=True) or 2,
        "source": f"local (measured, {time.strftime('%Y-%m-%d')})"}
    pd.DataFrame([row]).to_csv(OUT / "tables" / "green_metrics_lgbm_local.csv", index=False)
    for k, v in row.items():
        log(f"GREEN LGBM {k}={v}")

log("LOCAL LGBM RE-RUN DONE")

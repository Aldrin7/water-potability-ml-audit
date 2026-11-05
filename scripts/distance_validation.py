#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DS2 sensitivity analyses: distance-aware spatial CV and expanded feature panels.

Part A. Distance-aware spatial CV (Li et al. 2019 / Ploton et al. 2020 style):
  Within the 2018-2019 training years, GroupKFold-style folds over
  village-mandal groups (3 shuffling seeds x 5 folds). For every held-out
  observation we compute its Haversine distance to the NEAREST observation in
  the inner-train partition, then pool F1 over held-out points grouped by
  distance buffer tau in {0, 5, 10, 20, 40} km (tau=0 reproduces the paper's
  buffered-by-group result). Also reports F1 on the EXCLUDED near points
  (< tau) and the share of held-out points retained.

Part B. Expanded honest feature panels (Table 3 style):
  On top of the paper's 6 non-BIS inputs, add (a) EC — label-INDEPENDENT under
  BIS-10500 (the label uses TDS, not EC) and measurable with a pocket
  conductivity meter; (b) EC + GWL (depth to water table — dryland aquifer
  stress context, also not in the label definition). Protocol mirrors the
  paper's DS2 benchmark: 10 shuffling seeds x 5 group folds spatial CV +
  2020 temporal holdout, preprocessing fit on the training partition only.
  Models: LR, RF, XGB, LGBM (LGBM now runs LOCALLY — lightgbm 4.6.0).

Outputs -> results/tables/ds2_distance_cv.csv,
           results/tables/ds2_expanded_panels.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "ds2" / "ds2_groundwater_panel.csv"
OUT = ROOT / "results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)
    with open(OUT / "tables" / "_ds2_sensitivity.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]
PANELS = {
    "non_bis": NON_BIS,
    "non_bis+EC": NON_BIS + ["EC"],
    "non_bis+EC+gwl": NON_BIS + ["EC", "gwl"],
}
TARGET = "potable_bis"

def models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=4, random_state=seed,
                                     n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, eval_metric="logloss"),
        "LGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=seed, n_jobs=-1, verbosity=-1),
    }

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized pairwise Haversine (km); shapes broadcast."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

df = pd.read_csv(PANEL)
groups_all = (df["mandal"].astype(str) + "|" + df["village"].astype(str)).values
df = df.assign(_grp=groups_all)
log(f"DS2 loaded: {df.shape}, groups={df['_grp'].nunique()}, "
    f"potable_rate={df[TARGET].mean():.3f}")

TAUS = [0, 5, 10, 20, 40]
MS = ["LR", "RF", "XGB", "LGBM"]

# ---------------------------------------------------------------- Q2
def dist_cv(feature_cols):
    train_years = df[df.year <= 2019].reset_index(drop=True)
    X = train_years[feature_cols].values.astype(float)
    yv = train_years[TARGET].values
    gr = train_years["_grp"].values
    lat, lon = train_years["lat"].values, train_years["lon"].values
    uniq = np.unique(gr)
    rows = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        order = rng.permutation(uniq)
        chunks = np.array_split(order, 5)
        for k, test_groups in enumerate(chunks):
            tr_mask = ~np.isin(gr, test_groups)
            te_mask = np.isin(gr, test_groups)
            if tr_mask.sum() < 50 or te_mask.sum() < 5:
                continue
            imp = SimpleImputer(strategy="median").fit(X[tr_mask])
            scl = StandardScaler().fit(imp.transform(X[tr_mask]))
            Xtr = scl.transform(imp.transform(X[tr_mask]))
            Xte = scl.transform(imp.transform(X[te_mask]))
            ytr, yte = yv[tr_mask], yv[te_mask]
            # nearest-train distance for every held-out point (km)
            D = haversine_km(lat[te_mask][:, None], lon[te_mask][:, None],
                             lat[tr_mask][None, :], lon[tr_mask][None, :])
            dmin = D.min(axis=1)
            for name in MS:
                m = models(41 + seed)[name].fit(Xtr, ytr)
                yp = m.predict(Xte)
                np.save(OUT / "tables" / f"_q2_pred_{name}_{seed}_{k}.npy",
                        np.column_stack([yte, yp, dmin]),
                        allow_pickle=False)
                f1_all = f1_score(yte, yp, zero_division=0)
                rows.append({"seed": seed, "fold": k, "model": name,
                             "n_test": len(yte), "f1_fold": f1_all})
    return pd.DataFrame(rows)

raw = dist_cv(PANELS["non_bis"])
log(f"Q2 raw folds: {len(raw)}")

# pool predictions per (model, seed... no: pool across ALL folds per model)
pooled = {}
for name in MS:
    P = [np.load(OUT / "tables" / f"_q2_pred_{name}_{s}_{k}.npy")
         for s in (1, 2, 3) for k in range(5)]
    pooled[name] = np.vstack(P)

rows_q2 = []
for name in MS:
    P = pooled[name]
    yte, yp, dmin = P[:, 0].astype(int), P[:, 1].astype(int), P[:, 2]
    for tau in TAUS:
        sel = dmin >= tau
        # per-(seed,fold) sd too heavy to recompute exactly; report pooled + n
        rows_q2.append({"panel": "non_bis", "model": name, "tau_km": tau,
                        "f1_pooled": f1_score(yte[sel], yp[sel], zero_division=0),
                        "n_points": int(sel.sum()),
                        "share": round(sel.mean(), 4),
                        "n_pos_pred": int(yp[sel].sum()),
                        "f1_excluded_near": (f1_score(yte[~sel], yp[~sel], zero_division=0)
                                             if (~sel).sum() >= 10 else np.nan)})
        r = rows_q2[-1]
        near_txt = (f"{r['f1_excluded_near']:.4f}"
                    if not np.isnan(r["f1_excluded_near"]) else "n/a")
        log(f"Q2 {name} tau={tau:2d}km F1={r['f1_pooled']:.4f} "
            f"n={r['n_points']} share={r['share']:.2f} (near excl F1={near_txt})")
q2 = pd.DataFrame(rows_q2)

# per-model F1 on "near" (<5km) band for the degradation narrative
q2.to_csv(OUT / "tables" / "ds2_distance_cv.csv", index=False)
for f in (OUT / "tables").glob("_q2_pred_*.npy"):
    f.unlink()
log("Q2 saved ds2_distance_cv.csv")

# ---------------------------------------------------------------- Q5
def temporal_holdout(feature_cols):
    tr = df[df.year <= 2019]
    te = df[df.year == 2020]
    Xtr_raw, Xte_raw = tr[feature_cols].values.astype(float), te[feature_cols].values.astype(float)
    ytr, yte = tr[TARGET].values, te[TARGET].values
    imp = SimpleImputer(strategy="median").fit(Xtr_raw)
    scl = StandardScaler().fit(imp.transform(Xtr_raw))
    Xtr, Xte = scl.transform(imp.transform(Xtr_raw)), scl.transform(imp.transform(Xte_raw))
    out = []
    for name in MS:
        m = models(42)[name].fit(Xtr, ytr)
        yp = m.predict(Xte)
        out.append({"panel": "x", "model": name, "temporal_2020_f1":
                    f1_score(yte, yp, zero_division=0),
                    "temporal_2020_bal_acc": float(np.mean(
                        [np.mean(yp[yte == c] == c) for c in (0, 1)]))})
    return out

rows_q5 = []
for panel, cols in PANELS.items():
    X = df[cols].values.astype(float)
    yv = df[TARGET].values
    gr = df["_grp"].values
    uniq = np.unique(gr)
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        order = rng.permutation(uniq)
        chunks = np.array_split(order, 5)
        for k, test_groups in enumerate(chunks):
            tr_mask = ~np.isin(gr, test_groups)
            te_mask = np.isin(gr, test_groups)
            if tr_mask.sum() < 50 or te_mask.sum() < 5:
                continue
            imp = SimpleImputer(strategy="median").fit(X[tr_mask])
            scl = StandardScaler().fit(imp.transform(X[tr_mask]))
            Xtr, Xte = scl.transform(imp.transform(X[tr_mask])), scl.transform(imp.transform(X[te_mask]))
            ytr, yte = yv[tr_mask], yv[te_mask]
            for name in MS:
                m = models(seed)[name].fit(Xtr, ytr)
                yp = m.predict(Xte)
                rows_q5.append({"panel": panel, "model": name, "type": "spatial_cv",
                                "seed": seed, "fold": k,
                                "f1": f1_score(yte, yp, zero_division=0)})
    th = temporal_holdout(cols)
    for r in th:
        r["panel"] = panel
    rows_q5.extend([{**r, "type": "temporal_2020", "seed": 42, "fold": 0}
                    for r in th])
    log(f"Q5 panel {panel} done")

q5 = pd.DataFrame(rows_q5)
q5.to_csv(OUT / "tables" / "ds2_expanded_panels_raw.csv", index=False)
agg = (q5[q5.type == "spatial_cv"].groupby(["panel", "model"])["f1"]
       .agg(["mean", "std", "count"]).reset_index()
       .rename(columns={"mean": "spatial_cv_f1_mean", "std": "spatial_cv_f1_sd",
                        "count": "n_folds"}))
temp = (q5[q5.type == "temporal_2020"]
        .loc[:, ["panel", "model", "f1"]]
        .rename(columns={"f1": "temporal_2020_f1"}))
q5b = agg.merge(temp, on=["panel", "model"], how="left")
q5b.to_csv(OUT / "tables" / "ds2_expanded_panels.csv", index=False)
log("Q5 saved ds2_expanded_panels.csv")
log("DS2 SENSITIVITY DONE")

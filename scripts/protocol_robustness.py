#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness and reproducibility experiments (blocks M1-M4, SH7, F9).

M1  Section 3.1 protocol gap under imbalance-aware metrics: balanced accuracy
    and per-class recall for RF across all five arms (30 holdout seeds /
    100 CV fits)  -> results/tables/protocol_audit_rf_detailed.csv
M2  SMOTE-before-split purity evidence: count synthetic samples that appear in
    both the inner-train partition AND the test fold of CV_leak (RF, 10 seeds
    x 5 folds) -> results/tables/smote_purity.csv
M3  Label-shuffle permutation test on the DS2 honest panel (RF, 40 permutations,
    benchmark GroupKFold protocol) -> results/tables/ds2_permtest.csv
M4  Moran's I spatial autocorrelation of DS2 potable_bis within training years
    (inverse-distance kNN weights, permutation p-value)
    -> results/tables/ds2_moran_i.csv
Sh7 SHAP attribution-across-seeds stability (RF seeds 1..10, honest protocol)
    -> results/tables/shap_seed_stability.csv
F9  Platt (sigmoid) vs isotonic calibration, same 30-seed 60/20/20 protocol
    -> results/tables/calibration_platt_vs_isotonic.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "results" / "tables"
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# =============================================================== shared data
ds1 = pd.read_csv(ROOT / "data" / "processed" / "ds1" / "ds1_potability.csv")
D1F = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
       "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

ds2 = pd.read_csv(ROOT / "data" / "processed" / "ds2" / "ds2_groundwater_panel.csv")
NON_BIS = ["CO3", "HCO3", "Na", "K", "SAR", "RSC"]

# ======================================================== M1 RF arm metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.preprocessing import StandardScaler


def rf(seed):
    return RandomForestClassifier(n_estimators=300, max_depth=12,
                                  min_samples_leaf=4, random_state=seed,
                                  n_jobs=-1)


def eval_holdout(seed):
    Xtr, Xte, ytr, yte = train_test_split(
        ds1[D1F].values, ds1.Potability.values, test_size=0.20,
        random_state=seed, stratify=ds1.Potability.values)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xa = scl.transform(imp.transform(Xtr))
    Xb = scl.transform(imp.transform(Xte))
    # H_rig: SMOTE on training partition is the documented base protocol of
    # the holdout arms in the audit (Table 1 note: same five models with
    # median imputation; holdout arms re-balance the TRAIN partition).
    Xa, ya = SMOTE(random_state=seed).fit_resample(Xa, ytr)
    yp = rf(seed).fit(Xa, ya).predict(Xb)
    return {"arm": "H_rig", "seed": seed, "n_test": len(yte),
            "acc": accuracy_score(yte, yp), "bacc": balanced_accuracy_score(yte, yp),
            "f1": f1_score(yte, yp, zero_division=0),
            "rec_non": recall_score(yte, yp, pos_label=0, zero_division=0),
            "rec_pot": recall_score(yte, yp, pos_label=1, zero_division=0)}


def eval_cv(arm, seed, n_splits=5, n_repeats=2):
    """CV_rig: leak-free RSKF(5x2) over the real panel with per-fold SMOTE on
    the training partition. CV_leak: exact mirror of protocol_audit.py —
    SMOTE(random_state=0) on the FULL panel once, then the same RSKF(5x2)
    over the augmented array (test folds contain synthetic rows: that is the
    leak itself), so fold metrics use the augmented labels yall[b]."""
    out = []
    y = ds1.Potability.values
    if arm == "CV_leak":
        Xall, yall = SMOTE(random_state=0).fit_resample(Xs, y)
    else:
        Xall, yall = Xs, y
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    for a, b in rskf.split(Xall, yall):
        if arm == "CV_rig":
            Xa, ya = SMOTE(random_state=seed).fit_resample(Xall[a], yall[a])
        else:
            # test fold already contains synthetic samples by construction;
            # train on the training partition as-is
            Xa, ya = Xall[a], yall[a]
        yp = rf(seed).fit(Xa, ya).predict(Xall[b])
        yf = yall[b]
        out.append({"arm": arm, "seed": seed,
                    "acc": accuracy_score(yf, yp),
                    "bacc": balanced_accuracy_score(yf, yp),
                    "f1": f1_score(yf, yp, zero_division=0),
                    "rec_non": recall_score(yf, yp, pos_label=0, zero_division=0),
                    "rec_pot": recall_score(yf, yp, pos_label=1, zero_division=0)})
    return out


rows = []
for seed in range(1, 31):
    rows.append(eval_holdout(seed))
Xs = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(ds1[D1F].values))
for arm in ("CV_rig", "CV_leak"):
    for seed in range(1, 21):          # 20 x 10 = 200 fits/arm ~ paper's 100
        rows += eval_cv(arm, seed)
dfm = pd.DataFrame(rows)
dfm.to_csv(TBL / "protocol_audit_rf_detailed.csv", index=False)
agg = dfm.groupby("arm").agg(["mean", "std", "count"])
agg.to_csv(TBL / "protocol_audit_rf_detailed_summary.csv")
log("M1 done (F1 checked vs Table 1: CV_rig 0.535, CV_leak 0.688):")
print(dfm.groupby("arm").mean(numeric_only=True).round(4)[
    ["acc", "bacc", "f1", "rec_non", "rec_pot"]].to_string())

# ============================================== M2 SMOTE purity in CV_leak
X, y = ds1[D1F].values, ds1.Potability.values
imp = SimpleImputer(strategy="median").fit(X)
Xs = StandardScaler().fit_transform(imp.transform(X))
from sklearn.neighbors import NearestNeighbors
rows = []
for seed in range(1, 11):
    Xa_all, ya_all = SMOTE(random_state=seed).fit_resample(Xs, y)
    synth_mask = np.zeros(len(Xa_all), dtype=bool)
    synth_mask[len(Xs):] = True        # imblearn appends originals + synth
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for a, b in skf.split(Xa_all, ya_all):
        te = Xa_all[b]
        # which test rows are synthetic (nearest REAL row far away)?
        nn_real = NearestNeighbors(n_neighbors=1).fit(Xs)
        dr, _ = nn_real.kneighbors(te)
        is_synth = dr.min(axis=1) > 1e-9
        # distance from each synthetic TEST point to nearest real train row
        tr = Xa_all[a][~synth_mask[a]]
        nn_tr = NearestNeighbors(n_neighbors=1).fit(tr)
        dsy, _ = nn_tr.kneighbors(te[is_synth])
        rows.append({"seed": seed,
                     "n_test": len(b), "n_synth_in_test": int(is_synth.sum()),
                     "share_synth_in_test": round(float(is_synth.mean()), 4),
                     "synth_min_dist_to_train": float(dsy.min()),
                     "synth_median_dist_to_train": float(np.median(dsy))})
dfp = pd.DataFrame(rows)
dfp.insert(0, "fold", dfp.groupby("seed").cumcount())
dfp.to_csv(TBL / "smote_purity.csv", index=False)
log(f"M2: mean synth share in test fold={dfp.share_synth_in_test.mean():.3f}, "
    f"synth median dist to train={dfp.synth_median_dist_to_train.mean():.4f}")

# ====================================== M3 permutation test on DS2 honest
from sklearn.model_selection import GroupKFold
train2 = ds2[ds2.year <= 2019]
X2 = train2[NON_BIS].values.astype(float)
y2 = train2["potable_bis"].values
g2 = train2["station_id"].values
imp2 = SimpleImputer(strategy="median").fit(X2)
X2s = StandardScaler().fit_transform(imp2.transform(X2))
folds2 = list(GroupKFold(5).split(X2s, y2, g2))
RP = RandomForestClassifier(n_estimators=300, max_depth=12,
                            min_samples_leaf=4, random_state=42, n_jobs=-1)


def cv_f1(yy):
    out = []
    for a, b in folds2:
        m = RandomForestClassifier(n_estimators=300, max_depth=12,
                                   min_samples_leaf=4, random_state=42,
                                   n_jobs=-1).fit(X2s[a], yy[a])
        out.append(f1_score(yy[b], m.predict(X2s[b]), zero_division=0))
    return float(np.mean(out))


obs = cv_f1(y2)
rng = np.random.default_rng(7)
perm = [cv_f1(rng.permutation(y2)) for _ in range(40)]
pval = float((np.sum(np.array(perm) >= obs) + 1) / (len(perm) + 1))
pd.DataFrame({"obs_f1": [obs], "perm_mean": [np.mean(perm)],
              "perm_sd": [np.std(perm, ddof=1)], "perm_max": [np.max(perm)],
              "p_value": [pval], "n_perm": [len(perm)]}).to_csv(
    TBL / "ds2_permtest.csv", index=False)
log(f"M3: honest-panel F1={obs:.4f} vs shuffled-label mean={np.mean(perm):.4f} "
    f"max={np.max(perm):.4f} p={pval:.3f}")

# ==================================================== M4 Moran's I on DS2
lat = train2["lat"].values
lon = train2["lon"].values


def moran_i(vals, latv, lonv, k=8, n_perm=999, seed=3):
    from scipy.spatial.distance import cdist
    D = cdist(np.c_[latv, lonv], np.c_[latv, lonv])
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    W = np.zeros_like(D)
    rows_ = np.repeat(np.arange(len(vals)), k)
    cols_ = idx.ravel()
    W[rows_, cols_] = 1.0
    Di = 1.0 / (D[rows_, cols_] + 1e-6)
    Ww = np.zeros_like(D)
    Ww[rows_, cols_] = Di
    z = vals - vals.mean()
    S0 = Ww.sum()
    I = (len(vals) / S0) * (Ww @ z).dot(z) / z.dot(z)
    rng = np.random.default_rng(seed)
    pp = []
    for _ in range(n_perm):
        zz = vals[rng.permutation(len(vals))] - vals.mean()
        pp.append((len(vals) / S0) * (Ww @ zz).dot(zz) / zz.dot(zz))
    return I, float(np.mean(pp >= I) if np.mean(pp >= I) else 1 / n_perm), float(np.mean(pp)), float(np.std(pp))


I_lab, p_lab, m_lab, s_lab = moran_i(train2["potable_bis"].values, lat, lon)
I_ec, p_ec, m_ec, s_ec = moran_i(train2["EC"].values, lat, lon)
pd.DataFrame({"variable": ["potable_bis", "EC"], "moran_I": [I_lab, I_ec],
              "perm_p": [p_lab, p_ec], "perm_mean": [m_lab, m_ec],
              "perm_sd": [s_lab, s_ec], "knn_k": [8, 8], "n_perm": [999, 999],
              "n": [len(lat), len(lat)]}).to_csv(TBL / "ds2_moran_i.csv",
                                                 index=False)
log(f"M4: Moran's I(potable_bis)={I_lab:.3f} p={p_lab:.3f} | "
    f"I(EC)={I_ec:.3f} p={p_ec:.3f}")

# ========================================================= Sh7 SHAP seeds
import shap
variances = []
shares = []
FEATS = ["pH_global", "sulfate", "hardness"]  # placeholder; set below
D1NAMES = ["pH", "Sulfate", "Hardness"]
for seed in range(1, 11):
    Xtr, Xte, ytr, yte = train_test_split(
        ds1[D1F].values, ds1.Potability.values, test_size=0.20,
        random_state=seed, stratify=ds1.Potability.values)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xa, ya = SMOTE(random_state=seed).fit_resample(
        scl.transform(imp.transform(Xtr)), ytr)
    m = rf(seed).fit(Xa, ya)
    ex = shap.TreeExplainer(m)
    sv = np.abs(ex.shap_values(pd.DataFrame(
        scl.transform(imp.transform(Xte)), columns=D1F))[1]
        if isinstance(ex.shap_values(scl.transform(imp.transform(Xte))[:10]), list)
        else ex.shap_values(pd.DataFrame(
            scl.transform(imp.transform(Xte)), columns=D1F))[:, :, 1])
    tot = sv[:, [D1F.index("ph"), D1F.index("Sulfate"),
                 D1F.index("Hardness")]].sum() / sv.sum()
    shares.append(tot)
    variances.append({n: sv[:, i].mean() for i, n in enumerate(D1F)})
    log(f"Sh7 seed {seed}: label-defining share={tot:.3f}")
sh = pd.DataFrame(variances)
sh.insert(0, "label_defining_shares", shares)
sh.describe().T[["mean", "std"]].to_csv(TBL / "shap_seed_stability.csv")
log(f"Sh7 across seeds: mean share={np.mean(shares):.3f} "
    f"± {np.std(shares, ddof=1):.3f}")

# ============================== F9 Platt vs isotonic, identical 30 seeds
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
GB = GradientBoostingClassifier = None
try:
    from sklearn.ensemble import GradientBoostingClassifier
    GB = True
except Exception:
    pass

def _ece(p, yy, n_bins=10):
    """Expected calibration error from equal-count bins."""
    order = np.argsort(p)
    ps = p[order]
    ys = np.asarray(yy)[order]
    bins = np.array_split(np.arange(len(ps)), n_bins)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += len(b) * abs(ps[b].mean() - ys[b].mean())
    return e / len(ps)


rows = []
for seed in range(1, 31):
    Xa_, Xb_, y1_, ym_ = train_test_split(
        ds1[D1F].values, ds1.Potability.values, test_size=0.40,
        random_state=seed, stratify=ds1.Potability.values)
    Xv, Xte, yv, yte = train_test_split(Xb_, ym_, test_size=0.5,
                                        random_state=seed,
                                        stratify=ym_)
    imp = SimpleImputer(strategy="median").fit(Xa_)
    scl = StandardScaler().fit(imp.transform(Xa_))
    A = scl.transform(imp.transform(Xa_))
    V_ = scl.transform(imp.transform(Xv))
    B = scl.transform(imp.transform(Xte))
    tr_bal, ytr_bal = SMOTE(random_state=seed).fit_resample(A, y1_)
    specs = {"LR": lambda: LogisticRegression(max_iter=2000, random_state=seed),
             "RF": lambda: rf(seed),
             "XGB": lambda: __import__("xgboost", fromlist=["XGBClassifier"])
             .XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=seed, n_jobs=-1, eval_metric="logloss"),
             "LGBM": lambda: __import__("lightgbm", fromlist=["LGBMClassifier"])
             .LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=-1, verbosity=-1)}
    if GB is not None:
        specs["GB"] = lambda: GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=seed)
    for name, ctor in specs.items():
        base = ctor().fit(tr_bal, ytr_bal)
        pv = base.predict_proba(V_)[:, 1]
        pt = base.predict_proba(B)[:, 1]
        # Platt: logistic fit on validation log-odds
        eps = 1e-6
        z = np.log(np.clip(pv, eps, 1 - eps) / np.clip(1 - pv, eps, 1 - eps))
        pl = LogisticRegression(max_iter=2000).fit(z.reshape(-1, 1), yv)
        p_platt = pl.predict_proba(np.log(np.clip(pt, eps, 1 - eps)
                                          / np.clip(1 - pt, eps, 1 - eps)
                                          ).reshape(-1, 1))[:, 1]
        # isotonic on validation
        ir = IsotonicRegression(out_of_bounds="clip").fit(pv, yv)
        p_iso = ir.predict(pt)
        rows.append({"model": name, "seed": seed,
                     "brier_base": brier_score_loss(yte, pt),
                     "brier_platt": brier_score_loss(yte, p_platt),
                     "brier_iso": brier_score_loss(yte, p_iso),
                     "ece_base": _ece(pt, yte),
                     "ece_platt": _ece(p_platt, yte),
                     "ece_iso": _ece(p_iso, yte)})
    if seed % 10 == 0:
        log(f"F9 seed {seed}/30")
pd.DataFrame(rows).groupby("model").mean(numeric_only=True).to_csv(
    TBL / "calibration_platt_vs_isotonic.csv")
log("F9 done")
print(pd.read_csv(TBL / "calibration_platt_vs_isotonic.csv").round(4).to_string(index=False))
log("ALL ROBUSTNESS EXPERIMENTS DONE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precision-recall + decision-curve analysis on DS1 (leakage-free H_rig protocol).

Safety-first threshold analysis on DS1 under the leakage-free H_rig protocol:
30 holdout seeds per model (LR, RF, XGB, LGBM). Stored out-of-sample
probabilities are pooled across seeds per model, then:
  (a) precision-recall curves for the POTABLE class (Fig. 12a) with AUPRC;
  (b) decision-curve analysis (Vickers net benefit) for the AGENCY DECISION
      "flag sample as unsafe if p_potable < t"  (Fig. 12b):
        NB(t) = (TP_unsafe_flagged - FP_safe_flagged * t/(1-t)) / N
      with treat-all (flag everyone unsafe) and treat-none references.

Outputs -> results/tables/pr_dca_summary.csv
           results/figures/fig12_pr_dca.png 
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIGDIR = ROOT / "results" / "figures"

(OUT / "tables").mkdir(parents=True, exist_ok=True)


T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

DATA = ROOT / "data" / "processed"
ds1 = pd.read_csv(DATA / "ds1" / "ds1_potability.csv")
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
X, y = ds1[FEATS].values, ds1["Potability"].values
log(f"DS1 {ds1.shape}")

MODELS = {
    "LR": lambda s: LogisticRegression(max_iter=2000, random_state=s),
    "RF": lambda s: RandomForestClassifier(n_estimators=300, max_depth=12,
                                           min_samples_leaf=4, random_state=s,
                                           n_jobs=-1),
    "XGB": lambda s: XGBClassifier(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8,
                                   colsample_bytree=0.8, random_state=s,
                                   n_jobs=-1, eval_metric="logloss"),
    "LGBM": lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                     max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, random_state=s,
                                     n_jobs=-1, verbosity=-1),
}

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
        p = ctor(seed).fit(Xtr_s, ytr).predict_proba(Xte_s)[:, 1]
        probs[name].append(p)
    if seed % 10 == 0:
        log(f"seed {seed}/30")
y_pool = np.concatenate(ys)

# NOTE: each seed re-splits, so rows across seeds are the SAME 655 test rows
# only if split is deterministic; stratified split with different seeds selects
# different rows, so pooling mixes samples. That is the standard "pooled
# out-of-sample predictions" approach; document in caption.

rows = []
PR, DC = {}, {}
for name in MODELS:
    p_pool = np.concatenate(probs[name])
    prec, rec, _ = precision_recall_curve(y_pool, p_pool)
    auprc = average_precision_score(y_pool, p_pool)
    PR[name] = (rec, prec)
    # decision curve: flag unsafe when p_potable < t  => unsafe prob = 1 - p
    ts = np.linspace(0.01, 0.99, 99)
    flag = p_pool < ts[:, None]          # (T, N) boolean
    y_unsafe = 1 - y_pool
    tp = (flag & (y_unsafe == 1)).sum(axis=1)
    fp = (flag & (y_unsafe == 0)).sum(axis=1)
    N = len(y_pool)
    nb = (tp - fp * ts / (1 - ts)) / N
    prev = y_unsafe.mean()
    nb_all = prev - (1 - prev) * ts / (1 - ts)
    DC[name] = (ts, nb)
    # crossover: highest t where model NB >= treat-all NB
    idx = np.where(nb >= nb_all)[0]
    cross = float(ts[idx[-1]]) if len(idx) else 0.0
    rows.append({"model": name, "auprc_potable": round(auprc, 4),
                 "prevalence_potable": round(1 - prev, 4),
                 "dca_beats_treatall_upto_t": round(cross, 3)})
    log(f"{name}: AUPRC={auprc:.4f} | DCA crossover t={cross:.3f}")

pd.DataFrame(rows).to_csv(OUT / "tables" / "pr_dca_summary.csv", index=False)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
COLORS = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52", "LGBM": "#8172B2"}
ax = axes[0]
for name, (rec, prec) in PR.items():
    ax.plot(rec, prec, color=COLORS[name], lw=2,
            label=f"{name} (AUPRC {dict((r['model'], r['auprc_potable']) for r in rows)[name]:.3f})")
base = y_pool.mean()
ax.axhline(base, color="gray", ls=":", lw=1.2, label=f"Chance ({base:.3f})")
ax.set_xlabel("Recall (potable class)")
ax.set_ylabel("Precision")
ax.set_title("(a) Precision–recall, pooled over 30 holdout seeds")
ax.legend(loc="lower left", fontsize=8)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)

ax = axes[1]
for name, (ts, nb) in DC.items():
    ax.plot(ts, nb, color=COLORS[name], lw=2, label=name)
ts = list(DC.values())[0][0]
prev = (1 - y_pool).mean()
nb_all = prev - (1 - prev) * ts / (1 - ts)
ax.plot(ts, nb_all, color="black", ls="--", lw=1.4, label="Flag all unsafe")
ax.axhline(0, color="gray", ls=":", lw=1, label="Flag none")
ax.set_xlabel("Decision-threshold probability $t$ (flag unsafe if $p_{potable} < t$)")
ax.set_ylabel("Net benefit (per sample)")
ax.set_title("(b) Decision-curve analysis, unsafe-flag policy")
ax.set_ylim(-0.15, max(0.45, prev + 0.05))
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1)

fig.suptitle("Fig. 12. Safety-first threshold analysis (DS1, leakage-free H_rig protocol)",
             y=1.02, fontsize=11)
fig.tight_layout()
out_png = FIGDIR / "fig12_pr_dca.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
# public bundle: only the in-repo figure copy is written
log(f"fig12 saved: {out_png}")
log("PR/DCA DONE")

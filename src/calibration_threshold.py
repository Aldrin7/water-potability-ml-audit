#!/usr/bin/env python3
"""RQ3b Calibration & decision-threshold analysis for water potability."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve, IsotonicRegression
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "logs").mkdir(parents=True, exist_ok=True)
SEEDS = list(range(1, 31))
FEATS = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
TARGET = "Potability"

def models(seed):
    return {
        "LR": LogisticRegression(max_iter=2000, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss"),
    }

def prep_fit(Xtr):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    return imp, scl

def ece(y_true, proba, n_bins=10):
    fp, mp = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(fp - mp)))

def main():
    log_lines = []
    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        log_lines.append(line)
        print(line, flush=True)
    
    df = pd.read_csv(PROC / "ds1" / "ds1_potability.csv")
    base_rate = df[TARGET].mean()
    base_rate_brier = 1 - 2 * base_rate * (1 - base_rate)
    log(f"Loaded DS1 {df.shape}, target rate={base_rate:.4f}, base-rate Brier={base_rate_brier:.4f}")
    
    rows = []
    for seed in SEEDS:
        X, y = df[FEATS].values, df[TARGET].values
        X_tr, X_rem, y_tr, y_rem = train_test_split(X, y, test_size=0.40, random_state=seed, stratify=y)
        X_val, X_te, y_val, y_te = train_test_split(X_rem, y_rem, test_size=0.50, random_state=seed, stratify=y_rem)
        imp, scl = prep_fit(X_tr)
        Xtr_s = scl.transform(imp.transform(X_tr))
        Xval_s = scl.transform(imp.transform(X_val))
        Xte_s = scl.transform(imp.transform(X_te))
        Xtr_b, ytr_b = SMOTE(random_state=seed).fit_resample(Xtr_s, y_tr)
        
        for name, ctor in models(seed).items():
            m = ctor.fit(Xtr_b, ytr_b)
            proba_base = m.predict_proba(Xte_s)[:, 1]
            ypred_base = (proba_base >= 0.5).astype(int)
            brier_base = brier_score_loss(y_te, proba_base)
            ece_base = ece(y_te, proba_base)
            auc_base = roc_auc_score(y_te, proba_base)
            
            # Fix 1: Isotonic calibration on val
            val_proba = m.predict_proba(Xval_s)[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            cal.fit(val_proba, y_val)
            proba_cal = cal.predict(proba_base)
            val_proba_cal = cal.predict(val_proba)  # calibrated val probs (v2)
            ypred_cal = (proba_cal >= 0.5).astype(int)
            brier_cal = brier_score_loss(y_te, proba_cal)
            ece_cal = ece(y_te, proba_cal)
            
            # Fix 2 (CORRECTED v2): policy threshold for >=90% recall of the
            # non-potable class (0). Selected on the VALIDATION set (no test
            # leakage). Safety-first rule: flag a sample UNSAFE when its
            # calibrated P(potable) is BELOW the threshold.
            # (v1 bug: threshold derived from the descending-sorted TEST curve
            #  and applied as potable-if-proba>=thr, labelling ~97% of samples
            #  potable and catching <7% of unsafe water - results invalid.)
            rec0_at_90 = 0.5
            order = np.argsort(val_proba_cal)
            y_sorted = y_val[order]
            proba_sorted = val_proba_cal[order]
            cum_neg = np.cumsum(y_sorted == 0)
            total_neg = int(np.sum(y_val == 0))
            if total_neg > 0:
                rec_curve = cum_neg / total_neg
                idx_90 = np.where(rec_curve >= 0.90)[0]
                if len(idx_90) > 0:
                    rec0_at_90 = float(proba_sorted[idx_90[0]])
            
            yp_pol = np.where(proba_cal < rec0_at_90, 0, 1)
            rows.append({
                "seed": seed, "model": name,
                "f1_base": f1_score(y_te, ypred_base, zero_division=0),
                "acc_base": accuracy_score(y_te, ypred_base),
                "brier_base": brier_base, "ece_base": ece_base, "auc_base": auc_base,
                "f1_cal": f1_score(y_te, ypred_cal, zero_division=0),
                "brier_cal": brier_cal, "ece_cal": ece_cal,
                "auc_cal": roc_auc_score(y_te, proba_cal),
                "thr_policy": rec0_at_90,
                "f1_unsafe_at_policy": f1_score(y_te, yp_pol, pos_label=0, zero_division=0),
                "precision_unsafe_at_policy": precision_score(y_te, yp_pol, pos_label=0, zero_division=0),
                "recall_unsafe_at_policy": recall_score(y_te, yp_pol, pos_label=0, zero_division=0),
                "acc_at_policy": accuracy_score(y_te, yp_pol),
                "balacc_at_policy": balanced_accuracy_score(y_te, yp_pol),
                "f1_compliant_at_policy": f1_score(y_te, yp_pol, pos_label=1, zero_division=0),
                "precision_compliant_at_policy": precision_score(y_te, yp_pol, pos_label=1, zero_division=0),
                "recall_compliant_at_policy": recall_score(y_te, yp_pol, pos_label=1, zero_division=0),
            })
            if seed == 1 and name in ["LR", "RF"]:
                log(f"  seed 1 {name}: brier {brier_base:.4f}->{brier_cal:.4f}, ece {ece_base:.4f}->{ece_cal:.4f}, thr={rec0_at_90:.2f}")
        if seed % 10 == 0:
            log(f"Done seed {seed}/{SEEDS[-1]}")
    
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tables" / "calibration_threshold.csv", index=False)
    
    # Summary
    summary_rows = []
    for name in res["model"].unique():
        sub = res[res.model == name]
        r = {"model": name, "n_seeds": len(sub)}
        for col in [c for c in res.columns if c not in ("seed", "model")]:
            r[f"{col}_mean"] = sub[col].mean()
            r[f"{col}_sd"] = sub[col].std(ddof=1)
        summary_rows.append(r)
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUT / "tables" / "calibration_threshold_summary.csv", index=False)
    
    bb = summ["brier_base_mean"].mean()
    bc = summ["brier_cal_mean"].mean()
    eb = summ["ece_base_mean"].mean()
    ec = summ["ece_cal_mean"].mean()
    imp_pct = (bb - bc) / bb * 100
    
    report = [
        "# CALIBRATION & DECISION-THRESHOLD ANALYSIS",
        f"Protocol: DS1, {len(SEEDS)} seeds, 60/20/20 split (train/val-cal/test)",
        f"Base rate={base_rate:.4f}, base-rate Brier={base_rate_brier:.4f}",
        "",
        "| Model | Brier base->cal | ECE base->cal | F1 base->cal | Policy thr | F1_unsafe | Recall_unsafe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in summ.model:
        s = summ[summ.model == name].iloc[0]
        report.append(f"| {name} | {s.brier_base_mean:.4f}->{s.brier_cal_mean:.4f} | {s.ece_base_mean:.4f}->{s.ece_cal_mean:.4f} | {s.f1_base_mean:.4f}->{s.f1_cal_mean:.4f} | {s.thr_policy_mean:.2f} | {s.f1_unsafe_at_policy_mean:.4f} | {s.recall_unsafe_at_policy_mean:.4f} |")
    report += [
        "",
        f"1. BASELINE: Brier={bb:.4f} avg ({(base_rate_brier-bb)/base_rate_brier*100:.1f}% better than base rate)",
        f"2. ISOTONIC CAL: Brier {bb:.4f}->{bc:.4f} ({imp_pct:.1f}% reduction); ECE {eb:.4f}->{ec:.4f}",
        f"3. POLICY THRESHOLD (v2, val-selected, flag unsafe if P(pot)<thr): {summ.thr_policy_mean.mean():.2f}",
        f"   unsafe recall={summ.recall_unsafe_at_policy_mean.mean():.4f}, F1_unsafe={summ.f1_unsafe_at_policy_mean.mean():.4f}",
        f"4. COST-BENEFIT: compliant precision at policy threshold = {summ.precision_compliant_at_policy_mean.mean():.4f}",
        "", "## Execution log", "```",
    ]
    report += log_lines + ["```"]
    (OUT / "logs" / "calibration_threshold_report.md").write_text("\n".join(report))
    log("ALL CALIBRATION ANALYSIS DONE")
    print(f"\nDone. Summary: Brier {bb:.4f}->{bc:.4f} ({imp_pct:.1f}%), threshold={summ.thr_policy_mean.mean():.2f}")

if __name__ == "__main__":
    main()

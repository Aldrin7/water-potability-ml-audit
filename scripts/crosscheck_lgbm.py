# -*- coding: utf-8 -*-
"""Cross-check LGBM vs local RF/GB protocol audit + LGBM calibration."""
import pandas as pd

la = pd.read_csv(f"{ROOT}/results/tables/protocol_audit_lgbm.csv")
lo = pd.read_csv(f"{ROOT}/results/tables/protocol_audit.csv")

print("=== LGBM vs RF/GB protocol audit ===")
for arm in ["H_rig", "H_nosmote", "CV_rig", "CV_leak", "CV_patel"]:
    lg = la[la.arm == arm].iloc[0]
    rf = lo[(lo.arm == arm) & (lo.model == "RF")]
    gb = lo[(lo.arm == arm) & (lo.model == "GB")]
    s = f"{arm:10s} LGBM acc={lg.acc_mean:.4f} f1={lg.f1_mean:.4f}"
    if len(rf):
        s += f" | RF acc={rf.acc_mean.iloc[0]:.4f} f1={rf.f1_mean.iloc[0]:.4f}"
    if len(gb):
        s += f" | GB f1={gb.f1_mean.iloc[0]:.4f}"
    print(s)

print()
print("=== LGBM calibration summary ===")
ck = pd.read_csv(f"{ROOT}/results/tables/calibration_threshold_kaggle_summary.csv",
                 index_col=0)
cols = ["brier_base", "brier_cal", "ece_base", "ece_cal", "auc_cal",
        "thr_policy", "recall_unsafe_at_policy", "precision_unsafe_at_policy"]
for i in ck.index:
    print(i, "|", " ".join(f"{c}={ck.loc[i, (c, 'mean')]:.4f}" for c in cols))
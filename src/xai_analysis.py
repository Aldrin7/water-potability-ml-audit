#!/usr/bin/env python3
"""XAI analysis — SHAP feature importance for DS1 (water potability).

Objective:
  Quantify which physicochemical features drive the Random Forest's
  potability predictions on DS1, and test whether the BIS-label-defining
  ions (pH, sulfate, etc.) dominate SHAP values — a quantitative
  confirmation of the label-circularity argument in Section 3.3.

Protocol (matches H_rig in §2.2):
  80/20 stratified split; median imputation + standard scaling fit on
  train only; SMOTE on train only; RF (300 trees, max_depth=12,
  random_state=42). SHAP TreeExplainer (exact for RF) on the held-out
  test set.

Outputs:
  results/tables/xai_shap_summary.csv  — per-feature mean |SHAP|, rank
  results/figures/fig9_xai_shap.png     — horizontal bar chart
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import shap

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

# BIS 10500:2012 potability-defining parameters on DS1
BIS_DEFINITING = {"ph", "sulfate", "hardness"}

df = pd.read_csv(ROOT / "data/processed/ds1/ds1_potability.csv")
X = df[FEATS].values
y = df["Potability"].values

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
imp = SimpleImputer(strategy="median").fit(Xtr)
scl = StandardScaler().fit(imp.transform(Xtr))
Xtr_s = scl.transform(imp.transform(Xtr))
Xte_s = scl.transform(imp.transform(Xte))
Xtr_s, ytr = SMOTE(random_state=42).fit_resample(Xtr_s, ytr)

model = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=4,
    random_state=42, n_jobs=-1
).fit(Xtr_s, ytr)

# SHAP (TreeExplainer — exact for RF, no sampling error)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xte_s)

# SHAP 0.49+: binary RF returns shape (n_samples, n_features, n_classes)
# Older versions return [class_0_array, class_1_array]
# Even older: single array (n_samples, n_features)
if isinstance(shap_values, list):
    sv = shap_values[1]  # positive class (potable)
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]  # take positive class
else:
    sv = shap_values  # already (n_samples, n_features)
mean_abs_shap = np.abs(sv).mean(axis=0)
feature_names = FEATS

ranking = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
ranking["rank"] = ranking.index + 1
ranking["bis_defining"] = ranking["feature"].apply(lambda f: f.lower() in BIS_DEFINITING)

OUT = ROOT / "results"
ranking.to_csv(OUT / "tables" / "xai_shap_summary.csv", index=False)

# --- Chart ---
bis_color = ["crimson" if f else "steelblue"
             for f in ranking["bis_defining"]]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(ranking["feature"][::-1], ranking["mean_abs_shap"][::-1],
        color=bis_color[::-1])
ax.set_xlabel("Mean |SHAP value| (test set, n = 656)")
ax.set_title("SHAP feature importance — DS1 Random Forest\n"
             "(crimson = BIS 10500 label-defining parameters)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig9_xai_shap.png", dpi=300)
plt.close(fig)

# --- Summary stats ---
top3 = ranking.head(3)["feature"].tolist()
bis_share = ranking[ranking["bis_defining"]]["mean_abs_shap"].sum()
nonbis_share = ranking[~ranking["bis_defining"]]["mean_abs_shap"].sum()
total = bis_share + nonbis_share

print("=== SHAP feature importance (DS1 RF) ===")
print(ranking.to_string(index=False))
print(f"\nTop-3 features: {top3}")
print(f"BIS-defining ions share of total |SHAP|: {bis_share/total:.1%}")
print(f"Non-defining ions share: {nonbis_share/total:.1%}")
print(f"BIS-defining ions in top-3: "
      f"{sum(1 for f in top3 if f.lower() in BIS_DEFINITING)}/3")
print("DONE")

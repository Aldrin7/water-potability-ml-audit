#!/usr/bin/env python3
"""M9 — Test-time robustness: measurement-error noise sensitivity (DS1, RF).

For each documented sensor measurement error, add Gaussian noise of that
magnitude to the test set and measure F1/accuracy shift. Uses published
(rational) sensor tolerances for the nine DS1 parameters. Outputs a table
so robustness claims trace to a run.

v2 (NaN fix): the noise scale is the SD of OBSERVED training values only
(np.nanstd). v1 used np.std on the raw columns, which returns NaN for
ph/Sulfate/Trihalomethanes (they contain missing values); the downstream
median imputer then silently replaced those three features by constants,
so v1 measured a feature ablation (F1 -> 0.264), not noise sensitivity.
Missing test values remain missing: noise perturbs measured readings only.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FEATS = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
         "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
# Nominal sensor tolerance applied as absolute SD on the test inputs,
# expressed as a FRACTION of the feature's training SD (scale-relative).
# (ph: ~0.1 pH unit electrode; Conductivity: ~2%; other bench params ~5%.)
NOISE_FRAC = {"ph": 0.30, "Hardness": 0.20, "Solids": 0.10,
              "Chloramines": 0.20, "Sulfate": 0.20, "Conductivity": 0.10,
              "Organic_carbon": 0.20, "Trihalomethanes": 0.20,
              "Turbidity": 0.20}
SEEDS = list(range(1, 16))


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "ds1" / "ds1_potability.csv")
    X, y = df[FEATS].values, df["Potability"].values
    rows = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y)
        imp = SimpleImputer(strategy="median").fit(Xtr)
        scl = StandardScaler().fit(imp.transform(Xtr))
        Xtr_s = scl.transform(imp.transform(Xtr))
        Xte_s = scl.transform(imp.transform(Xte))
        Xtr_s, ytr = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
        m = RandomForestClassifier(n_estimators=300, max_depth=12,
                                   min_samples_leaf=4, random_state=seed,
                                   n_jobs=-1).fit(Xtr_s, ytr)
        # clean test
        f1_clean = f1_score(yte, m.predict(Xte_s), zero_division=0)
        acc_clean = accuracy_score(yte, m.predict(Xte_s))
        # noise-added test (absolute SD = frac * training SD)
        # v2 fix: SD of OBSERVED training values only. v1's np.std returned
        # NaN for ph/Sulfate/Trihalomethanes (missing data), and the median
        # imputer downstream turned that into replacement of those columns
        # by training medians -- a feature ablation, not a noise test.
        rng = np.random.RandomState(seed)
        Xte_noisy = Xte.astype(float).copy()
        for j, f in enumerate(FEATS):
            sd = np.nanstd(Xtr[:, j])
            Xte_noisy[:, j] += rng.normal(0, NOISE_FRAC[f] * sd, size=len(Xte))
        Xte_noisy_s = scl.transform(imp.transform(Xte_noisy))
        f1_noisy = f1_score(yte, m.predict(Xte_noisy_s), zero_division=0)
        acc_noisy = accuracy_score(yte, m.predict(Xte_noisy_s))
        rows.append({"seed": seed, "f1_clean": f1_clean, "f1_noisy": f1_noisy,
                     "acc_clean": acc_clean, "acc_noisy": acc_noisy})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "tables" / "m9_noise_sensitivity.csv", index=False)
    print(f"clean F1 {d.f1_clean.mean():.4f}+-{d.f1_clean.std(ddof=1):.4f} | "
          f"noisy F1 {d.f1_noisy.mean():.4f}+-{d.f1_noisy.std(ddof=1):.4f}")
    print(f"clean acc {d.acc_clean.mean():.4f} | "
          f"noisy acc {d.acc_noisy.mean():.4f}")
    rep = ["# M9 — MEASUREMENT-ERROR NOISE SENSITIVITY (RF, DS1, 15 seeds)",
           "",
           "Noise added to held-out test inputs with SD = nominal sensor",
           "tolerance as fraction of each feature's OBSERVED training SD (v2: nanstd; missing readings stay missing), per feature:",
           str(NOISE_FRAC), "",
           f"- Clean test F1: {d.f1_clean.mean():.4f} ± "
           f"{d.f1_clean.std(ddof=1):.4f}",
           f"- Noisy test F1: {d.f1_noisy.mean():.4f} ± "
           f"{d.f1_noisy.std(ddof=1):.4f}",
           f"- F1 shift: {d.f1_noisy.mean()-d.f1_clean.mean():+.4f}",
           "",
           "Interpretation: if F1 shift > ~0.02, the model is operating on",
           "features whose measurement precision matters; deployment",
           "comparisons must use realistic sensor error budgets."]
    (OUT / "logs" / "m9_noise_sensitivity.md").write_text(
        "\n".join(rep), encoding="utf-8")


if __name__ == "__main__":
    main()
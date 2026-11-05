import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FEATS = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
NOISE = {'ph':0.30,'Hardness':0.20,'Solids':0.10,'Chloramines':0.20,'Sulfate':0.20,'Conductivity':0.10,'Organic_carbon':0.20,'Trihalomethanes':0.20,'Turbidity':0.20}
df = pd.read_csv(ROOT / "data" / "processed" / "ds1" / "ds1_potability.csv")
X, y = df[FEATS].values, df["Potability"].values
t0 = time.time()
rows = []
for seed in range(1,16):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    scl = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s = scl.transform(imp.transform(Xtr))
    Xte_s = scl.transform(imp.transform(Xte))
    Xtr_s, ytr = SMOTE(random_state=seed).fit_resample(Xtr_s, ytr)
    m = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=4, random_state=seed, n_jobs=-1).fit(Xtr_s, ytr)
    f1_clean = f1_score(yte, m.predict(Xte_s), zero_division=0)
    for j, f in enumerate(FEATS):
        rng = np.random.RandomState(seed)
        Xte_n = Xte.astype(float).copy()
        sd = np.nanstd(Xtr[:, j])  # v2: observed training values only (v1 np.std -> NaN for ph/Sulfate/Trihalomethanes -> accidental ablation)
        Xte_n[:, j] += rng.normal(0, NOISE[f] * sd, size=len(Xte))
        f1_noisy = f1_score(yte, m.predict(scl.transform(imp.transform(Xte_n))), zero_division=0)
        rows.append({'kind': 'noise', 'feature': f, 'seed': seed, 'f1_clean': f1_clean, 'f1_noisy': f1_noisy, 'shift': f1_noisy - f1_clean})
        Xte_a = Xte.astype(float).copy()
        Xte_a[:, j] = imp.statistics_[j]  # ablation: measurement dropped -> training-median constant
        f1_abl = f1_score(yte, m.predict(scl.transform(imp.transform(Xte_a))), zero_division=0)
        rows.append({'kind': 'ablation', 'feature': f, 'seed': seed, 'f1_clean': f1_clean, 'f1_noisy': f1_abl, 'shift': f1_abl - f1_clean})
    print(f"Seed {seed} done {time.time()-t0:.1f}s", flush=True)
d = pd.DataFrame(rows)
d.to_csv(OUT / "tables" / "m9_per_feature_seeds.csv", index=False)  # per-seed trace for stats
d2 = d.groupby(['kind','feature']).agg(f1_clean=('f1_clean','mean'), f1_noisy=('f1_noisy','mean'), shift=('shift','mean'), shift_sd=('shift','std')).reset_index().sort_values(['kind','shift'])
d2.to_csv(OUT / "tables" / "m9_per_feature.csv", index=False)
print(d2.to_string(index=False))
print("DONE")

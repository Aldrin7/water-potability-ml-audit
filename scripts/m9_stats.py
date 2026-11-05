#!/usr/bin/env python3
"""Summary statistics for M9 noise-sensitivity CSVs.

Usage: python scripts/m9_stats.py <csv> [<csv> ...]

Handles three schemas:
  - aggregate local:  seed,f1_clean,f1_noisy,acc_clean,acc_noisy
  - Kaggle fixed:    model,seed,f1_clean,f1_noisy,f1_drop
  - per-feature:     feature,seed,f1_clean,f1_noisy,shift
Prints clean/noisy means+SD, mean shift+SD, paired t-test and dz.
"""
import sys

import pandas as pd
from scipy import stats


def report(sub, label):
    c = sub["f1_clean"]
    n = sub["f1_noisy"]
    sh = n - c
    t = stats.ttest_rel(n, c)
    sd = sh.std(ddof=1)
    dz = sh.mean() / sd if sd and sd > 0 else float("nan")
    print(f"[{label}] n={len(sub)} seeds")
    print(f"  clean {c.mean():.4f} +- {c.std(ddof=1):.4f}")
    print(f"  noisy {n.mean():.4f} +- {n.std(ddof=1):.4f}")
    print(f"  shift (noisy-clean) {sh.mean():+.4f} +- {sd:.4f}")
    print(f"  paired t={t.statistic:+.3f}  p={t.pvalue:.4g}  dz={dz:+.2f}")


def main():
    for path in sys.argv[1:]:
        d = pd.read_csv(path)
        print(f"===== {path}")
        if "kind" in d.columns and "feature" in d.columns:
            keys = ["kind", "feature"]
        elif "model" in d.columns:
            keys = ["model"]
        elif "feature" in d.columns:
            keys = ["feature"]
        else:
            keys = []
        groups = d.groupby(keys) if keys else [(None, d)]
        for k, sub in groups:
            label = "/".join(map(str, k)) if isinstance(k, tuple) else str(k)
            report(sub, label)





if __name__ == "__main__":
    main()

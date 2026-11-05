# water-potability-ml-audit

Code, datasets, and results for a controlled audit of machine-learning
water-quality screening: how evaluation choices - splitting, imputation,
resampling, and label definition - inflate (or do not inflate) reported
performance on real monitoring data.

Archived on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23973485.svg)](https://doi.org/10.5281/zenodo.23973485)

## What is measured

- **Protocol gap decomposition** on a public potability benchmark (3,276 samples):
  holdout vs cross-validation, per-fold vs global preprocessing, SMOTE before vs
  after splitting, published-style protocols
- **Overfitting diagnostics**: train/test gap, capacity sweep, learning curve
- **Label circularity** on a state groundwater panel (1,106 station-years,
  2018-2020): labels defined as threshold functions of the input ions vs the
  non-label-defining panel, spatial/temporal cross-validation,
  distance-aware validation, Moran's I and label-permutation controls
- **Deployment tiers** on a national surface-water survey (620 stations):
  feature-cost tiers for compliance screening
- **Calibration** and safety-first operating thresholds (Brier/ECE, isotonic,
  Platt scaling, precision-recall and decision-curve analysis)
- **Sensor-noise sensitivity** and per-feature measurement loss
- **SHAP explainability** with seed-stability checks
- **Green-compute audit**: training energy, CO2, inference latency, model size

## Repository layout

```
data/
  raw/                    original archives as downloaded (Kaggle mirrors)
  processed/              verified Gate-A outputs (ds1, ds2, ds3 CSVs)
src/                      core analysis modules (one script per research question)
scripts/                  auxiliary/robustness pipelines and diagnostics
kaggle/                   cloud re-run kernels (LGBM cross-check)
results/
  tables/                 all measured result tables (CSV)
  figures/                final figures (300 DPI PNG, fig1-fig12)
requirements.txt          Python dependencies
LICENSE                   MIT license for the code
```

## Datasets

| Tag | Source | Records | Terms |
|-----|--------|---------|-------|
| DS1 | Kaggle: adityakadiwal/water-potability | 3,276 rows | CC0-1.0 (public domain) |
| DS2 | Telangana Ground Water Department, 2018-2020 (public Kaggle mirror: sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data) | 1,106 station-years | government monitoring data |
| DS3 | CPCB water-body monitoring, 2021 (Kaggle mirror: balabaskar/water-quality-data-india) | 620 stations | government monitoring data |

Raw archives and the processed, label-checked versions are both included.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+; LightGBM 4.6.0 (the Kaggle-era 3.x behavior differs slightly and
is documented in the cross-check script).

## Reproducing

Each script writes its outputs into `results/tables/` or `results/figures/`
with fixed random seeds recorded in the code:

1. `scripts/download_data.py` - fetch the three raw datasets
2. `scripts/gateA_data.py` - verify/clean raw data into `data/processed/`
3. `src/protocol_audit.py` - protocol-gap decomposition (DS1)
4. `src/overfitting_diagnostics.py` - capacity, learning curve, gaps
5. `src/ds2_benchmark.py` - DS2 spatial/temporal screening
6. `src/ds3_deployment.py` - DS3 deployment tiers
7. `src/calibration_threshold.py` + `scripts/decision_curve_analysis.py` -
   calibration, safety-first thresholds, PR + decision-curve analysis
8. `src/m9_noise_sensitivity.py`, `src/m9_per_feature.py` - noise/loss robustness
9. `src/xai_analysis.py` - SHAP attribution (+ seed checks in
   `scripts/protocol_robustness.py`)
10. `src/green_metrics.py` (+ `scripts/green_metrics_quiet.py` guard) -
    energy, CO2, latency, model size
11. `scripts/expanded_panels.py`, `scripts/distance_validation.py` - DS2
    extended honesty panels and distance-aware validation
12. `scripts/lgbm_local_rerun.py`, `scripts/lgbm_determinism.py`,
    `scripts/lgbm_overfit_local.py`, `scripts/ds3_lgbm_local.py` - LGBM
    cross-environment checks
13. `src/make_figures.py` + `scripts/green_figures.py` - regenerate all figures

`kaggle/kaggle_benchmark.py` re-runs the full benchmark in the cloud
(Kaggle kernel) for environment-independence cross-checking.

## Results

Every table in the paper regenerates from a named script and fixed seeds.
Result CSVs live in `results/tables/`, final figures in `results/figures/`.

## License & dataset terms

Code: MIT (see `LICENSE`). The source datasets keep their original provider
terms: DS1 is CC0-1.0; DS2/DS3 are derived from Indian government monitoring
data published through open portals or Kaggle mirrors.

## Contact

Aldrin Manon - aldrinmanon7@gmail.com
ICFAI Tech School, The ICFAI University, Dehradun, India

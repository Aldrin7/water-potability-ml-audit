#!/usr/bin/env python3
"""Download the three study datasets from Kaggle via kagglehub.

Datasets (all public, verified by search; provenance documented in plan):
  DS1: adityakadiwal/water-potability        (drinking water potability, 3,276 rows)
  DS2: sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data (groundwater monitoring)
  DS3: balabaskar/water-quality-data-india   (surface water bodies, time series)
"""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import kagglehub  # noqa: E402

DEST = f"{ROOT}/data/raw"
DATASETS = [
    "adityakadiwal/water-potability",
    "sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data",
    "balabaskar/water-quality-data-india",
]

os.makedirs(DEST, exist_ok=True)

for slug in DATASETS:
    print(f"Downloading {slug} ...", flush=True)
    try:
        path = kagglehub.dataset_download(slug)
        print(f"  OK -> {path}", flush=True)
        # Copy into project data/raw/<short-name>/
        short = slug.split("/")[-1]
        target = os.path.join(DEST, short)
        if os.path.abspath(path) != os.path.abspath(target):
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(path, target)
        print(f"  Copied to {target}", flush=True)
        for root, _, files in os.walk(target):
            for f in files:
                fp = os.path.join(root, f)
                print(f"    {fp} ({os.path.getsize(fp)} bytes)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {slug}: {e!r}", flush=True)

print("DONE", flush=True)

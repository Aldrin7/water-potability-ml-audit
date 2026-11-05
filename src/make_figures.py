#!/usr/bin/env python3
"""Generate all paper figures from the actual result CSVs.

Every figure reads a results/tables/*.csv produced by executed code.
No synthetic data, no hand-drawn curves.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
F = ROOT / "results" / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})


def fig1_protocol_audit():
    """Accuracy and F1 by arm (DS1) with SD error bars."""
    df = pd.read_csv(T / "protocol_audit.csv")
    df = df[df.model == "RF"].sort_values("acc_mean")
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, df.acc_mean, w, yerr=df.acc_sd, capsize=3,
           label="Accuracy", color="#4C72B0")
    ax.bar(x + w/2, df.f1_mean, w, yerr=df.f1_sd, capsize=3,
           label="F1 (potable)", color="#DD8452")
    ax.set_xticks(x, df.arm, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("DS1: Protocol arm comparison (Random Forest)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig1_protocol_audit.png")
    plt.close(fig)


def fig2_capacity_curve():
    """RF capacity sweep: train vs test F1 by max_depth (overfitting elbow)."""
    df = pd.read_csv(T / "d2_capacity_sweep_summary.csv")
    order = ["4", "6", "8", "10", "12", "16", "None"]
    df = df.set_index("max_depth").reindex(order).reset_index()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, df.train_f1, "-o", label="Train F1")
    ax.plot(x, df.test_f1, "-s", label="Test F1")
    ax.axvline(order.index("12"), color="gray", ls="--", lw=1)
    ax.annotate("elbow @ depth 12", xy=(order.index("12"), 0.55),
                xytext=(order.index("12")-0.6, 0.82),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_xticks(x, df.max_depth)
    ax.set_xlabel("RF max_depth")
    ax.set_ylabel("F1")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig2_capacity_curve.png")
    plt.close(fig)


def fig3_learning_curve():
    df = pd.read_csv(T / "d3_learning_curve_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(df.frac, df.test_f1, yerr=df.sd, capsize=4, marker="o")
    ax.set_xlabel("Training fraction")
    ax.set_ylabel("Test F1")
    ax.set_title("DS1 RF learning curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig3_learning_curve.png")
    plt.close(fig)


def fig4_ds2_generalization():
    df = pd.read_csv(T / "d4_ds2_generalization.csv")
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w, df.random_cv_f1, w, label="Random CV (leaky)", color="#4C72B0")
    ax.bar(x, df.spatial_cv_f1, w, label="Spatial CV (by village)", color="#DD8452")
    ax.bar(x + w, df.temporal_2020_f1, w, label="Temporal 2020", color="#55A868")
    ax.set_xticks(x, df.model)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("F1")
    ax.set_title("DS2 non-BIS: how evaluation split changes F1")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig4_ds2_generalization.png")
    plt.close(fig)


def fig5_ds3_deployment():
    df = pd.read_csv(T / "ds3_deployment.csv")
    arms = {"cheap_sensor_T0": "Cheap sensor (T0)",
            "do_probe_T1": "DO probe (T1)",
            "full_params_circular": "Full params (circular)"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (arm, label) in enumerate(arms.items()):
        sub = df[df.arm == arm]
        ax.errorbar(sub.f1_compliant_mean, sub.model, xerr=sub.f1_compliant_sd,
                    fmt="o", label=label, capsize=3)
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("F1 (Class-B compliant)")
    ax.set_title("DS3: feature-cost tier vs performance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig5_ds3_deployment.png")
    plt.close(fig)


def fig6_reliability():
    r = pd.read_csv(T / "d5_reliability_curve.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(r.mean_predicted, r.fraction_positive, "-o", color="#C44E52",
            label="RF (seed 42)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive")
    ax.set_title("DS1 reliability curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig6_reliability_curve.png")
    plt.close(fig)


def fig7_calibration_comparison():
    """Brier score and ECE before/after isotonic calibration."""
    df = pd.read_csv(T / "calibration_threshold_summary.csv")
    models = df.model
    x = np.arange(len(df))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(x - w/2, df.brier_base_mean, w, yerr=df.brier_base_sd,
            capsize=3, label="Before", color="#4C72B0")
    ax1.bar(x + w/2, df.brier_cal_mean, w, yerr=df.brier_cal_sd,
            capsize=3, label="After isotonic", color="#DD8452")
    ax1.set_xticks(x, models)
    ax1.set_ylabel("Brier score (lower = better)")
    ax1.set_title("Calibration: Brier score")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x - w/2, df.ece_base_mean, w, yerr=df.ece_base_sd,
            capsize=3, label="Before", color="#4C72B0")
    ax2.bar(x + w/2, df.ece_cal_mean, w, yerr=df.ece_cal_sd,
            capsize=3, label="After isotonic", color="#DD8452")
    ax2.set_xticks(x, models)
    ax2.set_ylabel("ECE (lower = better)")
    ax2.set_title("Calibration: Expected Calibration Error")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig7_calibration_comparison.png")
    plt.close(fig)


def fig8_noise_sensitivity():
    """M9 v2: per-feature sensor-noise sensitivity (RF, DS1, 15 seeds)."""
    df = pd.read_csv(T / "m9_per_feature.csv")
    df = df.rename(columns={"shift": "f1_shift", "shift_sd": "f1_shift_sd"})
    if "kind" in df.columns:
        df = df[df["kind"] == "noise"]
        df = df.sort_values("f1_shift", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df.feature, df.f1_shift, xerr=df.f1_shift_sd, capsize=3,
            color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.axvline(-0.05, color="gray", ls="--", lw=1, label="5% F1 loss threshold")
    ax.set_xlabel("F1 shift under sensor-grade noise (negative = worse)")
    ax.set_title("M9: Per-feature noise sensitivity (RF, DS1)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig8_noise_sensitivity.png")
    plt.close(fig)


def fig9_policy_threshold():
    """Recall at policy threshold for unsafe-water detection."""
    df = pd.read_csv(T / "calibration_threshold_summary.csv")
    models = df.model
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, df.recall_unsafe_at_policy_mean, w,
           yerr=df.recall_unsafe_at_policy_sd, capsize=3,
           label="Recall (unsafe)", color="#C44E52")
    ax.bar(x + w/2, df.f1_unsafe_at_policy_mean, w,
           yerr=df.f1_unsafe_at_policy_sd, capsize=3,
           label="F1 (unsafe)", color="#4C72B0")
    ax.set_xticks(x, models)
    ax.set_ylabel("Score")
    ax.set_title("Policy threshold: unsafe-water recall vs F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "fig9_policy_threshold.png")
    plt.close(fig)


def main():
    fig1_protocol_audit()
    fig2_capacity_curve()
    fig3_learning_curve()
    fig4_ds2_generalization()
    fig5_ds3_deployment()
    fig6_reliability()
    fig7_calibration_comparison()
    fig8_noise_sensitivity()
    fig9_policy_threshold()
    print("Figures written to", F)


if __name__ == "__main__":
    main()
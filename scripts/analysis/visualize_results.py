#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = None
OUTPUT = None

METRICS = {
    "Accuracy": "accuracy_mm",
    "Completeness": "completeness_mm",
    "Overall": "overall_mm",
}


def load_data(scan_id):
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"No se encontró: {CSV_PATH}")

    with CSV_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    data = {row["method"]: row for row in rows}

    if set(data) != {"qgs", "vcr_gaus"}:
        raise ValueError("El CSV debe contener QGS y VCR-GauS.")

    if any(int(row["scan_id"]) != scan_id for row in data.values()):
        raise ValueError(f"El CSV no corresponde a scan{scan_id}.")

    return data


def main():
    global CSV_PATH, OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-id", type=int, default=24)
    args = parser.parse_args()

    if args.scan_id < 1:
        parser.error("--scan-id debe ser positivo")

    scan = f"scan{args.scan_id}"

    CSV_PATH = (
        ROOT / "results/metrics/comparison.csv"
        if args.scan_id == 24
        else ROOT / "results/metrics" / f"comparison_{scan}.csv"
    )

    OUTPUT = ROOT / "results/figures" / f"metrics_comparison_{scan}.png"
    data = load_data(args.scan_id)

    methods = ["QGS", "VCR-GauS"]
    x = np.arange(len(METRICS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (method_key, method_label) in enumerate(
        [("qgs", "QGS"), ("vcr_gaus", "VCR-GauS")]
    ):
        values = [
            float(data[method_key][column])
            for column in METRICS.values()
        ]

        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=method_label
        )

        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

    ax.set_title(f"Comparación geométrica — DTU {scan.capitalize()}")
    ax.set_ylabel("Distancia (mm)")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.keys())
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Gráfica guardada en: {OUTPUT}")


if __name__ == "__main__":
    main()

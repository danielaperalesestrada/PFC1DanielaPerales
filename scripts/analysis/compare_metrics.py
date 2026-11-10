#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPORTS = {
    "qgs": ROOT / "results/dtu/scan24/comparison/qgs/metrics_common.json",
    "vcr_gaus": ROOT / "results/dtu/scan24/comparison/vcr_gaus/metrics_common.json",
}

OUTPUT = ROOT / "results/metrics/comparison.csv"

FIELDS = [
    "method",
    "scan_id",
    "accuracy_mm",
    "completeness_mm",
    "overall_mm",
    "downsample_density",
    "patch_size",
    "max_dist_mm",
    "random_seed",
    "sampling_workers",
    "mesh_vertices",
    "mesh_faces",
    "points_inside_observation_mask",
    "stl_points_above_plane",
]


def load_report(method, path):
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró: {path}")

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    if report["method"] != method:
        raise ValueError(f"El método del informe no coincide: {path}")

    if report["scan_id"] != 24:
        raise ValueError(f"Se esperaba scan24: {path}")

    metrics = report["metrics"]
    params = report["parameters"]
    counts = report["point_counts"]

    expected = (metrics["mean_d2s"] + metrics["mean_s2d"]) / 2

    if not math.isclose(metrics["overall"], expected, rel_tol=1e-9):
        raise ValueError(f"Overall no coincide con el promedio: {path}")

    return report


def main():
    reports = {
        method: load_report(method, path)
        for method, path in REPORTS.items()
    }

    # Verificar que ambos métodos utilizaron el mismo protocolo.
    qgs_params = reports["qgs"]["parameters"]
    vcr_params = reports["vcr_gaus"]["parameters"]

    if qgs_params != vcr_params:
        raise ValueError("Los parámetros de evaluación no coinciden.")

    rows = []

    for method, report in reports.items():
        metrics = report["metrics"]
        params = report["parameters"]
        counts = report["point_counts"]

        rows.append({
            "method": method,
            "scan_id": report["scan_id"],
            "accuracy_mm": f'{metrics["mean_d2s"]:.8f}',
            "completeness_mm": f'{metrics["mean_s2d"]:.8f}',
            "overall_mm": f'{metrics["overall"]:.8f}',
            "downsample_density": params["downsample_density"],
            "patch_size": params["patch_size"],
            "max_dist_mm": params["max_dist"],
            "random_seed": params["random_seed"],
            "sampling_workers": params["sampling_workers"],
            "mesh_vertices": counts["mesh_vertices"],
            "mesh_faces": counts["mesh_faces"],
            "points_inside_observation_mask":
                counts["points_inside_observation_mask"],
            "stl_points_above_plane": counts["stl_points_above_plane"],
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    qgs = reports["qgs"]["metrics"]["overall"]
    vcr = reports["vcr_gaus"]["metrics"]["overall"]
    improvement = (vcr - qgs) / vcr * 100

    print(f"CSV generado: {OUTPUT}")
    print(f"Overall QGS: {qgs:.8f} mm")
    print(f"Overall VCR-GauS: {vcr:.8f} mm")
    print(f"Mejora relativa de QGS: {improvement:.2f} %")


if __name__ == "__main__":
    main()

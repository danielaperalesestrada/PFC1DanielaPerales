#!/usr/bin/env python3
"""Resume results/efficiency/training_runs/*/summary.txt en un CSV."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/efficiency/training_runs"
OUT = ROOT / "results/efficiency/efficiency_summary.csv"
FIELDS = ["run", "method", "scan", "phase", "duration_s",
          "gpu_process_mem_max_mib", "gpu_mem_max_mib_global",
          "gpu_util_max_pct", "gpu_temp_max_c", "gpu_power_max_w"]

def parse(path):
    d = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            break
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d

rows = []
for summary in sorted(RUNS.glob("*/summary.txt")):
    d = parse(summary)
    if d.get("training_exit_code") != "0":
        print(f"Omitida (código {d.get('training_exit_code')}): {summary.parent.name}")
        continue
    scan, _, phase = d["scan_id"].partition("_")
    rows.append({
        "run": summary.parent.name, "method": d["method"], "scan": scan,
        "phase": phase or "total", "duration_s": d["duration_seconds"],
        "gpu_process_mem_max_mib": d.get("gpu_process_memory_max_mib", ""),
        "gpu_mem_max_mib_global": d.get("gpu_memory_max_mib", ""),
        "gpu_util_max_pct": d.get("gpu_utilization_max_percent", ""),
        "gpu_temp_max_c": d.get("gpu_temperature_max_c", ""),
        "gpu_power_max_w": d.get("gpu_power_max_w", ""),
    })

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} ejecuciones -> {OUT}")

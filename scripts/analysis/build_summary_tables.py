#!/usr/bin/env python3
"""Imprime tablas Markdown (geometría, fotometría, eficiencia) desde los resultados guardados.
Solo lee; no escribe archivos."""
import csv, json, statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results"
key = lambda p: int(p.name[4:])

print("### Fidelidad geométrica (evaluador común)\n")
print("| Escena | QGS Acc. | QGS Comp. | QGS Overall | VCR Acc. | VCR Comp. | VCR Overall | Mejora de QGS |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
q_all, v_all, wins = [], [], 0
for d in sorted((RES / "dtu").glob("scan*"), key=key):
    try:
        q = json.load(open(d / "comparison/qgs/metrics_common.json"))["metrics"]
        v = json.load(open(d / "comparison/vcr_gaus/metrics_common.json"))["metrics"]
    except FileNotFoundError:
        print(f"| {d.name} | faltan métricas | | | | | | |"); continue
    imp = (v["overall"] - q["overall"]) / v["overall"] * 100
    q_all.append(q["overall"]); v_all.append(v["overall"]); wins += q["overall"] < v["overall"]
    print(f"| {d.name} | {q['mean_d2s']:.5f} | {q['mean_s2d']:.5f} | **{q['overall']:.5f}** | "
          f"{v['mean_d2s']:.5f} | {v['mean_s2d']:.5f} | **{v['overall']:.5f}** | {imp:.2f} % |")
print(f"\nMedia del Overall: QGS {st.mean(q_all):.5f} mm, VCR-GauS {st.mean(v_all):.5f} mm. QGS mejor en {wins} de {len(q_all)} escenas.\n")

print("### Fotometría (vistas de entrenamiento, sobre el objeto)\n")
print("| Escena | QGS PSNR | QGS SSIM | QGS LPIPS | VCR PSNR | VCR SSIM | VCR LPIPS |")
print("|---|---:|---:|---:|---:|---:|---:|")
for f in sorted((RES / "photometric").glob("scan*/photometric_metrics.json"), key=lambda p: int(p.parent.name[4:])):
    d = json.load(open(f))
    g = lambda m, k: d[m]["summary"][k]["mean"]
    print(f"| {f.parent.name} | {g('qgs','psnr_obj'):.2f} | {g('qgs','ssim_obj'):.4f} | {g('qgs','lpips_bbox'):.4f} | "
          f"{g('vcr_gaus','psnr_obj'):.2f} | {g('vcr_gaus','ssim_obj'):.4f} | {g('vcr_gaus','lpips_bbox'):.4f} |")

print("\n### Eficiencia (una ejecución por método y escena)\n")
rows = {}
for r in csv.DictReader(open(RES / "efficiency/efficiency_summary.csv")):
    rows[(r["scan"], r["method"], r["phase"])] = r
f = lambda s, m, p, c: float(rows[(s, m, p)][c]) if (s, m, p) in rows else None
print("| Escena | QGS opt. (s) | VCR opt. (s) | Razón | QGS mem. (MiB) | VCR mem. (MiB) | QGS malla (s) | VCR malla (s) | VCR normales (s) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
ratios = []
for s in sorted({k[0] for k in rows}, key=lambda x: int(x[4:])):
    qt, vt = f(s, "qgs", "train", "duration_s"), f(s, "vcr-gaus", "train", "duration_s")
    if qt is None or vt is None:
        continue
    ratios.append(vt / qt)
    n = lambda x, fmt: "—" if x is None else format(x, fmt)
    print(f"| {s} | {qt:,.1f} | {vt:,.1f} | {vt/qt:.2f} | {n(f(s,'qgs','train','gpu_process_mem_max_mib'), '.0f')} | "
          f"{n(f(s,'vcr-gaus','train','gpu_process_mem_max_mib'), '.0f')} | {n(f(s,'qgs','mesh','duration_s'), '.1f')} | "
          f"{n(f(s,'vcr-gaus','mesh','duration_s'), '.1f')} | {n(f(s,'vcr-gaus','normals','duration_s'), '.0f')} |")
print(f"\nRazón VCR/QGS de optimización: media {st.mean(ratios):.2f}, rango {min(ratios):.2f}–{max(ratios):.2f} ({len(ratios)} escenas).")

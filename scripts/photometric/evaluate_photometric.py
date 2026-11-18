#!/usr/bin/env python3
"""PSNR, SSIM y LPIPS comunes para QGS y VCR-GauS sobre las vistas de entrenamiento.
Ejecutar desde la raíz del proyecto en el entorno vcr (necesita lpips y scikit-image)."""
import argparse, json
from pathlib import Path
import numpy as np, torch, lpips
from PIL import Image
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument("--images", required=True, help="carpeta datasets/dtu/DTU/scanXX/images")
parser.add_argument("--qgs", required=True)
parser.add_argument("--vcr", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--lpips-net", default="vgg")
args = parser.parse_args()

f32 = lambda im: np.asarray(im, dtype=np.float32) / 255.0

def load_gt(path, size):
    im = Image.open(path)
    if im.mode != "RGBA":
        raise ValueError(f"{path} no es RGBA")
    rgb = f32(im.convert("RGB").resize(size, Image.LANCZOS))
    alpha = f32(im.getchannel("A").resize(size, Image.LANCZOS))
    return rgb, alpha > 0.5

def load_render(path, size):
    im = Image.open(path).convert("RGB")
    return f32(im if im.size == size else im.resize(size, Image.LANCZOS))

def psnr_obj(a, b, m):
    return float(-10 * np.log10(max(float(((a - b)[m] ** 2).mean()), 1e-12)))

def ssim_obj(a, b, m):
    _, smap = structural_similarity(a, b, channel_axis=2, data_range=1.0,
                                    gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False, full=True)
    return float(smap[m].mean())

dev = "cuda" if torch.cuda.is_available() else "cpu"
lp = lpips.LPIPS(net=args.lpips_net).to(dev)

def lpips_bbox(a, b, m):
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x[y0:y1, x0:x1])).permute(2, 0, 1)[None].to(dev) * 2 - 1
    with torch.no_grad():
        return float(lp(t(a), t(b)).item())

names = sorted(p.stem for p in Path(args.qgs).glob("*.png"))
if names != sorted(p.stem for p in Path(args.vcr).glob("*.png")):
    raise SystemExit("QGS y VCR no tienen las mismas vistas.")
missing = [n for n in names if not (Path(args.images) / f"{n}.png").is_file()]
if missing:
    raise SystemExit(f"Sin imagen de referencia: {missing[:5]}")

size = Image.open(Path(args.qgs) / f"{names[0]}.png").size
print(f"{len(names)} vistas | resolución común {size[0]}x{size[1]} | LPIPS-{args.lpips_net} en {dev}")
gts = {n: load_gt(Path(args.images) / f"{n}.png", size) for n in names}

results = {}
for method, folder in (("qgs", args.qgs), ("vcr_gaus", args.vcr)):
    rows = {}
    for i, n in enumerate(names):
        gt, m = gts[n]
        ren = load_render(Path(folder) / f"{n}.png", size)
        other = gts[names[(i + 1) % len(names)]][0]
        own, wrong = psnr_obj(ren, gt, m), psnr_obj(ren, other, m)
        if own <= wrong:
            raise SystemExit(f"{method}: la vista {n} se parece más a otra imagen ({wrong:.1f} dB) que a la suya ({own:.1f} dB). Posible emparejamiento erróneo.")
        a, b = gt * m[..., None], ren * m[..., None]
        rows[n] = {"psnr_obj": own, "ssim_obj": ssim_obj(a, b, m), "lpips_bbox": lpips_bbox(a, b, m)}
    stats = {k: {"mean": float(np.mean([r[k] for r in rows.values()])),
                 "std": float(np.std([r[k] for r in rows.values()]))} for k in ("psnr_obj", "ssim_obj", "lpips_bbox")}
    results[method] = {"summary": stats, "per_view": rows}
    s = stats
    print(f"{method:9s} PSNR {s['psnr_obj']['mean']:.2f}±{s['psnr_obj']['std']:.2f} | SSIM {s['ssim_obj']['mean']:.4f} | LPIPS {s['lpips_bbox']['mean']:.4f}")
    worst = sorted(rows.items(), key=lambda kv: kv[1]["psnr_obj"])[:3]
    print("   peores vistas (PSNR):", [(k, round(v["psnr_obj"], 1)) for k, v in worst])

results["protocol"] = {"views": "entrenamiento (todas)", "resolution": list(size),
    "psnr_ssim": "solo píxeles del objeto (alfa de la imagen original)",
    "lpips": f"caja delimitadora del objeto, LPIPS-{args.lpips_net}",
    "background": "negro en ambos renders; referencia enmascarada con el alfa"}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("Guardado en", args.out)

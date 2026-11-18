#!/usr/bin/env python3
"""Renderiza las vistas de entrenamiento de VCR-GauS con fondo NEGRO.
Ejecutar desde VCR-GauS/ en el entorno vcr."""
import argparse, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from PIL import Image
from configs.config import Config
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tools.general_utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--iteration", type=int, default=30000)
args = parser.parse_args()

cfg = Config(os.path.join(args.model_path, "config.yaml"))
cfg.model.data_device = "cuda"
cfg.model.load_normal = False
set_random_seed(cfg.seed)
model = GaussianModel(cfg.model)
scene = Scene(cfg.model, model, load_iteration=args.iteration, shuffle=False)
bg = torch.zeros(3, device="cuda")
cams = scene.getTrainCameras()

os.makedirs(args.out, exist_ok=True)
for cam in cams:
    name = getattr(cam, "image_name", None)
    if name is None:
        raise SystemExit("La cámara no tiene image_name; no se puede emparejar con la imagen original.")
    with torch.no_grad():
        pkg = render(cam, model, cfg, bg, dirs=scene.dirs)
    img = (pkg["render"][:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.out, f"{name}.png"))
print(f"{len(cams)} vistas guardadas en {args.out} | tamaño {img.shape[1]}x{img.shape[0]} | fondo negro")

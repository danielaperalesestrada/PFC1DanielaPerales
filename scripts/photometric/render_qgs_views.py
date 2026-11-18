#!/usr/bin/env python3
"""Renderiza las vistas de entrenamiento de QGS, una imagen por vista con el
nombre de la imagen original. Ejecutar desde QGS/ en el entorno QGS.
No genera ni modifica mallas."""
import argparse, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.system_utils import load_config
from utils.mesh_utils import GaussianExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--conf_path", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--iteration", type=int, default=-1)
args, extras = parser.parse_known_args()

config = OmegaConf.load(args.conf_path)
config = load_config(f"{config.load_model_path}/config.yaml", cli_args=extras)
gaussians = GaussianModel(config.gs_model)
scene = Scene(config, gaussians, load_iteration=args.iteration, shuffle=False)
bg = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
extractor = GaussianExtractor(gaussians, render, config.pipeline, config.optimizer, bg_color=bg)
cams = scene.getTrainCameras()
extractor.reconstruction(cams)

os.makedirs(args.out, exist_ok=True)
for cam, rgb in zip(cams, extractor.rgbmaps):
    img = (rgb[:3].detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.out, f"{cam.image_name}.png"))
print(f"{len(cams)} vistas guardadas en {args.out} | tamaño {img.shape[1]}x{img.shape[0]} | fondo {bg}")

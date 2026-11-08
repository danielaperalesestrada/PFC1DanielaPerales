import sys, os, numpy as np, torch
sys.path.append(os.getcwd())
from configs.config import Config
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tools.general_utils import set_random_seed
from tools.mesh_utils import GaussianExtractor, to_cam_open3d
import open3d as o3d
model_path = sys.argv[1]
iteration  = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
depth_trunc = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
voxel_size  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.004
cam_stride  = int(sys.argv[5]) if len(sys.argv) > 5 else 2

cfg = Config(os.path.join(model_path, "config.yaml"))
cfg.model.data_device = 'cpu'
cfg.model.load_normal = False
cfg.model.load_mask = False
set_random_seed(cfg.seed)
model = GaussianModel(cfg.model)
scene = Scene(cfg.model, model, load_iteration=iteration, shuffle=False)
model.trans = torch.from_numpy(scene.trans).cuda()
model.scale = torch.from_numpy(scene.scale).cuda() * 1.1
model.extent = scene.cameras_extent
cams = scene.getTrainCameras().copy()[::cam_stride]
model.training_setup(cfg.optim)
model.max_radii2D = torch.zeros((model.get_xyz.shape[0]), device="cuda")
model.scale = torch.from_numpy(scene.scale).cuda()
model.prune_outliers()
bg = torch.zeros(3, dtype=torch.float32, device="cuda")
ext = GaussianExtractor(model, render, cfg, bg_color=bg, dirs=scene.dirs, prob_thres=0.15, alpha_thres=0.5)
ext.reconstruction(cams)
d = ext.depthmaps
print("depthmaps shape:", tuple(d.shape), d.dtype)
print("frac depth>0: %.3f | frac 0<depth<%.2f: %.3f" % ((d>0).float().mean(), depth_trunc, ((d>0)&(d<depth_trunc)).float().mean()))
print("percentiles 1/50/99 de depth>0:", np.percentile(d[d>0].numpy(), [1,50,99]).round(3))
m = ext.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=voxel_size*5, depth_trunc=depth_trunc)
n_v = len(m.vertices); n_t = len(m.triangles)
print("mesh: vertices=%d triangles=%d" % (n_v, n_t))
if n_v > 0:
    out_path = os.path.join(model_path, f"mesh_iter{iteration}.ply")
    try:
        m.export(out_path)
    except Exception:
        o3d.io.write_triangle_mesh(out_path, m)
    print("saved:", out_path)
else:
    print("ERROR: malla vacia. Prueba con depth_trunc mas alto (ej. 10, 30, 100).")

import numpy as np
from PIL import Image as _Img
_d = ext.depthmaps.cpu().numpy()
for i in [0, 5, 10, 15, 20]:
    if i < len(_d):
        _dd = _d[i, 0]
        _norm = (255 * _dd / max(_dd.max(), 1e-6)).astype(np.uint8)
        _Img.fromarray(_norm).save(f"/tmp/depth_{i:02d}.png")
print("guardados /tmp/depth_*.png")

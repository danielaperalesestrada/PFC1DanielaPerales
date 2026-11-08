import sys, os, torch
sys.path.append(os.getcwd())
from configs.config import Config
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tools.general_utils import set_random_seed
mp = sys.argv[1]; it = int(sys.argv[2])
cfg = Config(os.path.join(mp, "config.yaml"))
cfg.model.data_device = 'cuda'; cfg.model.load_normal = False
set_random_seed(cfg.seed)
model = GaussianModel(cfg.model)
scene = Scene(cfg.model, model, load_iteration=it, shuffle=False)
bg = torch.ones(3, device="cuda")
bad = 0
for i, cam in enumerate(scene.getTrainCameras()):
    with torch.no_grad():
        pkg = render(cam, model, cfg, bg, dirs=scene.dirs)
    r = pkg['render'].cpu(); g = cam.original_image.cpu()
    gm = cam.gt_alpha_mask[0].cpu() > 0.5
    mse = ((r.clamp(0, 1) - g) ** 2)[:, gm].mean()
    psnr = float(-10 * torch.log10(mse))
    out = float(((r < 0) | (r > 1)).float().mean())
    bad += psnr < 8
    print("cam %2d PSNR %5.1f render[min,max]=[%8.2f,%8.2f] fuera_de_[0,1]=%.4f" % (i, psnr, r.min(), r.max(), out))
print("vistas con PSNR<8:", bad)

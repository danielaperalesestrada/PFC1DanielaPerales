import sys, os, numpy as np, torch
sys.path.append(os.getcwd())
from PIL import Image
from configs.config import Config
from gaussian_renderer import render
from scene import Scene, GaussianModel
from tools.general_utils import set_random_seed
mp = sys.argv[1]; it = int(sys.argv[2])
if len(sys.argv) > 3 and sys.argv[3] == 'clamp':
    from scene.gaussian_model import GaussianModel as _G
    _G.get_scaling = property(lambda self: torch.clamp(self.scaling_activation(self._scaling), min=1e-2))
    print('usando suelo de escala 1e-2')
cfg = Config(os.path.join(mp, "config.yaml"))
cfg.model.data_device = 'cuda'; cfg.model.load_normal = False
set_random_seed(cfg.seed)
model = GaussianModel(cfg.model)
scene = Scene(cfg.model, model, load_iteration=it, shuffle=False)
bg = (torch.ones if cfg.model.white_background else torch.zeros)(3, device="cuda")
cams = scene.getTrainCameras()
tiles = []
for i in [0, 10, 20]:
    cam = cams[i]
    with torch.no_grad():
        pkg = render(cam, model, cfg, bg, dirs=scene.dirs)
    a = pkg['alpha'][0].cpu(); gm = (cam.gt_alpha_mask[0].cpu() > 0.5)
    r = pkg['render'].clamp(0,1).cpu(); g = cam.original_image.cpu()
    gw = g * gm + (~gm) * 1.0
    psnr = (-10*torch.log10(((r-gw)**2)[:, gm].mean())).item()
    print(f"cam {i}: alpha medio en fondo={a[~gm].mean():.3f} en edificio={a[gm].mean():.3f} | PSNR en edificio={psnr:.1f}")
    row = [gw.permute(1,2,0).numpy(), r.permute(1,2,0).numpy(), np.repeat(a[...,None].numpy(),3,2)]
    tiles.append(np.concatenate([np.asarray(Image.fromarray((t*255).astype(np.uint8)).resize((518,387))) for t in row],1))
Image.fromarray(np.concatenate(tiles,0)).save("check_render.png"); print("guardado check_render.png (foto | render | alfa)")

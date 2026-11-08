import os
import sys

sys.path.append(os.getcwd())
from python_scripts.run_base import train_cmd, extract_mesh_cmd, eval_cd_cmd, check_finish

TRIAL_NAME = 'vcr_gaus'
PROJECT = 'vcr_gaus'
PROJECT_wandb = 'vcr_gaus_dtu'
DATASET = 'dtu'
base_dir = "/media/daniela/Archivos/PFC1DanielaPerales"
output_dir = f"{base_dir}/outputs/{PROJECT}/{DATASET}"
data_dir = f"{base_dir}/datasets/dtu/DTU"

do_train = True
do_extract_mesh = True
do_cd = False
dry_run = False

scenes = ['scan24']
factors = [-1]
debug_from = -1
data_device = 'cuda'
voxel_size = 0.004
step = 1
PLY = "ours.ply"
NUM_THREADS = 4
prob_thr = 0.15
num_cluster = 1
max_depth = 3
fuse_method = 'tsdf_cpu'

def train_scene(gpu, scene, factor):
    os.system('ulimit -n 9000')
    log_dir = f"{output_dir}/{scene}/{TRIAL_NAME}"
    fail = 0

    if do_train:
        cmd = train_cmd.format(gpu=gpu, dataset=DATASET, cfg='base',
                            scene=scene, log_dir=log_dir,
                            data_dir=data_dir, debug_from=debug_from,
                            data_device=data_device, resolution=factor, project=PROJECT_wandb)
        print(cmd)
        fail = os.system(cmd)

    if fail == 0 and do_extract_mesh:
        if not check_finish(scene, f"{log_dir}/point_cloud", 'train'): return False
        cmd = extract_mesh_cmd.format(gpu=gpu, ply=PLY, step=step, fuse_method=fuse_method,
                                      voxel_size=voxel_size, num_cluster=num_cluster,
                                      max_depth=max_depth, log_dir=log_dir, prob_thr=prob_thr)
        fail = os.system(cmd)
        print(cmd)

    if fail == 0 and do_cd:
        scan_id = scene[4:]
        cmd = eval_cd_cmd.format(num_threads=NUM_THREADS, gpu=gpu,
                                 tri_mesh_path=f'{log_dir}/{PLY}', scan_id=scan_id,
                                 output_dir=log_dir, data_dir=data_dir)
        print(cmd)
        fail = os.system(cmd)

    return fail == 0

train_scene(0, 'scan24', -1)
print(TRIAL_NAME, " done")

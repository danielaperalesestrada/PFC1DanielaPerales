import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from scipy.ndimage import binary_dilation
from skimage.morphology import disk


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = None
METHODS = {}
OUTPUT_DIR = None


def load_camera(world_mat, scale_mat):
    P = (world_mat @ scale_mat)[:3, :4]

    out = cv2.decomposeProjectionMatrix(P)

    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return (
        torch.from_numpy(intrinsics).float(),
        torch.from_numpy(pose).float(),
    )


def cull_mesh(mesh_path, output_path):
    print(f"\nCargando malla: {mesh_path}")

    mesh = trimesh.load(
        str(mesh_path),
        force="mesh",
        process=False
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    print(f"Vértices originales: {len(vertices)}")
    print(f"Caras originales: {len(faces)}")

    image_paths = sorted(p for p in (DATA_DIR / "images").glob("*.png") if not p.name.startswith("._"))
    mask_paths = sorted(p for p in (DATA_DIR / "mask").glob("*.png") if not p.name.startswith("._"))

    if len(image_paths) != len(mask_paths):
        raise ValueError("La cantidad de imágenes y máscaras no coincide.")

    n_images = len(image_paths)

    print(f"Cámaras y máscaras: {n_images}")

    W, H = 1600, 1200

    camera_file = DATA_DIR / "cameras.npz"

    with np.load(camera_file) as camera_dict:
        cameras = []

        for i in range(n_images):
            scale_key = f"scale_mat_{i}"
            world_key = f"world_mat_{i}"

            if scale_key not in camera_dict or world_key not in camera_dict:
                raise KeyError(f"Faltan matrices de cámara para la vista {i}")

            scale_mat = camera_dict[scale_key].astype(np.float32)
            world_mat = camera_dict[world_key].astype(np.float32)

            cameras.append(load_camera(world_mat, scale_mat))

        scale_mat = camera_dict["scale_mat_0"].astype(np.float32)

    if not image_paths:
        raise ValueError("No se encontraron imágenes.")

    keep = np.ones(len(vertices), dtype=bool)

    vertices_h = torch.from_numpy(vertices)
    vertices_h = torch.cat(
        (vertices_h, torch.ones_like(vertices_h[:, :1])),
        dim=1
    ).T

    print("\nAplicando filtrado común...")

    for i, (intrinsics, pose) in enumerate(cameras):
        mask = cv2.imread(str(mask_paths[i]))

        if mask is None:
            raise ValueError(f"No se pudo leer la máscara: {mask_paths[i]}")

        if mask.shape[:2] != (H, W):
            raise ValueError(
                f"Dimensiones inesperadas en {mask_paths[i]}: {mask.shape}"
            )

        mask = mask[:, :, 0].astype(np.float32) / 256.0
        mask = binary_dilation(mask, disk(24))

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask)
        ).float()[None, None]

        w2c = torch.linalg.inv(pose)

        cam_points = intrinsics @ w2c @ vertices_h

        pix_coords = cam_points[:2] / (
            cam_points[2:3] + 1e-6
        )

        pix_coords = pix_coords.T

        pix_coords[:, 0] /= W - 1
        pix_coords[:, 1] /= H - 1

        pix_coords = (pix_coords - 0.5) * 2

        valid = (
            (pix_coords > -1.0) &
            (pix_coords < 1.0)
        ).all(dim=-1).float()

        sampled_mask = F.grid_sample(
            mask_tensor,
            pix_coords[None, None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=True
        )[0, 0, 0]

        sampled_mask = sampled_mask + (1.0 - valid)

        keep &= (sampled_mask.numpy() > 0.0)

        print(
            f"Vista {i + 1}/{n_images}: "
            f"{keep.sum()} vértices conservados"
        )

    face_keep = keep[faces].all(axis=1)

    # Crear nuevos índices sin alterar la geometría de las caras.
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(keep.sum())

    filtered_faces = old_to_new[faces[face_keep]]

    # Aplicar la misma transformación usada por los evaluadores originales.
    filtered_vertices = vertices[keep]
    filtered_vertices = (
        filtered_vertices * scale_mat[0, 0]
        + scale_mat[:3, 3][None]
    )

    # Conservar colores de vértices, si existen.
    extra = {}

    try:
        colors = np.asarray(mesh.visual.vertex_colors)

        if len(colors) == len(vertices):
            extra["vertex_colors"] = colors[keep]
    except (AttributeError, ValueError):
        pass

    common_mesh = trimesh.Trimesh(
        vertices=filtered_vertices,
        faces=filtered_faces,
        process=False,
        **extra
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    common_mesh.export(str(output_path))

    report = {
        "input_mesh": str(mesh_path),
        "output_mesh": str(output_path),
        "camera_count": n_images,
        "mask_dilation_radius": 24,
        "vertices_original": int(len(vertices)),
        "vertices_conservados": int(len(filtered_vertices)),
        "faces_originales": int(len(faces)),
        "faces_conservadas": int(len(filtered_faces)),
        "largest_component_selection": False,
        "automatic_mesh_processing": False,
        "hole_repair": False,
    }

    report_path = output_path.with_name("culling_report.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\nFiltrado terminado.")
    print(f"Vértices finales: {len(filtered_vertices)}")
    print(f"Caras finales: {len(filtered_faces)}")
    print(f"Malla guardada en: {output_path}")
    print(f"Informe guardado en: {report_path}")


def main():
    global DATA_DIR, METHODS, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-id", type=int, default=24)

    parser.add_argument(
        "--method",
        choices=["qgs", "vcr_gaus", "both"],
        default="both"
    )

    args = parser.parse_args()

    if args.scan_id < 1:
        parser.error("--scan-id debe ser positivo")

    scan = f"scan{args.scan_id}"
    DATA_DIR = ROOT / "datasets/dtu/DTU" / scan

    METHODS = {
        "qgs": {
            "input": ROOT / "outputs" / f"qgs_{scan}" / "train/ours_30000/fuse_post.ply",
        },
        "vcr_gaus": {
            "input": ROOT / "outputs/vcr_gaus/dtu" / scan / "final_shpy/ours.ply",
        },
    }

    OUTPUT_DIR = ROOT / "results/dtu" / scan / "comparison"

    methods = (
        ["qgs", "vcr_gaus"]
        if args.method == "both"
        else [args.method]
    )

    for method in methods:
        input_path = METHODS[method]["input"]

        output_path = (
            OUTPUT_DIR / method / "culled_mesh_common.ply"
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"No se encontró la malla de entrada: {input_path}"
            )

        print(f"\n========== {method.upper()} ==========")

        cull_mesh(input_path, output_path)


if __name__ == "__main__":
    main()

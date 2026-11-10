#!/usr/bin/env python3
"""Evaluacion DTU comun para las mallas ya filtradas de QGS y VCR-GauS."""

import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/dtu/scan24/comparison"
GT = ROOT / "datasets/dtu_gt/DTU_eval"
MESHES = {
    "qgs": BASE / "qgs/culled_mesh_common.ply",
    "vcr_gaus": BASE / "vcr_gaus/culled_mesh_common.ply",
}
SCAN = 24


def sample_single_tri(task):
    n1, n2, v1, v2, tri = task
    n1, n2 = int(n1), int(n2)
    c = np.mgrid[:n1+1, :n2+1].astype(np.float64)
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    return v1 * k[:, :1] + v2 * k[:, 1:] + tri


def sample_mesh(mesh, density, workers):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("La malla no contiene vertices o triangulos.")

    tri = vertices[triangles]
    v1, v2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    valid = area2[:, 0] > 0
    l1, l2, area2, v1, v2, tri = [
        a[valid] for a in (l1, l2, area2, v1, v2, tri)
    ]
    if len(tri) == 0:
        raise ValueError("No hay triangulos de area positiva.")

    threshold = density * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / threshold).astype(np.int64)
    n2 = np.floor(l2 / threshold).astype(np.int64)
    tasks = (
        (n1[i, 0], n2[i, 0], v1[i:i+1], v2[i:i+1], tri[i:i+1, 0])
        for i in range(len(n1))
    )
    print(f"Triangulos a muestrear: {len(n1):,}")
    with mp.Pool(processes=workers) as pool:
        chunks = list(tqdm(
            pool.imap(sample_single_tri, tasks, chunksize=1024),
            total=len(n1), desc="Muestreo de triangulos"
        ))
    surface = np.concatenate(chunks, axis=0)
    return np.concatenate((vertices, surface), axis=0), len(surface)


def downsample(points, radius, seed):
    points = points.copy()
    np.random.default_rng(seed).shuffle(points, axis=0)
    print(f"Submuestreo por radio: {radius}; puntos de entrada: {len(points):,}")
    nn = NearestNeighbors(
        n_neighbors=1, radius=radius, algorithm="kd_tree", n_jobs=-1
    )
    nn.fit(points)
    neighbors = nn.radius_neighbors(
        points, radius=radius, return_distance=False
    )
    keep = np.ones(len(points), dtype=bool)
    for i, indices in enumerate(tqdm(neighbors, desc="Filtrado por radio")):
        if keep[i]:
            keep[indices] = False
            keep[i] = True
    return points[keep]


def evaluate(method, args):
    mesh_path = MESHES[method]
    out_dir = BASE / method
    report_path = out_dir / "metrics_common.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Ya existe {report_path}; no se sobrescribira. "
            "Usa --overwrite explicitamente si lo deseas."
        )

    obs_path = GT / "ObsMask" / f"ObsMask{SCAN}_10.mat"
    plane_path = GT / "ObsMask" / f"Plane{SCAN}.mat"
    stl_path = GT / "Points/stl" / f"stl{SCAN:03}_total.ply"
    for path in (mesh_path, obs_path, plane_path, stl_path):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontro: {path}")

    print(f"\nEvaluando {method}: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"No se pudo cargar la malla: {mesh_path}")
    vertex_count, face_count = len(mesh.vertices), len(mesh.triangles)

    points, surface_count = sample_mesh(mesh, args.density, args.workers)
    points_before = len(points)
    data_down = downsample(points, args.density, args.seed)

    obs_data = loadmat(obs_path)
    obs_mask, bb, resolution = [
        obs_data[key] for key in ("ObsMask", "BB", "Res")
    ]
    bb = bb.astype(np.float32)
    in_bb = (
        (data_down >= bb[:1] - args.patch)
        & (data_down < bb[1:] + args.patch * 2)
    ).sum(axis=-1) == 3
    data_in = data_down[in_bb]

    grid = np.around((data_in - bb[:1]) / resolution).astype(np.int32)
    grid_in = (
        (grid >= 0) & (grid < np.expand_dims(obs_mask.shape, 0))
    ).sum(axis=-1) == 3
    grid_valid = grid[grid_in]
    observed = obs_mask[
        grid_valid[:, 0], grid_valid[:, 1], grid_valid[:, 2]
    ].astype(bool)
    data_in_obs = data_in[grid_in][observed]
    if len(data_in) == 0 or len(data_in_obs) == 0:
        raise ValueError("No hay puntos suficientes dentro de BB/ObsMask.")

    stl_cloud = o3d.io.read_point_cloud(str(stl_path))
    stl = np.asarray(stl_cloud.points)
    if len(stl) == 0:
        raise ValueError("La nube STL de referencia esta vacia.")

    nn = NearestNeighbors(
        n_neighbors=1, radius=args.density, algorithm="kd_tree", n_jobs=-1
    )
    print("Calculando D2S...")
    nn.fit(stl)
    d2s, _ = nn.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    d2s_valid = d2s[d2s < args.max_dist]
    if len(d2s_valid) == 0:
        raise ValueError("No hay distancias D2S menores que max_dist.")
    mean_d2s = float(d2s_valid.mean())

    print("Calculando S2D...")
    plane = loadmat(plane_path)["P"]
    stl_h = np.concatenate((stl, np.ones_like(stl[:, :1])), axis=-1)
    above = (plane.reshape((1, 4)) * stl_h).sum(axis=-1) > 0
    stl_above = stl[above]
    if len(stl_above) == 0:
        raise ValueError("No hay puntos STL por encima del plano.")
    nn.fit(data_in)
    s2d, _ = nn.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    s2d_valid = s2d[s2d < args.max_dist]
    if len(s2d_valid) == 0:
        raise ValueError("No hay distancias S2D menores que max_dist.")
    mean_s2d = float(s2d_valid.mean())
    overall = (mean_d2s + mean_s2d) / 2

    report = {
        "method": method,
        "scan_id": SCAN,
        "metrics": {
            "mean_d2s": mean_d2s,
            "mean_s2d": mean_s2d,
            "overall": overall,
        },
        "parameters": {
            "downsample_density": args.density,
            "patch_size": args.patch,
            "max_dist": args.max_dist,
            "random_seed": args.seed,
            "sampling_workers": args.workers,
        },
        "point_counts": {
            "mesh_vertices": vertex_count,
            "mesh_faces": face_count,
            "surface_sample_points": int(surface_count),
            "points_before_downsampling": int(points_before),
            "points_after_downsampling": int(len(data_down)),
            "points_inside_bounding_box": int(len(data_in)),
            "points_inside_observation_mask": int(len(data_in_obs)),
            "stl_points": int(len(stl)),
            "stl_points_above_plane": int(len(stl_above)),
            "d2s_distances_below_max_dist": int(len(d2s_valid)),
            "s2d_distances_below_max_dist": int(len(s2d_valid)),
        },
        "protocol": {
            "mesh_input": str(mesh_path.relative_to(ROOT)),
            "ground_truth_directory": str(GT.relative_to(ROOT)),
            "culling": "common camera-mask culling; all components retained; no hole repair",
            "sampling": "same triangle sampling formula as original QGS/VCR-GauS evaluators",
            "downsampling": "radius_neighbors with fixed-seed shuffled point order",
            "d2s": "observable reconstruction points to complete STL",
            "s2d": "STL points above Plane24 to reconstruction points inside BB",
            "distance_filter": "only distances strictly below max_dist are averaged",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nD2S:     {mean_d2s:.8f}")
    print(f"S2D:     {mean_s2d:.8f}")
    print(f"Overall: {overall:.8f}")
    print(f"Informe: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Metricas DTU comunes para QGS y VCR-GauS."
    )
    parser.add_argument("--method", choices=("qgs", "vcr_gaus", "both"), default="both")
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--patch", type=float, default=60.0)
    parser.add_argument("--max-dist", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers debe ser al menos 1.")
    methods = list(MESHES) if args.method == "both" else [args.method]
    for method in methods:
        evaluate(method, args)


if __name__ == "__main__":
    mp.freeze_support()
    main()

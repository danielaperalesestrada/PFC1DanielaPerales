#!/usr/bin/env bash
# Pipeline de una escena DTU: preprocesado, entrenamiento, mallas, evaluación y eficiencia.
#   bash scripts/run_scene.sh 65
#   DRY_RUN=1 bash scripts/run_scene.sh 65     # solo muestra qué haría
set -euo pipefail

ID="${1:?Uso: bash scripts/run_scene.sh <id de dos dígitos, p. ej. 65>}"
[[ "$ID" =~ ^[0-9]{2}$ ]] || { echo "ERROR: el id debe tener dos dígitos." >&2; exit 1; }
(( 10#$ID >= 10 )) || { echo "ERROR: ids menores de 10 no están soportados (los evaluadores usan scanN sin cero)." >&2; exit 1; }

SCAN="scan${ID}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENVS_DIR="${ENVS_DIR:-/media/daniela/Archivos/conda/envs}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
DRY="${DRY_RUN:-0}"
CLEAN_CKPT="${CLEAN_CKPT:-1}"
export PYTHONUNBUFFERED=1

DATA="datasets/dtu/DTU/$SCAN"
GT="datasets/dtu_gt/dtu-sample/SampleSet/MVS_Data"
VCR_OUT="outputs/vcr_gaus/dtu/$SCAN/final_shpy"
QGS_OUT="outputs/qgs_$SCAN"
RES="results/dtu/$SCAN/comparison"
PH="results/photometric/$SCAN"
QGS_CFG="configs/qgs/dtu_$SCAN.yaml"
VCR_CFG="configs/vcr_gaus/dtu_$SCAN.yaml"
MON="bash scripts/monitoring/monitor_training.sh"

[[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || { echo "ERROR: no encuentro conda.sh (CONDA_BASE='$CONDA_BASE')" >&2; exit 1; }
for e in geowizard vcr QGS; do [[ -d "$ENVS_DIR/$e" ]] || { echo "ERROR: no existe el entorno $ENVS_DIR/$e" >&2; exit 1; }; done
source "$CONDA_BASE/etc/profile.d/conda.sh"

say() { printf '\n=== %s ===\n' "$*"; }
run() { if [[ "$DRY" == 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }
act() {
    if [[ "$DRY" == 1 ]]; then echo "[dry-run] conda activate $1"; return 0; fi
    set +u; conda activate "$1"; set -u
}
nonempty() { [[ -d "$1" && -n "$(ls -A "$1")" ]]; }

say "Comprobaciones previas ($SCAN)"
for f in "$DATA/cameras.npz" "$DATA/sparse" "$DATA/mask" \
         "$GT/Points/stl/stl0${ID}_total.ply" "$GT/ObsMask/ObsMask${ID}_10.mat" "$GT/ObsMask/Plane${ID}.mat"; do
    [[ -e "$f" ]] || { echo "ERROR: falta $f" >&2; exit 1; }
done
n_img=$(find "$DATA/images" -name '*.png' | wc -l)
[[ "$n_img" -eq 49 ]] || { echo "ERROR: $DATA/images tiene $n_img imágenes (se esperaban 49)." >&2; exit 1; }
avail=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[[ "$avail" -ge 20 ]] || { echo "ERROR: solo hay ${avail} GB libres (mínimo 20)." >&2; exit 1; }
echo "OK: 49 imágenes, GT completo, ${avail} GB libres."

say "Configs"
if [[ ! -f "$QGS_CFG" ]]; then
    echo "Creando $QGS_CFG a partir de scan40"
    [[ "$DRY" == 1 ]] || sed "s/scan40/$SCAN/g" configs/qgs/dtu_scan40.yaml > "$QGS_CFG"
fi
if [[ ! -f "$VCR_CFG" ]]; then
    echo "Creando $VCR_CFG a partir de scan63"
    [[ "$DRY" == 1 ]] || cp configs/vcr_gaus/dtu_scan63.yaml "$VCR_CFG"
fi
if [[ "$DRY" != 1 ]]; then
    grep -q "case_name: $SCAN" "$QGS_CFG" || { echo "ERROR: $QGS_CFG no tiene case_name: $SCAN" >&2; exit 1; }
    for c in "$QGS_CFG" "$VCR_CFG"; do grep -qiE "eval: *false" "$c" || { echo "ERROR: eval no es false en $c." >&2; exit 1; }; done
fi

say "1/8 Normales con GeoWizard"
if [[ $(find "$DATA/normal_npz_indoor" -name '*.npz' 2>/dev/null | wc -l) -eq 49 ]]; then
    echo "Ya generadas; se omite."
else
    act "$ENVS_DIR/geowizard"
    run $MON vcr-gaus "${SCAN}_normals" -- bash -c "cd GeoWizard && python -W ignore $ROOT/VCR-GauS/process_data/extract_normal_geo.py --code_path $ROOT/GeoWizard/geowizard --pretrained_model_path lemonaddie/geowizard --input_dir $ROOT/$DATA/images/ --output_dir $ROOT/$DATA/ --ensemble_size 3 --denoise_steps 10 --processing_res 768 --seed 0 --domain indoor"
fi

say "2/8 VCR-GauS: entrenamiento"
if [[ -d "$VCR_OUT/point_cloud/iteration_30000" ]]; then
    echo "Ya entrenado; se omite."
elif nonempty "$VCR_OUT"; then
    echo "ERROR: $VCR_OUT existe pero el entrenamiento no terminó. Revísalo o muévelo antes de seguir." >&2; exit 1
else
    act "$ENVS_DIR/vcr"
    run $MON vcr-gaus "${SCAN}_train" -- env SCAN_ID="$ID" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 bash scripts/training/train_vcr_gaus.sh
fi

say "3/8 VCR-GauS: malla"
if [[ -f "$VCR_OUT/ours.ply" ]]; then
    echo "Ya generada; se omite."
else
    act vcr
    run $MON vcr-gaus "${SCAN}_mesh" -- bash -c "cd VCR-GauS && python tools/depth2mesh.py --mesh_name ours.ply --split 1 --method tsdf_cpu --voxel_size 0.004 --num_cluster 1 --max_depth 3 --clean --prob_thres 0.15 --cfg_path ../$VCR_OUT/config.yaml"
fi

say "4/8 QGS: entrenamiento"
if [[ -f "$QGS_OUT/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    echo "Ya entrenado; se omite."
elif nonempty "$QGS_OUT"; then
    echo "ERROR: $QGS_OUT existe pero el entrenamiento no terminó. Revísalo o muévelo antes de seguir." >&2; exit 1
else
    act QGS
    run $MON qgs "${SCAN}_train" -- env SCAN_ID="$ID" bash scripts/training/train_qgs.sh
fi

say "5/8 QGS: malla"
if [[ -f "$QGS_OUT/train/ours_30000/fuse_post.ply" ]]; then
    echo "Ya generada; se omite."
else
    act QGS
    run $MON qgs "${SCAN}_mesh" -- bash -c "cd QGS && python render.py --conf_path ../$QGS_OUT/config.yaml --skip_train --skip_test"
fi

say "6/8 Fidelidad geométrica"
act vcr
if [[ -f "$RES/qgs/metrics_common.json" && -f "$RES/vcr_gaus/metrics_common.json" ]]; then
    echo "Métricas ya calculadas; se omiten."
else
    run python scripts/evaluation/evaluate_common.py --scan-id "$ID" --method both
    run python scripts/evaluation/evaluate_metrics.py --scan-id "$ID" --method both --density 0.2 --patch 60 --max-dist 20 --seed 42 --workers 4
fi
run python scripts/analysis/compare_metrics.py --scan-id "$ID"
run python scripts/analysis/visualize_results.py --scan-id "$ID"

say "7/8 Métricas fotométricas (vistas de entrenamiento)"
if [[ -f "$PH/photometric_metrics.json" ]]; then
    echo "Ya calculadas; se omiten."
else
    act QGS
    run bash -c "cd QGS && python ../scripts/photometric/render_qgs_views.py --conf_path ../$QGS_OUT/config.yaml --out ../$PH/qgs"
    act vcr
    run bash -c "cd VCR-GauS && python ../scripts/photometric/render_vcr_views.py --model_path ../$VCR_OUT --out ../$PH/vcr_gaus"
    run python scripts/photometric/evaluate_photometric.py --images "$DATA/images" --qgs "$PH/qgs" --vcr "$PH/vcr_gaus" --out "$PH/photometric_metrics.json"
fi

say "8/8 Resumen de eficiencia y limpieza"
run python scripts/analysis/summarize_efficiency.py
if [[ "$CLEAN_CKPT" == 1 ]]; then
    echo "Borrando checkpoints intermedios de VCR (se conserva chkpnt30000.pth)."
    run find "$VCR_OUT" -name 'chkpnt*.pth' ! -name 'chkpnt30000.pth' -delete
fi
echo; echo "Escena $SCAN terminada."

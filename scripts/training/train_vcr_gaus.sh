#!/usr/bin/env bash
set -euo pipefail

# Directorio raíz del proyecto
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Rutas del experimento
VCR_DIR="$ROOT/VCR-GauS"
CONFIG="$ROOT/configs/vcr_gaus/dtu_scan24.yaml"
DATA="$ROOT/datasets/dtu/DTU/scan24"
LOGDIR="${VCR_LOGDIR:-$ROOT/outputs/vcr_gaus/dtu/scan24/final_shpy}"

# Comprobaciones previas
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: No se encuentra la configuración: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$VCR_DIR" || ! -d "$DATA" ]]; then
    echo "ERROR: No se encuentra VCR-GauS o el dataset DTU." >&2
    exit 1
fi

if [[ -d "$LOGDIR" ]] && [[ -n "$(find "$LOGDIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: El directorio de salida ya contiene archivos:" >&2
    echo "$LOGDIR" >&2
    echo "Utiliza VCR_LOGDIR para indicar otro directorio." >&2
    exit 1
fi

# Parámetros del entorno
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Compatible con la configuración de PyTorch 2.0.1.
# No añadimos expandable_segments porque no hemos confirmado su compatibilidad.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "$LOGDIR"
cd "$VCR_DIR"

# Entrenamiento
python train.py \
    --config="$CONFIG" \
    --logdir="$LOGDIR" \
    --model.source_path="$DATA/" \
    --model.data_device=cuda \
    --model.resolution=-1 \
    --model.white_background=True \
    --pipline.convert_SHs_python=True \
    '--train.checkpoint_iterations=[2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000]' \
    '--train.save_iterations=[14000,29000,30000]' \
    --wandb_name=final_shpy \
    2>&1 | tee "$LOGDIR/train.log"

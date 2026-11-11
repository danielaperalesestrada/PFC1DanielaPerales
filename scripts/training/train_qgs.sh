#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCAN_ID="${SCAN_ID:-24}"

if [[ ! "$SCAN_ID" =~ ^[0-9]{2}$ ]]; then
    echo "ERROR: SCAN_ID debe tener dos dígitos, por ejemplo 24 o 37." >&2
    exit 1
fi

QGS_DIR="$ROOT/QGS"
CONFIG="$ROOT/configs/qgs/dtu_scan${SCAN_ID}.yaml"
DATA="$ROOT/datasets/dtu/DTU/scan${SCAN_ID}"
OUTPUT_DIR="$ROOT/outputs/qgs_scan${SCAN_ID}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: No se encuentra la configuración: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$QGS_DIR" || ! -d "$DATA" ]]; then
    echo "ERROR: No se encuentra QGS o el dataset DTU scan${SCAN_ID}." >&2
    exit 1
fi

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: El directorio de salida ya contiene archivos:" >&2
    echo "$OUTPUT_DIR" >&2
    exit 1
fi


# Reparar enlaces CUDA afectados por el sistema de archivos.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: Activa el entorno Conda QGS antes de ejecutar este script." >&2
    exit 1
fi

CUDA_LINK_DIR="/tmp/qgs-cuda-links"
mkdir -p "$CUDA_LINK_DIR"

CUSOLVER_REAL="$(find "$CONDA_PREFIX/lib" -maxdepth 1 -type f \
    -name 'libcusolver.so.11.*' -print -quit)"

NVJITLINK_REAL="$(find "$CONDA_PREFIX/lib" -maxdepth 1 -type f \
    -name 'libnvJitLink.so.12.*' -print -quit)"

if [[ -z "$CUSOLVER_REAL" || -z "$NVJITLINK_REAL" ]]; then
    echo "ERROR: No se encontraron las bibliotecas CUDA requeridas." >&2
    exit 1
fi

ln -sfn "$CUSOLVER_REAL" "$CUDA_LINK_DIR/libcusolver.so.11"
ln -sfn "$NVJITLINK_REAL" "$CUDA_LINK_DIR/libnvJitLink.so.12"

export LD_LIBRARY_PATH="$CUDA_LINK_DIR:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ldd "$CUSOLVER_REAL" | grep -q 'not found'; then
    echo "ERROR: Todavía faltan dependencias de cuSOLVER:" >&2
    ldd "$CUSOLVER_REAL" | grep 'not found' >&2
    exit 1
fi

echo "Entrenando QGS con DTU scan${SCAN_ID}"
echo "Configuración: $CONFIG"
echo "Datos: $DATA"
echo "Resultados: $OUTPUT_DIR"

cd "$QGS_DIR"

python train.py \
    --conf_path "../configs/qgs/dtu_scan${SCAN_ID}.yaml"

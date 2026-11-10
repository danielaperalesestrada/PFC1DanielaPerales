#!/usr/bin/env bash
set -euo pipefail

# Directorio raíz del proyecto
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QGS_DIR="$ROOT/QGS"
CONFIG="$ROOT/configs/qgs/dtu_scan24.yaml"
OUTPUT_DIR="$ROOT/outputs/qgs_scan24"

# Comprobaciones previas
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: No se encuentra la configuración: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$QGS_DIR" || ! -d "$ROOT/datasets/dtu/DTU/scan24" ]]; then
    echo "ERROR: No se encuentra QGS o el dataset DTU." >&2
    exit 1
fi

# Evitar sobrescribir resultados anteriores
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: El directorio de salida ya contiene archivos:" >&2
    echo "$OUTPUT_DIR" >&2
    exit 1
fi

cd "$QGS_DIR"

python train.py \
    --conf_path ../configs/qgs/dtu_scan24.yaml

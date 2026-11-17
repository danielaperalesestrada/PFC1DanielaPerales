#!/usr/bin/env bash
#
# Monitoriza una ejecución de entrenamiento QGS o VCR-GauS.
#
# Uso desde la raíz del proyecto:
#   bash scripts/monitoring/monitor_training.sh qgs scan24 -- bash scripts/training/train_qgs.sh
#   bash scripts/monitoring/monitor_training.sh vcr-gaus scan24 -- bash scripts/training/train_vcr_gaus.sh
#
# El entorno Conda debe estar activado antes de ejecutar este script.
#

set -uo pipefail
export LC_ALL=C
usage() {
    cat <<'EOF'
Uso:
  monitor_training.sh <qgs|vcr-gaus> <scan_id> -- <comando> [argumentos...]

Ejemplos:
  conda activate QGS
  bash scripts/monitoring/monitor_training.sh qgs scan24 -- bash scripts/training/train_qgs.sh

  conda activate vcr
  bash scripts/monitoring/monitor_training.sh vcr-gaus scan24 -- bash scripts/training/train_vcr_gaus.sh

El script registra tiempos, métricas GPU/RAM, información del entorno y el log
completo del entrenamiento. No modifica los scripts ni parámetros de entrenamiento.
EOF
}

if [[ $# -lt 4 ]]; then
    usage
    exit 1
fi

METHOD="$1"
SCAN_ID="$2"
shift 2

if [[ "$1" != "--" ]]; then
    echo "Error: falta '--' antes del comando de entrenamiento." >&2
    usage
    exit 1
fi
shift

if [[ $# -eq 0 ]]; then
    echo "Error: no se indicó el comando de entrenamiento." >&2
    usage
    exit 1
fi

case "$METHOD" in
    qgs|vcr-gaus) ;;
    *)
        echo "Error: método no reconocido: $METHOD (usa qgs o vcr-gaus)." >&2
        exit 1
        ;;
esac

# La ubicación del proyecto se obtiene a partir de este archivo:
# scripts/monitoring/monitor_training.sh -> raíz del proyecto.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$PROJECT_ROOT/results/efficiency/training_runs/${METHOD}_${SCAN_ID}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

TRAIN_LOG="$RUN_DIR/training.log"
GPU_CSV="$RUN_DIR/gpu_metrics.csv"
RAM_CSV="$RUN_DIR/ram_metrics.csv"
PROCESS_CSV="$RUN_DIR/gpu_processes.csv"
METADATA="$RUN_DIR/metadata.txt"
SUMMARY="$RUN_DIR/summary.txt"

# Información del equipo y del entorno.
{
    echo "method=$METHOD"
    echo "scan_id=$SCAN_ID"
    echo "metadata_created=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "working_directory=$PROJECT_ROOT"
    echo "conda_environment=${CONDA_DEFAULT_ENV:-not_set}"
    echo "pytorch_cuda_alloc_conf_in_monitor_env=${PYTORCH_CUDA_ALLOC_CONF:-<no definida>}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<no definida>}"
    echo "python_path=$(command -v python 2>/dev/null || true)"
    echo "python_version=$(python --version 2>&1 || true)"
    echo "uname=$(uname -a)"
    echo
    echo "[GPU]"
    nvidia-smi -L 2>&1 || true
    nvidia-smi 2>&1 || true
    echo
    echo "[CUDA / PyTorch]"
    python - <<'PY' 2>&1 || true
try:
    import torch
    print("torch_version:", torch.__version__)
    print("torch_cuda_version:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("No se pudo consultar PyTorch:", repr(exc))
PY
    echo
    echo "[Git]"
    git -C "$PROJECT_ROOT" rev-parse HEAD 2>&1 || true
    git -C "$PROJECT_ROOT/QGS" rev-parse HEAD 2>&1 || true
    git -C "$PROJECT_ROOT/VCR-GauS" rev-parse HEAD 2>&1 || true
    echo
    echo "[Disco al inicio]"
    df -h "$PROJECT_ROOT" 2>&1 || true
    echo
    echo "[Comando]"
    printf '%q ' "$@"
    echo
} > "$METADATA"

# Cabeceras CSV.
echo "timestamp,gpu_name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w,graphics_clock_mhz" > "$GPU_CSV"
echo "timestamp,ram_total_bytes,ram_used_bytes,ram_available_bytes" > "$RAM_CSV"
echo "timestamp,pid,process_name,used_gpu_memory" > "$PROCESS_CSV"

MONITOR_PID=""

monitor_resources() {
    while true; do
        TS="$(date --iso-8601=seconds)"

        # Métricas globales de la GPU. memory.used incluye el uso total de la GPU.
        nvidia-smi \
            --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr \
            --format=csv,noheader,nounits 2>/dev/null |
            while IFS= read -r line; do
                [[ -n "$line" ]] && echo "$line" >> "$GPU_CSV"
            done

        # Memoria GPU atribuida a procesos CUDA, cuando el controlador la expone.
        nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null |
            while IFS= read -r line; do
                [[ -n "$line" ]] && echo "\"$TS\",$line" >> "$PROCESS_CSV"
            done

        # Memoria RAM del sistema.
        if command -v free >/dev/null 2>&1; then
            read -r RAM_TOTAL RAM_USED RAM_FREE < <(free -b | awk '/^Mem:/ {print $2, $3, $7}')
            echo "$TS,$RAM_TOTAL,$RAM_USED,$RAM_FREE" >> "$RAM_CSV"
        fi

        sleep 1
    done
}

monitor_resources &
MONITOR_PID=$!

echo "=================================================="
echo "Monitorizando $METHOD - $SCAN_ID"
echo "Resultados: $RUN_DIR"
echo "=================================================="
echo

START_ISO="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s.%N)"
echo "Inicio: $START_ISO"

# Ejecutar exactamente el comando recibido y conservar su salida.
set +e
"$@" 2>&1 | tee "$TRAIN_LOG"
TRAIN_STATUS=${PIPESTATUS[0]}

END_ISO="$(date --iso-8601=seconds)"
END_EPOCH="$(date +%s.%N)"

# Detener el monitor y cerrar sus archivos.
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

# Duración y resumen de métricas.
DURATION="$(awk -v start="$START_EPOCH" -v end="$END_EPOCH" 'BEGIN {printf "%.3f", end-start}')"
GPU_MAX="$(awk -F',' 'NR>1 && $4 ~ /[0-9]/ {v=$4+0; if (!f || v>m) m=v; f=1} END {if (!f) print "N/D"; else printf "%.0f", m}' "$GPU_CSV")"
GPU_UTIL_MAX="$(awk -F',' 'NR>1 && $3 ~ /[0-9]/ {v=$3+0; if (!f || v>m) m=v; f=1} END {if (!f) print "N/D"; else printf "%.0f", m}' "$GPU_CSV")"
GPU_TEMP_MAX="$(awk -F',' 'NR>1 {gsub(/[^0-9.]/,"",$6); if ($6+0>max) max=$6+0} END {if (max=="") print "N/D"; else printf "%.0f", max}' "$GPU_CSV")"
GPU_POWER_MAX="$(awk -F',' 'NR>1 {gsub(/[^0-9.]/,"",$7); if ($7+0>max) max=$7+0} END {if (max=="") print "N/D"; else printf "%.1f", max}' "$GPU_CSV")"
RAM_MAX="$(awk -F',' 'NR>1 && $3 ~ /^[0-9]+$/ {if ($3>max) max=$3} END {if (max=="") print "N/D"; else printf "%.0f", max}' "$RAM_CSV")"
PROC_MAX="$(awk -F',' 'NR>1 {gsub(/[^0-9]/,"",$4); s[$1]+=$4} END {for (t in s) if (s[t]>m) m=s[t]; printf "%d", m+0}' "$PROCESS_CSV")"
PROC_MAX="$(awk -F',' 'NR>1 {gsub(/[^0-9]/,"",$4); s[$1]+=$4} END {for (t in s) if (s[t]>m) m=s[t]; printf "%d", m+0}' "$PROCESS_CSV")"

{
    echo "method=$METHOD"
    echo "scan_id=$SCAN_ID"
    echo "start_time=$START_ISO"
    echo "end_time=$END_ISO"
    echo "duration_seconds=$DURATION"
    echo "training_exit_code=$TRAIN_STATUS"
    echo "gpu_memory_max_mib=$GPU_MAX"
    echo "gpu_process_memory_max_mib=$PROC_MAX"
    echo "gpu_utilization_max_percent=$GPU_UTIL_MAX"
    echo "gpu_temperature_max_c=$GPU_TEMP_MAX"
    echo "gpu_power_max_w=$GPU_POWER_MAX"
    echo "system_ram_used_max_bytes=$RAM_MAX"
    echo
    echo "[Disk al finalizar]"
    df -h "$PROJECT_ROOT" 2>&1 || true
    echo "gpu_process_memory_max_mib=$PROC_MAX"
} > "$SUMMARY"

echo
echo "=================================================="
echo "Entrenamiento finalizado."
echo "Código de salida: $TRAIN_STATUS"
echo "Duración: $DURATION segundos"
echo "Pico de memoria GPU observado: $GPU_MAX MiB"
echo "Resultados guardados en: $RUN_DIR"
echo "=================================================="

exit "$TRAIN_STATUS"

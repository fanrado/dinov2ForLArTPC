#!/bin/bash
# ============================================================================
# Run linear evaluation of DINOv2 backbone on CVN data
# Usage:
#   bash run_linear_cvn.sh --model-config <path> --data-config <path> [--device <device>]
#
# Example:
#   bash run_linear_cvn.sh \
#       --model-config dinov2/configs/linear_cvn_model_config.yaml \
#       --data-config  dinov2/configs/linear_cvn_data_config.yaml \
#       --device cuda:0
# ============================================================================

set -euo pipefail

# ---- Default config paths (edit or override via CLI) -----------------------
MODEL_CONFIG="dinov2/configs/linear_cvn_model_config.yaml"
DATA_CONFIG="dinov2/configs/linear_cvn_data_config.yaml"
DEVICE=""

# ---- Parse command-line arguments ------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-config)
            MODEL_CONFIG="$2"; shift 2 ;;
        --data-config)
            DATA_CONFIG="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_linear_cvn.sh --model-config <path> --data-config <path> [--device <device>]"
            exit 1 ;;
    esac
done

# ---- Build the python command ----------------------------------------------
CMD="python -m dinov2.eval.linear_cvn \
    --model-config ${MODEL_CONFIG} \
    --data-config  ${DATA_CONFIG}"

if [[ -n "${DEVICE}" ]]; then
    CMD="${CMD} --device ${DEVICE}"
fi

echo "============================================"
echo "Running linear CVN evaluation"
echo "  Model config : ${MODEL_CONFIG}"
echo "  Data config  : ${DATA_CONFIG}"
echo "  Device       : ${DEVICE:-auto (cuda if available)}"
echo "============================================"

${CMD}

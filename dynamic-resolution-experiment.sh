#!/usr/bin/env bash
set -euo pipefail

# Dynamic-resolution MRI SR training entry point.
# Override CONFIG/GPU_ID/EXP_NAME/RESUME/TAG from the shell when needed.

CONFIG="${CONFIG:-configs/train/train_mri_seg_mask.yaml}"
# CONFIG="${CONFIG:-configs/train/train-mri_paired_swinir_syn_seg_k_4.0.yaml}"

GPU_ID="${GPU_ID:-0}"
EXP_NAME="${EXP_NAME:-mri_dynamic_resolution}"
TAG="${TAG:-}"
RESUME="${RESUME:-}"
RUN_TEST="${RUN_TEST:-1}"
TEST_CONFIG_DYNAMIC="${TEST_CONFIG_DYNAMIC:-configs/test/test_dynamic_train_mri_seg_mask.yaml}"
TEST_CONFIG_FASTMRI="${TEST_CONFIG_FASTMRI:-configs/test/test_fastmri_static_k_4.0.yaml}"

cmd=(
  python train_mri.py
  --config "${CONFIG}"
  --gpu "${GPU_ID}"
  --name "${EXP_NAME}"
)

if [[ -n "${TAG}" ]]; then
  cmd+=(--tag "${TAG}")
fi

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

echo "==> Training MRI SR model"
echo "Config: ${CONFIG}"
echo "Save dir: save/${EXP_NAME}${TAG:+_${TAG}}"
"${cmd[@]}"

SAVE_NAME="${EXP_NAME}${TAG:+_${TAG}}"
BEST_CKPT="save/${SAVE_NAME}/checkpoint_best.pth"

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "Best checkpoint not found: ${BEST_CKPT}" >&2
  exit 1
fi

if [[ "${RUN_TEST}" != "1" ]]; then
  echo "==> Training finished; RUN_TEST=${RUN_TEST}, skipping test."
  exit 0
fi

CONFIG_BASENAME="$(basename "${CONFIG}")"
case "${CONFIG_BASENAME}" in
  train_mri_seg_mask.yaml)
    echo "==> Testing with test.py"
    python test.py \
      --config "${TEST_CONFIG_DYNAMIC}" \
      --checkpoint "${BEST_CKPT}" \
      --device "cuda"
    ;;

  train-mri_paired_swinir_syn_seg_k_4.0.yaml)
    echo "==> Testing with test_fastMRI.py"
    python test_fastMRI.py \
      --test_config "${TEST_CONFIG_FASTMRI}" \
      --checkpoint "${BEST_CKPT}" \
      --gpu "${GPU_ID}"
    ;;

  *)
    echo "No test route configured for CONFIG=${CONFIG}" >&2
    echo "Supported configs:" >&2
    echo "  - configs/train/train_mri_seg_mask.yaml -> test.py" >&2
    echo "  - configs/train/train-mri_paired_swinir_syn_seg_k_4.0.yaml -> test_fastMRI.py" >&2
    exit 1
    ;;
esac

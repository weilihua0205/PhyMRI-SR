#!/usr/bin/env bash
set -euo pipefail

# 3T-to-5T fine-tuning pipeline:
# 1. Train the base MRI SR model with train_mri.py.
# 2. Fine-tune that best checkpoint on paired 3T/5T data.
# 3. Test the fine-tuned best checkpoint with test.py.

GPU_ID="${GPU_ID:-0}"
RUN_TEST="${RUN_TEST:-1}"

BASE_CONFIG="${BASE_CONFIG:-configs/train/train_mri_seg_mask.yaml}"
BASE_NAME="${BASE_NAME:-mri_seg_mask_pretrain_for_3t5t}"

FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/train/train-mri_finetune_3t5t.yaml}"
FINETUNE_NAME="${FINETUNE_NAME:-finetune_3t5t_from_train_mri_seg_mask}"

TEST_CONFIG="${TEST_CONFIG:-configs/test/test_3t5t_finetune.yaml}"

BASE_BEST="save/${BASE_NAME}/checkpoint_best.pth"
FINETUNE_BEST="save/${FINETUNE_NAME}/checkpoint_best.pth"

echo "==> [1/3] Training base MRI SR model"
python train_mri.py \
  --config "${BASE_CONFIG}" \
  --gpu "${GPU_ID}" \
  --name "${BASE_NAME}"

if [[ ! -f "${BASE_BEST}" ]]; then
  echo "Base best checkpoint not found: ${BASE_BEST}" >&2
  exit 1
fi

echo "==> [2/3] Fine-tuning on paired 3T/5T data"
python train_3t5t_finetune.py \
  --config "${FINETUNE_CONFIG}" \
  --pretrain "${BASE_BEST}" \
  --gpu "${GPU_ID}" \
  --save-name "${FINETUNE_NAME}"

if [[ ! -f "${FINETUNE_BEST}" ]]; then
  echo "Fine-tuned best checkpoint not found: ${FINETUNE_BEST}" >&2
  exit 1
fi

if [[ "${RUN_TEST}" != "1" ]]; then
  echo "==> Fine-tuning finished; RUN_TEST=${RUN_TEST}, skipping test."
  exit 0
fi

echo "==> [3/3] Testing fine-tuned checkpoint"
python test.py \
  --config "${TEST_CONFIG}" \
  --checkpoint "${FINETUNE_BEST}" \
  --device "cuda"

echo "==> Done"

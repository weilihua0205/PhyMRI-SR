#!/usr/bin/env bash
set -euo pipefail

# Static-resolution MRI SR experiment pipeline:
# 1. Train the base model with train_mri.py.
# 2. Initialize meta_train_mri.py from the best base checkpoint.
# 3. Test the best meta checkpoint with test.py using a YAML test config.

GPU_ID="${GPU_ID:-0}"

BASE_CONFIG="${BASE_CONFIG:-configs/train/train_mri_seg_mask.yaml}"
META_CONFIG="${META_CONFIG:-configs/train/train-mri-meta-template.yaml}"
TEST_CONFIG="${TEST_CONFIG:-configs/test/test_static.yaml}"

BASE_NAME="${BASE_NAME:train}"
META_NAME="${META_NAME:-train_meta}"

BASE_BEST="save/${BASE_NAME}/checkpoint_best.pth"
META_BEST="save/meta_learning/${META_NAME}/checkpoint_best.pth"

echo "==> [1/3] Training base model"
python train_mri.py \
  --config "${BASE_CONFIG}" \
  --gpu "${GPU_ID}" \
  --name "${BASE_NAME}"

if [[ ! -f "${BASE_BEST}" ]]; then
  echo "Base checkpoint not found: ${BASE_BEST}" >&2
  exit 1
fi

echo "==> [2/3] Meta-training from base best checkpoint"
python meta_train_mri.py \
  --config "${META_CONFIG}" \
  --pretrained "${BASE_BEST}" \
  --gpu "${GPU_ID}" \
  --name "${META_NAME}"

if [[ ! -f "${META_BEST}" ]]; then
  echo "Meta checkpoint not found: ${META_BEST}" >&2
  exit 1
fi

echo "==> [3/3] Testing meta best checkpoint"
python test.py \
  --config "${TEST_CONFIG}" \
  --checkpoint "${META_BEST}"

echo "==> Done"

import os
import argparse
import yaml
import torch
import copy
import numpy as np

import datasets
import models
from evaluate import validate
from utils import log


def load_config(save_dir):
    cfg_path = os.path.join(save_dir, 'config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def build_val_loader(config):
    if 'val_dataset' not in config:
        return None
    val_spec = config['val_dataset']
    val_dataset = datasets.make(val_spec['dataset'])
    val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=val_spec.get('batch_size', 1), shuffle=False, num_workers=4, pin_memory=True)
    return val_loader


def load_model_from_checkpoint(checkpoint_path, config, device='cuda'):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_spec = ck['model']
    model = models.make(model_spec)
    model.load_state_dict(model_spec.get('sd', ck['model'].get('sd', {})))
    model = model.to(device)
    return model, ck


def state_dict_from_checkpoint(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return ck['model']['sd'] if 'model' in ck and 'sd' in ck['model'] else ck.get('model', {}).get('sd', None)


def summarize_param_diffs(sd_a, sd_b, topk=10):
    diffs = []
    for k in sd_a:
        a = sd_a[k]
        b = sd_b.get(k)
        if b is None:
            diffs.append((k, float('nan'), 'missing_in_b'))
            continue
        d = (a - b).abs()
        maxd = float(d.max().item())
        meand = float(d.mean().item())
        diffs.append((k, maxd, meand))
    diffs_sorted = sorted(diffs, key=lambda x: (np.isnan(x[1]), -x[1] if not np.isnan(x[1]) else 0))
    return diffs_sorted[:topk], len(diffs)


def check_params_for_nan_inf(sd):
    nan_keys = []
    inf_keys = []
    for k, v in sd.items():
        if torch.isinf(v).any():
            inf_keys.append(k)
        if torch.isnan(v).any():
            nan_keys.append(k)
    return nan_keys, inf_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_dir = args.save_dir
    best_ck = os.path.join(save_dir, 'checkpoint_best.pth')
    latest_ck = os.path.join(save_dir, 'checkpoint_latest.pth')

    if not os.path.exists(best_ck):
        print('No checkpoint_best.pth found')
        return
    if not os.path.exists(latest_ck):
        print('No checkpoint_latest.pth found')
        return

    config = load_config(save_dir)
    val_loader = build_val_loader(config)

    print('Loading checkpoints (into CPU state dicts) ...')
    sd_best = state_dict_from_checkpoint(best_ck)
    sd_latest = state_dict_from_checkpoint(latest_ck)

    print('Checking NaN/Inf in parameters...')
    b_nans, b_infs = check_params_for_nan_inf(sd_best)
    l_nans, l_infs = check_params_for_nan_inf(sd_latest)
    print(f'Best checkpoint: {len(b_nans)} NaN keys, {len(b_infs)} Inf keys')
    if len(b_nans) > 0:
        print('NaN keys (best):', b_nans[:10])
    if len(b_infs) > 0:
        print('Inf keys (best):', b_infs[:10])
    print(f'Latest checkpoint: {len(l_nans)} NaN keys, {len(l_infs)} Inf keys')
    if len(l_nans) > 0:
        print('NaN keys (latest):', l_nans[:10])
    if len(l_infs) > 0:
        print('Inf keys (latest):', l_infs[:10])

    print('\nTop parameter diffs (best vs latest):')
    topk, total = summarize_param_diffs(sd_best, sd_latest, topk=20)
    for k, maxd, meand in topk:
        print(f'  {k}  max_diff={maxd:.6e}  mean_diff={meand:.6e}')
    print(f'  ... ({total} param blobs compared)')

    # Evaluate both on validation set if available
    if val_loader is not None:
        print('\nEvaluating best checkpoint on validation set...')
        model_best, ck_best = load_model_from_checkpoint(best_ck, config)
        psnr_b, ssim_b = validate(model_best, val_loader, config, verbose=False, compute_ssim=True)
        print(f'  Best: PSNR={psnr_b:.4f} dB, SSIM={ssim_b:.6f}')

        print('Evaluating latest checkpoint on validation set...')
        model_latest, ck_latest = load_model_from_checkpoint(latest_ck, config)
        psnr_l, ssim_l = validate(model_latest, val_loader, config, verbose=False, compute_ssim=True)
        print(f'  Latest: PSNR={psnr_l:.4f} dB, SSIM={ssim_l:.6f}')

        # Also compute per-parameter norms between loaded models
        sd_b_loaded = model_best.state_dict()
        sd_l_loaded = model_latest.state_dict()
        topk_loaded, total_loaded = summarize_param_diffs(sd_b_loaded, sd_l_loaded, topk=20)
        print('\nTop diffs (loaded models):')
        for k, maxd, meand in topk_loaded:
            print(f'  {k}  max_diff={maxd:.6e}  mean_diff={meand:.6e}')

    print('\nDone')

if __name__ == "__main__":
    main()

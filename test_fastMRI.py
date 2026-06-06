import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from evaluate import calc_psnr, calc_ssim


# =============================================================================
# ★ 直接配置区（服务器上直接修改这里，无需命令行参数）
#   命令行参数优先级更高；若命令行未传入，则使用此处的默认值。
# =============================================================================
RUN_CONFIG = {
    # 训练时使用的 config.yaml 路径
    'config':       './configs/train_kspace_fastmri/train-mri_paired_swinir_syn_seg_k_6.4.yaml',

    # 模型权重路径（通常选 checkpoint_best.pth 或 checkpoint_latest.pth）
    #./save/mri_syn_swinir_old_mri_dictionary_seg_4times_correct_matched_phy_kspace4.0_fastmri1200_3/checkpoint_best.pth
    #./save/mri_syn_swinir_old_mri_dictionary_seg_4times_correct_matched_phy_kspace5.0_fastmri1200_3/checkpoint_best.pth
    #./save_fastmri_320_1200/mri_syn_swinir_old_mri_dictionary_seg_4times_correct_matched_phy_kspace6.4_fastmri1200/checkpoint_best.pth
    'checkpoint':   './save_fastmri_320_1200/mri_syn_swinir_old_mri_dictionary_seg_4times_correct_matched_phy_kspace6.4_fastmri1200/checkpoint_best.pth',

    # LR / HR 数据目录（存放 .npy 文件）
    'lr_dir':       './kspace_fastmridata_320_1344_test_intersect/scale6.4_320_npy/LR',
    'hr_dir':       './kspace_fastmridata_320_1344_test_intersect/scale6.4_320_npy/HR',
    # 结果保存目录
    'save_dir':     './inference_output_1344_check_vis/kspace_6.4_320',

    # 使用的 GPU id（字符串，如 '0' 或 '0,1'）
    'gpu':          '0',

    # DataLoader 设置
    'batch_size':   1,
    'num_workers':  4,

    # 保存可视化对比图的数量
    'num_vis':      145,

    # 只评估前 K 个样本（None = 全部评估）
    'first_k':      None,

    # 数据集缓存模式：'none' | 'bin' | 'in_memory'
    'cache':        'in_memory',

    # Tile 推理设置（显存不足时开启）
    # tile_size=None 表示直接全图推理；显存不足时改为 48 或 64
    # tile_overlap 建议设为 tile_size 的 1/4 ~ 1/3，配合 Gaussian 窗口消除接缝
    'tile_size':    64,
    'tile_overlap': 16,
}
# =============================================================================


def load_test_config(config_path):
    if not config_path:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config is None:
        return {}
    if 'test' in config and isinstance(config['test'], dict):
        config = config['test']
    return config


def resolve_data_dirs(args):
    lr_dir = args.lr_dir
    hr_dir = args.hr_dir
    if args.data_root:
        lr_dir = lr_dir or os.path.join(args.data_root, args.lr_subdir)
        hr_dir = hr_dir or os.path.join(args.data_root, args.hr_subdir)
    if not lr_dir:
        raise ValueError('Missing LR directory: set lr_dir or data_root in --test_config, or pass --lr_dir.')
    if not hr_dir:
        raise ValueError('Missing HR directory: set hr_dir or data_root in --test_config, or pass --hr_dir.')
    return lr_dir, hr_dir


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--test_config', default=None, help='optional test YAML config path')
    bootstrap_args, _ = bootstrap.parse_known_args()

    defaults = dict(RUN_CONFIG)
    defaults.update(load_test_config(bootstrap_args.test_config))

    parser = argparse.ArgumentParser(
        description='Evaluate a checkpoint on held-out real paired MRI data and save visualizations.',
        parents=[bootstrap],
    )
    # Values from RUN_CONFIG are defaults; --test_config can replace them, and direct CLI args override both.
    parser.add_argument('--config',       default=defaults['config'],       help='training config path')
    parser.add_argument('--checkpoint',   default=defaults['checkpoint'],   help='checkpoint path')
    parser.add_argument('--data_root',    default=defaults.get('data_root'), help='root containing LR/HR subdirectories')
    parser.add_argument('--lr_subdir',    default=defaults.get('lr_subdir', 'LR'), help='LR subdirectory under data_root')
    parser.add_argument('--hr_subdir',    default=defaults.get('hr_subdir', 'HR'), help='HR subdirectory under data_root')
    parser.add_argument('--lr_dir',       default=defaults.get('lr_dir'),       help='directory of held-out real LR .npy files')
    parser.add_argument('--hr_dir',       default=defaults.get('hr_dir'),       help='directory of held-out real HR .npy files')
    parser.add_argument('--save_dir',     default=defaults['save_dir'],     help='directory to save metrics and visualizations')
    parser.add_argument('--gpu',          default=defaults['gpu'],          help='GPU id')
    parser.add_argument('--batch_size',   type=int, default=defaults['batch_size'],   help='evaluation batch size')
    parser.add_argument('--num_workers',  type=int, default=defaults['num_workers'],  help='number of dataloader workers')
    parser.add_argument('--num_vis',      type=int, default=defaults['num_vis'],      help='number of samples to visualize')
    parser.add_argument('--first_k',      type=int, default=defaults['first_k'],      help='only evaluate the first K samples')
    parser.add_argument('--cache',        default=defaults['cache'],
                        choices=['none', 'bin', 'in_memory'], help='dataset cache mode')
    parser.add_argument('--tile_size',    type=int, default=defaults['tile_size'],
                        help='LR tile size for memory-safe inference, e.g. 48 or 64')
    parser.add_argument('--tile_overlap', type=int, default=defaults['tile_overlap'],
                        help='overlap between LR tiles when tile inference is enabled')
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' not in checkpoint:
        raise KeyError(f'Checkpoint at {checkpoint_path} does not contain a "model" entry.')
    model = models.make(checkpoint['model'], load_sd=True)
    model = model.to(device)
    model.eval()
    return model


def build_loader(lr_dir, hr_dir, cache, batch_size, num_workers, first_k=None):
    dataset_spec = {
        'name': 'paired-npy-folders',
        'args': {
            'root_path_1': lr_dir,
            'root_path_2': hr_dir,
            'cache': cache,
            'repeat': 1,
            'first_k': first_k,
        }
    }
    wrapper_spec = {
        'name': 'sr-implicit-paired',
        'args': {
            'inp_size': None,
            'augment': False,
        }
    }
    base_dataset = datasets.make(dataset_spec)
    wrapped_dataset = datasets.make(wrapper_spec, args={'dataset': base_dataset})
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_norm(config, device):
    data_norm = config.get('data_norm')
    if not data_norm:
        return None
    return {
        'inp_sub': torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).to(device),
        'inp_div': torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).to(device),
        'gt_sub': torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).to(device),
        'gt_div': torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).to(device),
    }


def tensor_to_display_image(tensor):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0], 'gray'
    if array.ndim == 3 and array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0)), None
    raise ValueError(f'Unsupported tensor shape for display: {tensor.shape}')


def save_sample_figure(lr_up, pred, gt, filename, psnr, ssim, save_path):
    error_map = torch.abs(pred - gt)
    lr_img, lr_cmap = tensor_to_display_image(lr_up)
    pred_img, pred_cmap = tensor_to_display_image(pred)
    gt_img, gt_cmap = tensor_to_display_image(gt)
    err_img, _ = tensor_to_display_image(error_map)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f'{filename} | PSNR {psnr:.4f} dB | SSIM {ssim:.4f}', fontsize=12)

    axes[0].imshow(lr_img, cmap=lr_cmap, vmin=0, vmax=1)
    axes[0].set_title('LR Bicubic')
    axes[1].imshow(pred_img, cmap=pred_cmap, vmin=0, vmax=1)
    axes[1].set_title('SR')
    axes[2].imshow(gt_img, cmap=gt_cmap, vmin=0, vmax=1)
    axes[2].set_title('HR')
    axes[3].imshow(err_img, cmap='hot')
    axes[3].set_title('|SR - HR|')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_individual_images(lr_up, pred, gt, vis_imgs_dir, stem):
    """将 LR / SR / HR 三张小图分别以无框形式保存到 vis_imgs/{LR,SR,HR} 子文件夹。

    所有图像使用相同的 DPI 和 figsize，因此输出像素尺寸完全一致，
    可直接用于论文组图，无需裁剪。
    """
    def _save_one(tensor, sub_dir, fname):
        img, cmap = tensor_to_display_image(tensor)
        h, w = img.shape[:2]
        dpi = 100
        # figsize 精确匹配像素数，确保各子图输出尺寸完全相同
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])   # 充满整个画布，无边距
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        os.makedirs(sub_dir, exist_ok=True)
        fig.savefig(os.path.join(sub_dir, fname), dpi=dpi,
                    bbox_inches=None, pad_inches=0)
        plt.close(fig)

    def _save_delta(tensor_a, tensor_b, sub_dir, fname):
        """保存 |SR - HR| 差分图，使用 'hot' colormap，值域 [0, 1]。"""
        diff = torch.abs(tensor_a - tensor_b)
        img, _ = tensor_to_display_image(diff)
        h, w = img.shape[:2]
        dpi = 100
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        os.makedirs(sub_dir, exist_ok=True)
        fig.savefig(os.path.join(sub_dir, fname), dpi=dpi,
                    bbox_inches=None, pad_inches=0)
        plt.close(fig)

    fname = f'{stem}.png'
    _save_one(lr_up, os.path.join(vis_imgs_dir, 'LR'), fname)
    _save_one(pred,  os.path.join(vis_imgs_dir, 'SR'), fname)
    _save_one(gt,    os.path.join(vis_imgs_dir, 'HR'), fname)
    _save_delta(pred, gt, os.path.join(vis_imgs_dir, 'Delta'), fname)


def _make_gaussian_window(h, w, device, dtype):
    """生成 2D Gaussian 权重窗口，中心权重高、边缘接近 0，用于消除 tile 拼接接缝。"""
    def gauss_1d(n):
        # sigma = n/6 使边缘权重约为中心的 1%
        sigma = n / 6.0
        x = torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2.0
        return torch.exp(-0.5 * (x / sigma) ** 2)

    win_h = gauss_1d(h)
    win_w = gauss_1d(w)
    window = win_h[:, None] * win_w[None, :]   # (h, w)
    return window.clamp_min(1e-6)


def infer_one(model, inp, scale, tile_size=None, tile_overlap=8):
    if tile_size is None:
        return model(inp, scale)

    bs, c, h, w = inp.shape
    if bs != 1:
        raise ValueError('Tile inference currently expects batch_size=1.')

    if scale.ndim == 2 and scale.shape[-1] == 2:
        scale_h = float(scale[0, 0].item())
        scale_w = float(scale[0, 1].item())
    else:
        scale_h = float(scale.reshape(-1)[0].item())
        scale_w = scale_h

    # overlap 建议至少为 tile_size 的 1/4，以保证边缘有足够的 Gaussian 衰减
    stride = max(tile_size - tile_overlap, 1)
    h_positions = list(range(0, max(h - tile_size, 0) + 1, stride))
    w_positions = list(range(0, max(w - tile_size, 0) + 1, stride))
    if not h_positions or h_positions[-1] != max(h - tile_size, 0):
        h_positions.append(max(h - tile_size, 0))
    if not w_positions or w_positions[-1] != max(w - tile_size, 0):
        w_positions.append(max(w - tile_size, 0))

    out_h = round(h * scale_h)
    out_w = round(w * scale_w)
    output = torch.zeros((1, c, out_h, out_w), device=inp.device, dtype=inp.dtype)
    weight = torch.zeros((1, 1, out_h, out_w), device=inp.device, dtype=inp.dtype)

    for y in h_positions:
        for x in w_positions:
            lr_tile = inp[:, :, y:min(y + tile_size, h), x:min(x + tile_size, w)]
            tile_pred = model(lr_tile, scale).clamp(0, 1)

            y_hr = round(y * scale_h)
            x_hr = round(x * scale_w)
            tile_h_hr = tile_pred.shape[-2]
            tile_w_hr = tile_pred.shape[-1]

            # ── Gaussian 窗口加权融合，消除 tile 边缘接缝 ──────────────────
            win = _make_gaussian_window(tile_h_hr, tile_w_hr,
                                        device=inp.device, dtype=inp.dtype)
            win = win.unsqueeze(0).unsqueeze(0)   # (1,1,th,tw)

            output[:, :, y_hr:y_hr + tile_h_hr, x_hr:x_hr + tile_w_hr] += tile_pred * win
            weight[:, :, y_hr:y_hr + tile_h_hr, x_hr:x_hr + tile_w_hr] += win

    return output / weight.clamp_min(1e-6)


def evaluate_and_visualize(model, loader, config, save_dir, num_vis, device,
                           tile_size=None, tile_overlap=8):
    ensure_dir(save_dir)
    vis_dir = os.path.join(save_dir, 'visualizations')
    ensure_dir(vis_dir)
    
    vis_imgs_dir = os.path.join(save_dir, 'vis_imgs')
    for sub in ('LR', 'SR', 'HR', 'Delta'):
        ensure_dir(os.path.join(vis_imgs_dir, sub))

    norms = prepare_norm(config, device)
    psnr_values = []
    ssim_values = []
    per_sample = []

    progress = tqdm(loader, desc='Evaluating', ncols=100)
    vis_saved = 0
    sample_idx = 0  # 全局样本计数器，代替 filename

    with torch.no_grad():
        for batch in progress:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            lr_orig = batch['inp']
            if norms is not None:
                inp = (lr_orig - norms['inp_sub']) / norms['inp_div']
            else:
                inp = lr_orig

            scale = batch['scale']
            gt = batch['gt']
            try:
                pred = infer_one(
                    model,
                    inp,
                    scale,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    'CUDA out of memory during inference. '
                    'Please rerun with --tile_size 48 or --tile_size 64 '
                    '(and keep --batch_size 1).'
                ) from exc

            if norms is not None:
                pred = pred * norms['gt_div'] + norms['gt_sub']
            pred = pred.clamp(0, 1)

            lr_up = torch.nn.functional.interpolate(
                lr_orig, size=gt.shape[-2:], mode='bicubic', align_corners=False
            ).clamp(0, 1)

            batch_size = gt.shape[0]
            for i in range(batch_size):
                # dataset 未提供 filename，用全局序号代替
                filename = batch['filename'][i] if 'filename' in batch else f'sample_{sample_idx:04d}'
                sample_idx += 1
                pred_i = pred[i:i + 1]
                gt_i = gt[i:i + 1]
                lr_up_i = lr_up[i:i + 1]

                psnr = float(calc_psnr(pred_i, gt_i, dataset=config.get('eval_type', None)).item())
                ssim = float(calc_ssim(pred_i, gt_i, dataset=config.get('eval_type', None)).item())

                psnr_values.append(psnr)
                ssim_values.append(ssim)
                per_sample.append({
                    'filename': filename,
                    'psnr': psnr,
                    'ssim': ssim,
                })

                if num_vis is None or vis_saved < num_vis:
                    base = os.path.splitext(os.path.basename(filename))[0] if os.sep in filename or '.' in filename else filename
                    fig_name = f'{vis_saved:03d}_{base}.png'
                    save_sample_figure(
                        lr_up_i[0], pred_i[0], gt_i[0], filename, psnr, ssim,
                        os.path.join(vis_dir, fig_name)
                    )
                    
                    # 额外保存单独的 LR / SR / HR 小图（无框，尺寸一致，适合论文组图）
                    stem = f'{vis_saved:03d}_{base}'
                    save_individual_images(
                        lr_up_i[0], pred_i[0], gt_i[0], vis_imgs_dir, stem
                    )
                    vis_saved += 1

            progress.set_postfix({
                'PSNR': f'{np.mean(psnr_values):.4f}',
                'SSIM': f'{np.mean(ssim_values):.4f}',
            })

    mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else 0.0

    per_sample_sorted = sorted(per_sample, key=lambda item: item['psnr'], reverse=True)
    summary = {
        'num_samples': len(per_sample),
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'best_sample': per_sample_sorted[0] if per_sample_sorted else None,
        'worst_sample': per_sample_sorted[-1] if per_sample_sorted else None,
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, 'per_sample_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(per_sample_sorted, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Num samples: {summary["num_samples"]}\n')
        f.write(f'Mean PSNR: {summary["mean_psnr"]:.4f} dB\n')
        f.write(f'Mean SSIM: {summary["mean_ssim"]:.4f}\n')
        if summary['best_sample'] is not None:
            f.write(
                'Best sample: '
                f'{summary["best_sample"]["filename"]} | '
                f'PSNR {summary["best_sample"]["psnr"]:.4f} dB | '
                f'SSIM {summary["best_sample"]["ssim"]:.4f}\n'
            )
        if summary['worst_sample'] is not None:
            f.write(
                'Worst sample: '
                f'{summary["worst_sample"]["filename"]} | '
                f'PSNR {summary["worst_sample"]["psnr"]:.4f} dB | '
                f'SSIM {summary["worst_sample"]["ssim"]:.4f}\n'
            )

    return summary


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = load_config(args.config)
    ensure_dir(args.save_dir)

    model = load_model(args.checkpoint, device)
    lr_dir, hr_dir = resolve_data_dirs(args)
    loader = build_loader(
        lr_dir,
        hr_dir,
        args.cache,
        args.batch_size,
        args.num_workers,
        first_k=args.first_k,
    )

    summary = evaluate_and_visualize(
        model=model,
        loader=loader,
        config=config,
        save_dir=args.save_dir,
        num_vis=args.num_vis,
        device=device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )

    print('=' * 60)
    print('Evaluation finished')
    print(f'Number of samples: {summary["num_samples"]}')
    print(f'Mean PSNR: {summary["mean_psnr"]:.4f} dB')
    print(f'Mean SSIM: {summary["mean_ssim"]:.4f}')
    print(f'Results saved to: {args.save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""诊断可视化问题的脚本

用法示例：
  python tools/diagnose_visualization.py \
    --config configs/train/train-div2k.yaml \
    --model save/div2k_sample_swinir_1000epochs/checkpoint_latest.pth \
    --image natural_images/DIV2K_train_HR_sample/0001.png \
    --scale 4,4 \
    --outdir /tmp/diag

脚本会：
- 加载模型并对单张图片做一次前向
- 打印 pred 的总体/按行/按通道统计（min/max/mean/percentiles）
- 计算奇偶行均值差异以检测交替行伪影
- 保存每个通道的灰度可视化图片（pred_channel_R/G/B.png）和合成图
- （可选）尝试输出 models.gaussian.get_coord 的示例坐标以检查 x/y 顺序
"""
import os
import argparse
import yaml
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image

def parse_scale(s):
    parts = s.split(',')
    if len(parts) == 1:
        v = float(parts[0])
        return torch.tensor([[v, v]], dtype=torch.float32).cuda()
    else:
        return torch.tensor([[float(parts[0]), float(parts[1])]], dtype=torch.float32).cuda()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--scale', default='4,4')
    parser.add_argument('--outdir', default='save/diagnostics')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load config (not strictly necessary but useful)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # load model spec from checkpoint
    ck = torch.load(args.model, map_location='cpu')
    if 'model' in ck:
        model_spec = ck['model']
    else:
        # assume full model saved
        model_spec = ck

    import models
    model = models.make(model_spec, load_sd=True).cuda().eval()

    img = Image.open(args.image).convert('RGB')
    # Resize to smaller size to avoid OOM (original 1404x2040 -> 512x512)
    img = img.resize((256, 256), Image.BICUBIC)
    img_t = transforms.ToTensor()(img).cuda()

    scale = parse_scale(args.scale)

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0), scale).squeeze(0).clamp(0,1).cpu()

    C,H,W = pred.shape
    print(f'Pred shape: C={C}, H={H}, W={W}')

    arr = pred.numpy()
    print('Overall stats: min={:.6f}, max={:.6f}, mean={:.6f}'.format(arr.min(), arr.max(), arr.mean()))
    pcts = np.percentile(arr, [0,1,5,25,50,75,95,99,100])
    print('Percentiles:', [f'{x:.4f}' for x in pcts])

    # per-row mean (all channels)
    row_means = pred.permute(1,2,0).reshape(H, -1).mean(dim=1).numpy()
    print('Row means (first 60):', np.round(row_means[:60],4).tolist())

    # odd/even row statistics
    odd_mean = float(row_means[1::2].mean()) if H>1 else float(row_means.mean())
    even_mean = float(row_means[0::2].mean())
    print(f'Even rows mean={even_mean:.6f}, Odd rows mean={odd_mean:.6f}, diff={abs(even_mean-odd_mean):.6f}')

    # per-channel per-row means (print first 20)
    for c,name in enumerate(['R','G','B'][:C]):
        ch_row = pred[c].mean(dim=1).numpy()
        print(f'{name} channel first 20 row means:', np.round(ch_row[:20],4).tolist())

    # print top-left 12x12 pixels
    arr_hwc = pred.permute(1,2,0).numpy()
    print('Top-left 12x12 pixels (R,G,B) rounded:')
    for r in range(min(12,H)):
        row = [list(np.round(arr_hwc[r,c,:],3)) for c in range(min(12,W))]
        print(row)

    # save per-channel images
    for c,name in enumerate(['R','G','B'][:C]):
        out_path = os.path.join(args.outdir, f'pred_channel_{name}.png')
        # save single-channel as grayscale (0-1 -> 0-255)
        ch = pred[c].unsqueeze(0)
        save_image(ch, out_path, normalize=False)
        print('Saved', out_path)

    # save combined color image
    color_out = os.path.join(args.outdir, 'pred_color.png')
    save_image(pred, color_out, normalize=False)
    print('Saved', color_out)

    # Try to print some coords from models.gaussian.get_coord if available
    try:
        import models.gaussian as mg
        print('\nmodels.gaussian.get_coord signature check:')
        try:
            sample = mg.get_coord(8,6)
            print('get_coord(8,6) shape:', sample.shape)
            print('first 12 coords:', sample[:12])
        except Exception as e:
            print('get_coord call failed:', e)
    except Exception:
        print('\nmodels.gaussian module not available to query get_coord')

    print('\nDiagnosis artifacts saved to', args.outdir)


if __name__ == '__main__':
    main()

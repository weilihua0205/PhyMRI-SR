"""
验证/评估模块
在验证集上评估模型性能（PSNR, SSIM）
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils
from DISTS_pytorch import DISTS


def validate(model, val_loader, config, verbose=True, compute_ssim=True, compute_dists=False):
    """
    在验证集上评估模型
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        config: 配置字典（包含 data_norm 等信息）
        verbose: 是否显示进度条
        compute_ssim: 是否计算 SSIM（默认 True）
        compute_dists: 是否计算 DISTS（默认 False）
    
    Returns:
        如果 compute_dists=True: 返回 (平均 PSNR, 平均 SSIM, 平均 DISTS)
        如果 compute_ssim=True: 返回 (平均 PSNR, 平均 SSIM)
        如果 compute_ssim=False: 返回 平均 PSNR（向后兼容）
    """
    model.eval()

    # 初始化 DISTS（仅一次，放在 GPU 上）
    if compute_dists:
        dists_metric = DISTS().cuda()
        dists_metric.eval()
    
    # 数据归一化参数
    data_norm = config.get('data_norm', None)
    if data_norm:
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
        gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()
    
    psnr_list = []
    ssim_list = []
    dists_list = []
    
    if verbose:
        pbar = tqdm(val_loader, desc='Validating', ncols=100)
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            # 数据转移到 GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            
            # 归一化输入
            if data_norm:
                inp = (batch['inp'] - inp_sub) / inp_div
            else:
                inp = batch['inp']
            
            scale = batch['scale']
            gt = batch['gt']
            
            # 前向传播
            pred = model(inp, scale)
            
            # 反归一化预测结果
            if data_norm:
                pred = pred * gt_div + gt_sub
            
            # 裁剪到 [0, 1]
            pred = pred.clamp(0, 1)
            
            # 计算 PSNR
            psnr = calc_psnr(pred, gt, dataset=config.get('eval_type', None))
            psnr_val = psnr.item()
            
            # 异常值检测与过滤（PSNR）
            if psnr_val > 60.0 or psnr_val < 0.0 or not np.isfinite(psnr_val):
                # 记录警告但仍保留该值（避免丢失信息）
                if verbose:
                    print(f"\nWarning: Abnormal PSNR={psnr_val:.2f} detected (pred range: [{pred.min():.4f}, {pred.max():.4f}], gt range: [{gt.min():.4f}, {gt.max():.4f}])")
            
            psnr_list.append(psnr_val)
            
            # 计算 SSIM
            if compute_ssim:
                ssim = calc_ssim(pred, gt, dataset=config.get('eval_type', None))
                ssim_val = ssim.item()
                
                # 异常值检测（SSIM）
                if ssim_val > 1.0 or ssim_val < 0.0 or not np.isfinite(ssim_val):
                    if verbose:
                        print(f"\nWarning: Abnormal SSIM={ssim_val:.4f} detected (should be in [0,1])")
                
                ssim_list.append(ssim_val)
            
            # 计算 DISTS
            if compute_dists:
                dists_val = calc_dists(pred, gt, dists_metric)
                if not np.isfinite(dists_val):
                    if verbose:
                        print(f"\nWarning: Abnormal DISTS={dists_val:.4f} detected")
                dists_list.append(dists_val)
            
            if verbose:
                postfix = {'PSNR': f'{np.mean(psnr_list):.4f}'}
                if compute_ssim:
                    postfix['SSIM'] = f'{np.mean(ssim_list):.4f}'
                if compute_dists:
                    postfix['DISTS'] = f'{np.mean(dists_list):.4f}'
                pbar.set_postfix(postfix)
    
    model.train()
    
    # 计算最终平均值时，可选过滤极端异常值
    # 使用稳健统计：过滤超出 [Q1-3*IQR, Q3+3*IQR] 的离群点
    def robust_mean(values, filter_outliers=True):
        """计算稳健平均值，可选过滤极端离群点"""
        if not values or len(values) == 0:
            return 0.0
        
        arr = np.array(values)
        
        if not filter_outliers or len(arr) < 5:
            # 样本太少或不过滤，直接返回均值
            return np.mean(arr)
        
        # 使用四分位距（IQR）检测离群点
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        
        if iqr < 1e-6:
            # IQR 太小（数据几乎一致），直接返回均值
            return np.mean(arr)
        
        # 定义离群点阈值（3倍IQR）
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        
        # 过滤离群点
        filtered = arr[(arr >= lower) & (arr <= upper)]
        
        if len(filtered) == 0:
            # 所有点都被过滤（不太可能），返回中位数
            return np.median(arr)
        
        # 如果过滤掉的点超过20%，打印警告
        if len(filtered) < 0.8 * len(arr) and verbose:
            print(f"\nNote: {len(arr) - len(filtered)}/{len(arr)} outliers filtered in metric calculation")
        
        return np.mean(filtered)
    
    # 对于PSNR，过滤>60dB的极端异常值
    psnr_filtered = [p for p in psnr_list if p <= 60.0]
    if len(psnr_filtered) < len(psnr_list) and verbose:
        print(f"\nFiltered {len(psnr_list) - len(psnr_filtered)} PSNR outliers (>60 dB)")
    
    final_psnr = np.mean(psnr_filtered) if psnr_filtered else np.mean(psnr_list)
    
    if compute_dists:
        final_ssim = robust_mean(ssim_list, filter_outliers=True) if compute_ssim else None
        final_dists = robust_mean(dists_list, filter_outliers=False)
        return final_psnr, final_ssim, final_dists
    elif compute_ssim:
        # 使用稳健统计计算SSIM（过滤数值异常）
        final_ssim = robust_mean(ssim_list, filter_outliers=True)
        return final_psnr, final_ssim
    else:
        return final_psnr


def calc_dists(sr, hr, dists_metric):
    """
    计算 DISTS (Deep Image Structure and Texture Similarity)
    
    Args:
        sr: 超分辨率图像 [B, C, H, W]，值域 [0, 1]
        hr: 高分辨率图像 [B, C, H, W]，值域 [0, 1]
        dists_metric: 已初始化并移至 GPU 的 DISTS() 对象
    
    Returns:
        DISTS 值（float，越小越好，范围约 [0, 1]）
    
    Note:
        DISTS 内部使用 VGG16 特征，要求输入为 3 通道。
        单通道（MRI）图像会自动复制为 3 通道后计算。
    """
    # 单通道 → 3 通道（repeat 不额外分配大内存，共享底层存储）
    if sr.size(1) == 1:
        sr_in = sr.repeat(1, 3, 1, 1)
        hr_in = hr.repeat(1, 3, 1, 1)
    else:
        sr_in = sr
        hr_in = hr
    
    # 确保在 [0, 1] 范围内
    sr_in = sr_in.clamp(0, 1)
    hr_in = hr_in.clamp(0, 1)
    
    with torch.no_grad():
        dists_val = dists_metric(sr_in, hr_in)
    
    # dists_metric 返回 [B] 或标量，取均值
    if dists_val.dim() > 0:
        dists_val = dists_val.mean()
    
    return dists_val.item()


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    """
    计算 PSNR（支持单通道 MRI 和多通道 RGB）
    
    Args:
        sr: 超分辨率图像 [B, C, H, W]，C=1（MRI）或 C=3（RGB）
        hr: 高分辨率图像 [B, C, H, W]，C=1（MRI）或 C=3（RGB）
        dataset: 数据集类型 ('benchmark' 或 'div2k')
        scale: 缩放因子
        rgb_range: RGB 值范围（对 MRI 也适用，表示数值范围）
    
    Returns:
        PSNR 值（单位：dB）
    
    Note:
        - 单通道（C=1）：直接计算 PSNR
        - 多通道（C=3）：先转为灰度再计算（benchmark 模式）
    """
    diff = (sr - hr) / rgb_range
    
    if dataset is not None:
        if dataset == 'benchmark' or dataset.startswith('benchmark'):
            # Benchmark 数据集：需要裁剪边界并转换为灰度（仅多通道）
            if dataset.startswith('benchmark-'):
                scale = int(dataset.split('-')[1])
            
            shave = scale
            if diff.size(1) > 1:
                # 多通道（RGB）：转换为灰度（使用标准系数）
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1, keepdim=True)
            # 单通道（MRI）：直接使用，无需转换
        
        elif dataset == 'div2k' or dataset.startswith('div2k'):
            # DIV2K 数据集：裁剪更多边界
            if dataset.startswith('div2k-'):
                scale = int(dataset.split('-')[1])
            
            shave = scale + 6
        else:
            shave = 0
        
        if shave > 0:
            valid = diff[..., shave:-shave, shave:-shave]
        else:
            valid = diff
    else:
        valid = diff
    
    # 计算 MSE 和 PSNR
    mse = valid.pow(2).mean()
    
    # 避免除零和数值溢出
    # 当 MSE 极小时（如 < 1e-10），裁剪到合理上限（100 dB）
    if mse < 1e-10:
        return torch.tensor(100.0).to(sr.device)
    
    psnr = -10 * torch.log10(mse)
    
    # 裁剪 PSNR 到合理范围 [0, 100] dB
    # 避免数值异常导致的极端值
    psnr = torch.clamp(psnr, 0.0, 100.0)
    
    return psnr


def calc_ssim(sr, hr, dataset=None, scale=1, window_size=11, sigma=1.5):
    """
    计算 SSIM (Structural Similarity Index)（支持单通道 MRI 和多通道 RGB）
    
    Args:
        sr: 超分辨率图像 [B, C, H, W]，C=1（MRI）或 C=3（RGB）
        hr: 高分辨率图像 [B, C, H, W]，C=1（MRI）或 C=3（RGB）
        dataset: 数据集类型 ('benchmark' 或 'div2k')，用于边界裁剪
        scale: 缩放因子
        window_size: 高斯窗口大小（默认 11）
        sigma: 高斯窗口标准差（默认 1.5）
    
    Returns:
        SSIM 值（范围 [0, 1]，1 表示完全相同）
    
    Note:
        - 单通道（C=1）：直接计算 SSIM
        - 多通道（C=3）：先转为灰度再计算（benchmark 模式）
    """
    # 处理边界裁剪（与 PSNR 相同的逻辑）
    if dataset is not None:
        if dataset == 'benchmark' or dataset.startswith('benchmark'):
            if dataset.startswith('benchmark-'):
                scale = int(dataset.split('-')[1])
            shave = scale
            
            # 如果是多通道（RGB），转换为灰度
            if sr.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                sr = sr.mul(convert).sum(dim=1, keepdim=True)
                hr = hr.mul(convert).sum(dim=1, keepdim=True)
            # 单通道（MRI）：直接使用，无需转换
        
        elif dataset == 'div2k' or dataset.startswith('div2k'):
            if dataset.startswith('div2k-'):
                scale = int(dataset.split('-')[1])
            shave = scale + 6
        else:
            shave = 0
        
        if shave > 0:
            sr = sr[..., shave:-shave, shave:-shave]
            hr = hr[..., shave:-shave, shave:-shave]
    
    # 确保在 [0, 1] 范围
    sr = sr.clamp(0, 1)
    hr = hr.clamp(0, 1)
    
    # 创建高斯窗口
    channel = sr.size(1)
    window = _create_window(window_size, channel, sigma).to(sr.device)
    
    # 计算 SSIM
    mu1 = F.conv2d(sr, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(hr, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(sr * sr, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM 常数（基于 data_range=1.0）
    C1 = (0.01) ** 2  # (K1 * L)^2, K1=0.01, L=1
    C2 = (0.03) ** 2  # (K2 * L)^2, K2=0.03, L=1
    
    # 裁剪方差到非负值（避免数值精度导致负方差）
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
    
    # SSIM 计算（标准公式）
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-12)
    
    # 裁剪 SSIM 到理论范围 [0, 1]
    ssim_value = ssim_map.mean()
    ssim_value = torch.clamp(ssim_value, 0.0, 1.0)
    
    return ssim_value


def _create_window(window_size, channel, sigma):
    """
    创建高斯窗口用于 SSIM 计算
    
    Args:
        window_size: 窗口大小
        channel: 通道数
        sigma: 高斯标准差
    
    Returns:
        高斯窗口 tensor [channel, 1, window_size, window_size]
    """
    # 创建一维高斯核
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    # 不需要对一维核归一化，因为二维窗口会统一归一化
    
    # 创建二维高斯窗口
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)  # 外积
    window_2d = window_2d / window_2d.sum()  # 归一化到和为1
    window = window_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    
    return window


def validate_multiscale(model, dataset_configs, config, scales=[2, 3, 4]):
    """
    在多个尺度上验证模型
    
    Args:
        model: 待评估的模型
        dataset_configs: 数据集配置字典列表
        config: 全局配置
        scales: 要测试的缩放因子列表
    
    Returns:
        每个尺度的 PSNR 字典
    """
    import datasets
    from torch.utils.data import DataLoader
    
    results = {}
    
    for scale in scales:
        print(f'\n==> Evaluating scale x{scale}...')
        
        # 构建该尺度的验证集
        val_spec = dataset_configs.copy()
        val_spec['wrapper']['args']['scale_min'] = float(scale)
        val_spec['wrapper']['args']['scale_max'] = float(scale)
        
        val_dataset = datasets.make(val_spec['dataset'])
        val_dataset = datasets.make(val_spec['wrapper'], args={'dataset': val_dataset})
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 评估（只取 PSNR）
        psnr, ssim = validate(model, val_loader, config, verbose=True, compute_ssim=True)
        results[f'x{scale}'] = psnr
        print(f'Scale x{scale} PSNR: {psnr:.4f} dB')
    
    return results


def validate_on_benchmark(model, benchmark_paths, config, scale=4):
    """
    在标准 benchmark 数据集上验证
    
    Args:
        model: 待评估的模型
        benchmark_paths: benchmark 数据集路径字典
                        例如: {'Set5': '/path/to/Set5', 'Set14': '/path/to/Set14'}
        config: 配置字典
        scale: 缩放因子
    
    Returns:
        每个 benchmark 的 PSNR 字典
    """
    import datasets
    from torch.utils.data import DataLoader
    
    results = {}
    
    for benchmark_name, path in benchmark_paths.items():
        print(f'\n==> Evaluating {benchmark_name}...')
        
        # 构建数据集
        dataset_spec = {
            'dataset': {
                'name': 'image-folder',
                'args': {
                    'root_path': path,
                    'cache': 'in_memory'
                }
            },
            'wrapper': {
                'name': 'sr-implicit-downsampled',
                'args': {
                    'scale_min': float(scale),
                    'scale_max': float(scale)
                }
            }
        }
        
        dataset = datasets.make(dataset_spec['dataset'])
        dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': dataset})
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 更新 config 的 eval_type
        eval_config = config.copy()
        eval_config['eval_type'] = f'benchmark-{scale}'
        
        # 评估（只取 PSNR，benchmark 测试通常只关注 PSNR）
        psnr = validate(model, loader, eval_config, verbose=True, compute_ssim=False)
        results[benchmark_name] = psnr
        print(f'{benchmark_name} PSNR: {psnr:.4f} dB')
    
    return results


def save_validation_images(model, val_loader, save_dir, config, num_images=5):
    """
    保存验证图像用于可视化
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        save_dir: 保存目录
        config: 配置字典
        num_images: 保存图像的数量
    """
    import os
    from torchvision.utils import save_image
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # 数据归一化参数
    data_norm = config.get('data_norm', None)
    if data_norm:
        inp_sub = torch.FloatTensor(data_norm['inp']['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(data_norm['inp']['div']).view(1, -1, 1, 1).cuda()
        gt_sub = torch.FloatTensor(data_norm['gt']['sub']).view(1, -1, 1, 1).cuda()
        gt_div = torch.FloatTensor(data_norm['gt']['div']).view(1, -1, 1, 1).cuda()
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if saved_count >= num_images:
                break
            
            # 数据转移到 GPU
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            
            # 为了可视化，保留原始低分辨率输入（batch['inp']），
            # 用于上采样到 HR 大小后保存；同时构建传入模型的归一化输入 inp
            lr_orig = batch['inp']

            if data_norm:
                inp = (batch['inp'] - inp_sub) / inp_div
            else:
                inp = batch['inp']

            scale = batch['scale']
            gt = batch['gt']

            # 前向传播
            pred = model(inp, scale)

            # 反归一化预测结果（如果需要）
            if data_norm:
                pred = pred * gt_div + gt_sub

            pred = pred.clamp(0, 1)

            # 上采样 LR 到 HR 大小以便对比显示
            lr_upsampled = torch.nn.functional.interpolate(
                lr_orig, size=gt.shape[-2:], mode='bicubic', align_corners=False
            ).clamp(0, 1)

            # 如果为单通道，复制为 3 通道以便保存显示一致
            def _to_3ch(x):
                if x.size(1) == 1:
                    return x.repeat(1, 3, 1, 1)
                return x

            save_image(_to_3ch(lr_upsampled), os.path.join(save_dir, f'{batch_idx:03d}_lr.png'))
            save_image(_to_3ch(pred), os.path.join(save_dir, f'{batch_idx:03d}_sr.png'))
            save_image(_to_3ch(gt), os.path.join(save_dir, f'{batch_idx:03d}_hr.png'))

            saved_count += 1
    
    model.train()
    print(f'Saved {saved_count} validation images to {save_dir}')

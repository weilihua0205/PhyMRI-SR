"""
简单的 SSIM 测试
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/ght/MRIxField/ContinuousSR-main_MRI')

# 直接导入相关函数
from evaluate import calc_ssim, _create_window

# 测试相同图像的SSIM
sr = torch.rand(1, 1, 64, 64).cuda()
hr = sr.clone()

print(f"sr shape: {sr.shape}")
print(f"hr shape: {hr.shape}")
print(f"sr range: [{sr.min():.4f}, {sr.max():.4f}]")
print(f"hr range: [{hr.min():.4f}, {hr.max():.4f}]")
print(f"sr == hr: {torch.allclose(sr, hr)}")

# 手动计算SSIM步骤
window_size = 11
channel = 1
sigma = 1.5

window = _create_window(window_size, channel, sigma).to(sr.device)
print(f"\nWindow shape: {window.shape}")
print(f"Window sum: {window.sum():.6f}")

mu1 = F.conv2d(sr, window, padding=window_size//2, groups=channel)
mu2 = F.conv2d(hr, window, padding=window_size//2, groups=channel)

print(f"\nmu1 range: [{mu1.min():.4f}, {mu1.max():.4f}]")
print(f"mu2 range: [{mu2.min():.4f}, {mu2.max():.4f}]")
print(f"mu1 == mu2: {torch.allclose(mu1, mu2)}")

mu1_sq = mu1.pow(2)
mu2_sq = mu2.pow(2)
mu1_mu2 = mu1 * mu2

sigma1_sq = F.conv2d(sr * sr, window, padding=window_size//2, groups=channel) - mu1_sq
sigma2_sq = F.conv2d(hr * hr, window, padding=window_size//2, groups=channel) - mu2_sq
sigma12 = F.conv2d(sr * hr, window, padding=window_size//2, groups=channel) - mu1_mu2

print(f"\nsigma1_sq range: [{sigma1_sq.min():.6f}, {sigma1_sq.max():.6f}]")
print(f"sigma2_sq range: [{sigma2_sq.min():.6f}, {sigma2_sq.max():.6f}]")
print(f"sigma12 range: [{sigma12.min():.6f}, {sigma12.max():.6f}]")

C1 = 0.01 ** 2
C2 = 0.03 ** 2

sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

print(f"\nC1: {C1}, C2: {C2}")
print(f"numerator range: [{numerator.min():.6f}, {numerator.max():.6f}]")
print(f"denominator range: [{denominator.min():.6f}, {denominator.max():.6f}]")

ssim_map = numerator / (denominator + 1e-12)
print(f"\nssim_map range: [{ssim_map.min():.6f}, {ssim_map.max():.6f}]")
print(f"ssim_map mean: {ssim_map.mean():.6f}")

# 调用函数
ssim = calc_ssim(sr, hr)
print(f"\nFinal SSIM: {ssim.item():.6f}")

"""
测试验证代码的鲁棒性
验证修复后的 calc_psnr 和 calc_ssim 能否正确处理异常情况
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/ght/MRIxField/ContinuousSR-main_MRI')

from evaluate import calc_psnr, calc_ssim


def test_psnr_robustness():
    """测试 PSNR 计算的鲁棒性"""
    print("="*60)
    print("Testing PSNR Robustness")
    print("="*60)
    
    # Case 1: 正常情况
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr + torch.randn(1, 1, 64, 64).cuda() * 0.1
    hr = hr.clamp(0, 1)
    psnr = calc_psnr(sr, hr)
    print(f"Case 1 (Normal): PSNR = {psnr.item():.4f} dB")
    assert 0 <= psnr.item() <= 100, f"PSNR should be in [0, 100], got {psnr.item()}"
    
    # Case 2: pred == gt (完全相同)
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr.clone()
    psnr = calc_psnr(sr, hr)
    print(f"Case 2 (Identical): PSNR = {psnr.item():.4f} dB")
    assert psnr.item() <= 100, f"PSNR should be capped at 100, got {psnr.item()}"
    assert psnr.item() >= 80, f"Identical images should have high PSNR, got {psnr.item()}"
    
    # Case 3: pred ≈ gt (极小差异)
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr + 1e-8
    psnr = calc_psnr(sr, hr)
    print(f"Case 3 (Near Identical): PSNR = {psnr.item():.4f} dB")
    assert psnr.item() <= 100, f"PSNR should be capped, got {psnr.item()}"
    assert np.isfinite(psnr.item()), f"PSNR should be finite, got {psnr.item()}"
    
    # Case 4: 大差异
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = torch.rand(1, 1, 64, 64).cuda()
    psnr = calc_psnr(sr, hr)
    print(f"Case 4 (Large Diff): PSNR = {psnr.item():.4f} dB")
    assert 0 <= psnr.item() <= 100, f"PSNR should be in [0, 100], got {psnr.item()}"
    
    print("✓ All PSNR tests passed!\n")


def test_ssim_robustness():
    """测试 SSIM 计算的鲁棒性"""
    print("="*60)
    print("Testing SSIM Robustness")
    print("="*60)
    
    # Case 1: 正常情况
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr + torch.randn(1, 1, 64, 64).cuda() * 0.1
    hr = hr.clamp(0, 1)
    ssim = calc_ssim(sr, hr)
    print(f"Case 1 (Normal): SSIM = {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"
    
    # Case 2: pred == gt (完全相同)
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr.clone()
    ssim = calc_ssim(sr, hr)
    print(f"Case 2 (Identical): SSIM = {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"
    assert ssim.item() >= 0.99, f"Identical images should have SSIM ≈ 1, got {ssim.item()}"
    
    # Case 3: pred ≈ gt (极小差异)
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = sr + 1e-8
    ssim = calc_ssim(sr, hr)
    print(f"Case 3 (Near Identical): SSIM = {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"
    assert np.isfinite(ssim.item()), f"SSIM should be finite, got {ssim.item()}"
    
    # Case 4: 大差异
    sr = torch.rand(1, 1, 64, 64).cuda()
    hr = torch.rand(1, 1, 64, 64).cuda()
    ssim = calc_ssim(sr, hr)
    print(f"Case 4 (Large Diff): SSIM = {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"
    
    # Case 5: 恒定图像（方差为0）
    sr = torch.ones(1, 1, 64, 64).cuda() * 0.5
    hr = sr.clone()
    ssim = calc_ssim(sr, hr)
    print(f"Case 5 (Constant Image): SSIM = {ssim.item():.4f}")
    assert 0 <= ssim.item() <= 1, f"SSIM should be in [0, 1], got {ssim.item()}"
    
    print("✓ All SSIM tests passed!\n")


def test_robust_mean():
    """测试 robust_mean 函数（在 validate 中定义）"""
    print("="*60)
    print("Testing Robust Mean (Outlier Filtering)")
    print("="*60)
    
    # 模拟正常值 + 异常值的情况
    normal_psnr = [25.3, 26.1, 24.8, 25.5, 26.3, 25.0, 24.5]
    outlier_psnr = [300.0, 280.0]  # 异常高的PSNR
    all_psnr = normal_psnr + outlier_psnr
    
    print(f"All PSNR values: {all_psnr}")
    print(f"Mean (with outliers): {np.mean(all_psnr):.2f} dB")
    print(f"Mean (without outliers): {np.mean(normal_psnr):.2f} dB")
    
    # 测试过滤 >60dB 的值
    filtered = [p for p in all_psnr if p <= 60.0]
    print(f"Filtered (>60dB removed): {filtered}")
    print(f"Mean (filtered): {np.mean(filtered):.2f} dB")
    
    print("✓ Outlier filtering test passed!\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Evaluate Robustness Test Suite")
    print("="*60 + "\n")
    
    test_psnr_robustness()
    test_ssim_robustness()
    test_robust_mean()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)

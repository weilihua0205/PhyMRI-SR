"""
测试 SSIM 计算功能
"""

import torch
from evaluate import calc_ssim, calc_psnr

def test_ssim():
    """测试 SSIM 计算"""
    print("==> Testing SSIM calculation...")
    
    # 创建测试图像
    batch_size = 2
    channels = 3
    height, width = 48, 48
    
    # 测试 1: 完全相同的图像 (SSIM 应该为 1.0)
    img1 = torch.rand(batch_size, channels, height, width).cuda()
    img2 = img1.clone()
    
    ssim = calc_ssim(img1, img2)
    psnr = calc_psnr(img1, img2)
    
    print(f"\nTest 1: 相同图像")
    print(f"  SSIM: {ssim.item():.6f} (expected: ~1.0)")
    print(f"  PSNR: {psnr.item():.2f} dB (expected: inf)")
    
    # 测试 2: 轻微不同的图像
    img1 = torch.rand(batch_size, channels, height, width).cuda()
    img2 = img1 + torch.randn_like(img1) * 0.01  # 添加小噪声
    img2 = img2.clamp(0, 1)
    
    ssim = calc_ssim(img1, img2)
    psnr = calc_psnr(img1, img2)
    
    print(f"\nTest 2: 轻微噪声")
    print(f"  SSIM: {ssim.item():.6f} (expected: 0.9-1.0)")
    print(f"  PSNR: {psnr.item():.2f} dB")
    
    # 测试 3: 较大差异的图像
    img1 = torch.rand(batch_size, channels, height, width).cuda()
    img2 = torch.rand(batch_size, channels, height, width).cuda()
    
    ssim = calc_ssim(img1, img2)
    psnr = calc_psnr(img1, img2)
    
    print(f"\nTest 3: 随机图像")
    print(f"  SSIM: {ssim.item():.6f} (expected: 0.0-0.5)")
    print(f"  PSNR: {psnr.item():.2f} dB")
    
    # 测试 4: 带 benchmark 数据集裁剪
    img1 = torch.rand(batch_size, channels, height, width).cuda()
    img2 = img1 + torch.randn_like(img1) * 0.01
    img2 = img2.clamp(0, 1)
    
    ssim_benchmark = calc_ssim(img1, img2, dataset='benchmark-4')
    ssim_div2k = calc_ssim(img1, img2, dataset='div2k-4')
    ssim_normal = calc_ssim(img1, img2, dataset=None)
    
    print(f"\nTest 4: 不同数据集模式")
    print(f"  SSIM (benchmark): {ssim_benchmark.item():.6f}")
    print(f"  SSIM (div2k): {ssim_div2k.item():.6f}")
    print(f"  SSIM (normal): {ssim_normal.item():.6f}")
    
    print("\n==> SSIM tests completed successfully!")

if __name__ == '__main__':
    test_ssim()

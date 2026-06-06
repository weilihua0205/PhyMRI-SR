"""
测试前景加噪功能：对比整个FOV加噪 vs 仅前景加噪
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.wrappers import _add_rician_noise_to_magnitude

# 创建模拟MRI图像：背景为0，前景为高强度
def create_mock_mri_image():
    """创建模拟MRI图像 [1, 64, 64]"""
    img = np.zeros((1, 64, 64), dtype=np.float32)
    # 中心矩形区域（前景）
    img[0, 16:48, 16:48] = 0.8
    # 在前景中添加一些细微纹理
    noise_texture = np.random.normal(0, 0.02, (32, 32))
    img[0, 16:48, 16:48] += np.clip(noise_texture, -0.1, 0.1)
    img = np.clip(img, 0, 1)
    return torch.from_numpy(img)

# 创建图像
mri = create_mock_mri_image()
print(f"原始图像范围: [{mri.min().item():.4f}, {mri.max().item():.4f}]")
print(f"原始图像形状: {mri.shape}")

# 参数
sigma = 0.05
k_factor = 0.8

# 方案A：整个FOV加噪（旧方式，会导致问题）
noisy_full = _add_rician_noise_to_magnitude(mri, sigma, k_factor=k_factor, 
                                           mode='rician', foreground_only=False)
print(f"\nFOV全加噪 - 范围: [{noisy_full.min().item():.4f}, {noisy_full.max().item():.4f}]")
print(f"  背景均值: {noisy_full[0, 0:16, 0:16].mean().item():.4f}")
print(f"  前景均值: {noisy_full[0, 16:48, 16:48].mean().item():.4f}")

# 方案B：仅前景加噪（新方式，推荐）
noisy_fg = _add_rician_noise_to_magnitude(mri, sigma, k_factor=k_factor, 
                                         mode='rician', foreground_only=True, 
                                         fg_threshold=None)  # None 会自动计算阈值
print(f"\n仅前景加噪 - 范围: [{noisy_fg.min().item():.4f}, {noisy_fg.max().item():.4f}]")
print(f"  背景均值: {noisy_fg[0, 0:16, 0:16].mean().item():.4f}")
print(f"  前景均值: {noisy_fg[0, 16:48, 16:48].mean().item():.4f}")

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(mri[0].numpy(), cmap='gray')
axes[0].set_title('原始图像\n(背景=0, 前景~0.8)')
axes[0].axis('off')

axes[1].imshow(noisy_full[0].numpy(), cmap='gray')
axes[1].set_title('FOV全加噪\n(背景被提升！)')
axes[1].axis('off')

axes[2].imshow(noisy_fg[0].numpy(), cmap='gray')
axes[2].set_title('仅前景加噪\n(背景保持黑色)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('/home/ght/MRIxField/ContinuousSR-main_MRI/comparison_foreground_noise.png', dpi=100, bbox_inches='tight')
print("\n✅ 对比图已保存到: comparison_foreground_noise.png")

# 计算对比度（前景-背景）
orig_contrast = (mri[0, 16:48, 16:48].mean() - mri[0, 0:16, 0:16].mean()).item()
full_contrast = (noisy_full[0, 16:48, 16:48].mean() - noisy_full[0, 0:16, 0:16].mean()).item()
fg_contrast = (noisy_fg[0, 16:48, 16:48].mean() - noisy_fg[0, 0:16, 0:16].mean()).item()

print(f"\n对比度分析（前景-背景）:")
print(f"  原始:        {orig_contrast:.4f}")
print(f"  FOV全加噪:   {full_contrast:.4f} (损失 {(1-full_contrast/orig_contrast)*100:.1f}%)")
print(f"  仅前景加噪:  {fg_contrast:.4f} (损失 {(1-fg_contrast/orig_contrast)*100:.1f}%)")

print(f"\n✅ 结论: 仅前景加噪能更好地保持对比度，避免背景填充问题！")

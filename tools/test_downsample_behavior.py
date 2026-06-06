"""
Test SRImplicitDownsampled downsampling behavior with a synthetic image.
Saves comparison images and prints PSNR/SSIM between upsampled LR and HR.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.wrappers import SRImplicitDownsampled

# Create a simple synthetic dataset that returns a single HR image
class DummyHRDataset:
    def __init__(self, img):
        self.img = img
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        # return HR image tensor [C,H,W]
        return self.img

# make a synthetic textured image [1, 128, 128]
H = 128
W = 128
img = np.zeros((1, H, W), dtype=np.float32)
# center square with texture
img[0, 32:96, 32:96] = 0.8
img[0, 40:88, 40:88] += (np.random.randn(48,48) * 0.02).clip(-0.05,0.05)
img = np.clip(img, 0, 1)
img_t = torch.from_numpy(img)

# instantiate dataset wrapper with inp_size meaning HR target size (64), scale=2
base_ds = DummyHRDataset(img_t)
wrapper = SRImplicitDownsampled(base_ds, inp_size=64, scale_min=2.0, scale_max=2.0, add_noise=False)

item = wrapper[0]
inp = item['inp']  # LR patch
gt = item['gt']    # HR patch
s = item['scale']

print('Shapes:')
print('  inp:', inp.shape)
print('  gt :', gt.shape)
print('  scale:', s)
print('Ranges:')
print('  inp min/max/mean:', inp.min().item(), inp.max().item(), inp.mean().item())
print('  gt  min/max/mean:', gt.min().item(), gt.max().item(), gt.mean().item())

# Upsample inp to GT size for comparison
inp_up = torch.nn.functional.interpolate(inp.unsqueeze(0), size=gt.shape[-2:], mode='bicubic', align_corners=False).squeeze(0).clamp(0,1)

# compute simple MSE/PSNR and SSIM via evaluate.calc_ssim and utils.calc_psnr if available
try:
    from utils import calc_psnr
    from evaluate import calc_ssim
    lr_psnr = calc_psnr(inp_up.unsqueeze(0), gt.unsqueeze(0), dataset=None, scale=1, rgb_range=1)
    lr_ssim = calc_ssim(inp_up.unsqueeze(0), gt.unsqueeze(0), dataset=None, scale=1)
    print(f'Computed metrics: LR->GT PSNR={lr_psnr:.4f}, SSIM={lr_ssim:.4f}')
except Exception as e:
    print('Could not compute PSNR/SSIM:', e)

# Save comparison figure
os.makedirs('debug_outputs', exist_ok=True)
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(inp[0].cpu().numpy(), cmap='gray')
axes[0].set_title('LR (downsampled)')
axes[0].axis('off')

axes[1].imshow(inp_up[0].cpu().numpy(), cmap='gray')
axes[1].set_title('LR upsampled to HR')
axes[1].axis('off')

axes[2].imshow(gt[0].cpu().numpy(), cmap='gray')
axes[2].set_title('GT (HR)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_outputs/downsample_debug.png', dpi=150)
print('Saved debug image to debug_outputs/downsample_debug.png')

# Also save numpy arrays
np.save('debug_outputs/inp.npy', inp.cpu().numpy())
np.save('debug_outputs/inp_up.npy', inp_up.cpu().numpy())
np.save('debug_outputs/gt.npy', gt.cpu().numpy())
print('Saved numpy arrays in debug_outputs/')

"""
Quick test for SRImplicitDownsampled noise integration.
Creates a dummy HR dataset of shape [1, H, W], instantiates wrapper with noise,
and prints statistics for inp (LR) with and without noise.
"""
import torch
import numpy as np
from datasets.wrappers import SRImplicitDownsampled

class DummyHR:
    def __init__(self, H=128, W=128):
        self.H = H
        self.W = W
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        # return a single-channel float image CHW
        img = np.random.rand(1, self.H, self.W).astype(np.float32)
        return img

if __name__ == '__main__':
    base = DummyHR(128, 128)
    wrapper_noisy = SRImplicitDownsampled(base, inp_size=32, scale_min=2, scale_max=2, add_noise=True, noise_sigma=0.05, noise_k_factor=0.8)
    item = wrapper_noisy[0]
    inp = item['inp']
    gt = item['gt']
    print('inp type:', type(inp), 'shape:', inp.shape)
    if isinstance(inp, torch.Tensor):
        print('inp stats:', inp.min().item(), inp.max().item(), inp.mean().item())
    else:
        print('inp stats (np):', inp.min(), inp.max(), inp.mean())
    print('gt type:', type(gt), 'shape:', gt.shape)

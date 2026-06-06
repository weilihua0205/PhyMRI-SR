"""
Encoder-based Super-Resolution Wrapper Models

This module provides standalone super-resolution models using encoder architectures
(EDSR, SwinIR, RDN, HAT) without the ContinuousGaussian framework.

These models serve as baseline methods for comparison with ContinuousGaussian.
"""

import torch
import torch.nn as nn
from models import register
from argparse import Namespace


class EncoderSRWrapper(nn.Module):
    """
    Generic wrapper to convert encoder models into standalone SR models.
    
    The wrapper:
    1. Extracts features from the encoder
    2. Passes through optional refinement layers
    3. Applies upsampling
    """
    
    def __init__(self, encoder, upsampler=None, upscale=4, num_refine_layers=2):
        """
        Args:
            encoder: Base encoder model (EDSR, SwinIR, etc.)
            upsampler: Optional upsampler module (default: PixelShuffle-based)
            upscale: Upsampling scale factor
            num_refine_layers: Number of refinement conv layers before upsampling
        """
        super(EncoderSRWrapper, self).__init__()
        self.encoder = encoder
        self.upscale = upscale
        
        # Get output channel dimension from encoder
        # Most encoders output 256 or 64 channels
        self.out_channels = getattr(encoder, 'out_dim', 64)
        
        # Refinement layers (optional)
        refine_layers = []
        for i in range(num_refine_layers):
            refine_layers.append(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
            if i < num_refine_layers - 1:
                refine_layers.append(nn.ReLU(inplace=True))
        self.refine = nn.Sequential(*refine_layers) if refine_layers else None
        
        # Upsampler
        if upsampler is not None:
            self.upsampler = upsampler
        else:
            self.upsampler = self._build_default_upsampler()
    
    def _build_default_upsampler(self):
        """Build default PixelShuffle-based upsampler"""
        import math
        
        modules = []
        
        # PixelShuffle upsampling
        if (self.upscale & (self.upscale - 1)) == 0:  # upscale = 2^n
            for _ in range(int(math.log(self.upscale, 2))):
                modules.append(
                    nn.Conv2d(self.out_channels, 4 * self.out_channels, 3, padding=1)
                )
                modules.append(nn.PixelShuffle(2))
        elif self.upscale == 3:
            modules.append(
                nn.Conv2d(self.out_channels, 9 * self.out_channels, 3, padding=1)
            )
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Unsupported upscale factor: {self.upscale}')
        
        # Final conv to output 1 channel (for grayscale MRI)
        modules.append(nn.Conv2d(self.out_channels, 1, 3, padding=1))
        
        return nn.Sequential(*modules)
    
    def forward(self, x, scale=None, **kwargs):
        """
        Args:
            x: Input tensor [B, 1, H, W]
            scale: Scale factor (not used, for compatibility)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            Super-resolved output [B, 1, H*upscale, W*upscale]
        """
        # Extract features
        feat = self.encoder(x)
        
        # Handle different encoder output formats
        if isinstance(feat, tuple):
            feat = feat[0]  # Some encoders return (features, extra_info)
        
        # Refinement
        if self.refine is not None:
            feat = self.refine(feat)
        
        # Upsample
        out = self.upsampler(feat)
        
        return out


@register('edsr-sr')
def make_edsr_sr(scale=4, num_refine_layers=2):
    """
    EDSR as standalone super-resolution model.
    
    Args:
        scale: Upsampling factor (2, 3, or 4)
        num_refine_layers: Number of refinement layers
    
    Returns:
        Wrapped EDSR model
    """
    from models import make as models_make
    
    # Create base EDSR encoder
    encoder_spec = {
        'name': 'edsr-baseline',
        'args': {
            'n_resblocks': 32,
            'n_feats': 256,
            'res_scale': 0.1,
            'scale': scale,
            'no_upsampling': True,  # Important: disable upsampling in encoder
            'rgb_range': 1,
            'n_colors': 1  # Grayscale
        }
    }
    encoder = models_make(encoder_spec)
    
    # Wrap with upsampler
    return EncoderSRWrapper(encoder, upscale=scale, num_refine_layers=num_refine_layers)


@register('swinir-sr')
def make_swinir_sr(scale=4, num_refine_layers=2):
    """
    SwinIR as standalone super-resolution model.
    
    Args:
        scale: Upsampling factor (2, 3, or 4)
        num_refine_layers: Number of refinement layers
    
    Returns:
        Wrapped SwinIR model
    """
    from models import make as models_make
    
    # Create base SwinIR encoder
    encoder_spec = {
        'name': 'swinir',
        'args': {
            'img_size': 64,
            'patch_size': 1,
            'in_chans': 1,  # Grayscale
            'embed_dim': 180,
            'depths': [6, 6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6, 6],
            'window_size': 8,
            'mlp_ratio': 2.0,
            'drop_path_rate': 0.1,
            'upscale': scale,
            'upsampler': 'none',  # Important: disable upsampling in encoder
            'resi_connection': '1conv'
        }
    }
    encoder = models_make(encoder_spec)
    
    # Wrap with upsampler
    return EncoderSRWrapper(encoder, upscale=scale, num_refine_layers=num_refine_layers)


@register('rdn-sr')
def make_rdn_sr(scale=4, num_refine_layers=2):
    """
    RDN (Residual Dense Network) as standalone super-resolution model.
    
    Args:
        scale: Upsampling factor (2, 3, or 4)
        num_refine_layers: Number of refinement layers
    
    Returns:
        Wrapped RDN model
    """
    from models import make as models_make
    
    # Create base RDN encoder
    args = Namespace()
    args.n_resgroups = 10
    args.n_resblocks = 20
    args.n_colors = 1  # Grayscale
    args.rgb_range = 1
    args.n_feats = 64
    args.res_scale = 0.2
    args.scale = [scale]
    args.no_upsampling = True  # Important: disable upsampling in encoder
    
    encoder_spec = {
        'name': 'rdn',
        'args': vars(args)
    }
    encoder = models_make(encoder_spec)
    
    # Wrap with upsampler
    return EncoderSRWrapper(encoder, upscale=scale, num_refine_layers=num_refine_layers)


@register('hat-sr')
def make_hat_sr(scale=4, num_refine_layers=2):
    """
    HAT (Hybrid Attention Transformer) as standalone super-resolution model.
    
    Args:
        scale: Upsampling factor (2, 3, or 4)
        num_refine_layers: Number of refinement layers
    
    Returns:
        Wrapped HAT model
    """
    from models import make as models_make
    
    # Create base HAT encoder
    encoder_spec = {
        'name': 'hat',
        'args': {
            'num_in_ch': 1,  # Grayscale
            'num_out_ch': 1,
            'num_feat': 64,
            'num_block': 16,
            'num_grow_ch': 32,
            'upscale': scale,
            'do_upsampler': False,  # Important: disable upsampling in encoder
        }
    }
    encoder = models_make(encoder_spec)
    
    # Wrap with upsampler
    return EncoderSRWrapper(encoder, upscale=scale, num_refine_layers=num_refine_layers)


if __name__ == '__main__':
    """Test the wrapper models"""
    import sys
    sys.path.insert(0, '/home/ght/MRIxField/ContinuousSR-main_MRI_seg')
    
    # Test EDSR-SR
    print("Testing EDSR-SR...")
    model_edsr = make_edsr_sr(scale=4)
    x = torch.randn(1, 1, 64, 64)
    y_edsr = model_edsr(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_edsr.shape}")
    print(f"  Expected: [1, 1, 256, 256]")
    
    # Test SwinIR-SR
    print("\nTesting SwinIR-SR...")
    model_swinir = make_swinir_sr(scale=4)
    y_swinir = model_swinir(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y_swinir.shape}")
    print(f"  Expected: [1, 1, 256, 256]")

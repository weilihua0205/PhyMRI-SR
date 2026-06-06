import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import to_pixel_samples
import time
from torchvision.utils import save_image
from itertools import product

import gsplat
from gsplat import project_gaussians_2d, rasterize_gaussians_sum
import os
import numpy as np
import matplotlib.pyplot as plt


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape): #每一维度默认范围为（-1，1）
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) #flatten之后的返回（N，ndim）的扁平坐标矩阵
    return ret


def generate_meshgrid(height, width): #为图像内的每一个像素生成坐标
    """
    Generate a meshgrid of coordinates for a given image dimensions.
    Args:
        height (int): Height of the image.
        width (int): Width of the image.
    Returns:
        torch.Tensor: A tensor of shape [height * width, 2] containing the (x, y) coordinates for each pixel in the image.
    """
    # Generate all pixel coordinates for the given image dimensions
    y_coords, x_coords = torch.arange(0, height), torch.arange(0, width)
    # Create a grid of coordinates
    yy, xx = torch.meshgrid(y_coords, x_coords)
    # Flatten and stack the coordinates to obtain a list of (x, y) pairs
    all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return all_coords


def fetching_features_from_tensor(image_tensor, input_coords):
    """
    Extracts pixel values from a tensor of images at specified coordinate locations.
    Args:
        image_tensor (torch.Tensor): A 4D tensor of shape [batch, channel, height, width] representing a batch of images.
        input_coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the (x, y) coordinates at which to extract pixel values.
    Returns:
        color_values (torch.Tensor): A 3D tensor of shape [batch, N, channel] containing the pixel values at the specified coordinates.
        coords (torch.Tensor): A 2D tensor of shape [N, 2] containing the normalized coordinates in the range [-1, 1].
    """
    # Normalize pixel coordinates to [-1, 1] range
    input_coords = input_coords.to(image_tensor.device)
    coords = input_coords / torch.tensor([image_tensor.shape[-2], image_tensor.shape[-1]],
                                         device=image_tensor.device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=image_tensor.device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    batch_size = image_tensor.shape[0]
    input_coords_expanded = input_coords.unsqueeze(0).expand(batch_size, -1, -1)

    y_coords = input_coords_expanded[..., 0].long()
    x_coords = input_coords_expanded[..., 1].long()
    batch_indices = torch.arange(batch_size).view(-1, 1).to(input_coords.device)

    color_values = image_tensor[batch_indices, :, x_coords, y_coords]

    return color_values, coords


def scale_to_range(tensor, min_value, max_value):
    min_tensor = torch.min(tensor)
    max_tensor = torch.max(tensor)
    scaled_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)  
    return scaled_tensor * (max_value - min_value) + min_value


def get_coord(width, height):
    x_coords = torch.arange(width)
    y_coords = torch.arange(height)

    # Generate coordinate grid using torch.meshgrid
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # Map coordinates to the range of -1 to 1
    x_grid = 2 * (x_grid / (width)) - 1 #+ 1/width
    y_grid = 2 * (y_grid / (height)) - 1 #+ 1/height

    # Stack the x and y coordinates to form the final coordinate tensor
    coordinates = torch.stack((y_grid, x_grid), dim=-1).reshape(-1, 2)
    
    return coordinates



@register('continuous-gaussian')
class ContinuousGaussian(nn.Module):
    """A module that applies 2D Gaussian splatting to input features."""

    def __init__(self, encoder_spec, cnn_spec, fc_spec, **kwargs):
        """
        Initialize the ContinuousGaussian module.
        
        Args:
            encoder_spec (dict): Specifications for the encoder.
            cnn_spec (dict): Specifications for the CNN layers.
            fc_spec (dict): Specifications for the fully connected layers.
            kwargs: Additional arguments.
        """
        super(ContinuousGaussian, self).__init__()

        #self.k_density = 4 # Changing this from 2 to 4 increases kernels by 4x (Total 16 per pixel)
        
        # Create the encoder module based on the given specifications
        self.encoder = models.make(encoder_spec)
        
        # Initialize placeholders for various attributes
        self.feat = None  # Low-resolution (LR) features
        self.inp = None  # Input image
        self.feat_coord = None  # Feature coordinates
        self.init_num_points = None
        self.H, self.W = None, None  # Image height and width
        self.BLOCK_H, self.BLOCK_W = 16, 16  # Block size for tiling

        # NEW: Kernel Density Settings
        # Original code used 4 kernels per pixel (2x2 grid). 
        # To double it, we use 16 kernels (4x4 grid).
        self.k_h = 4
        self.k_w = 4
        self.k_total = self.k_h * self.k_w # 16 kernels per pixel

        # Define additional convolutional and activation layers
        # After expansion we expect 512 channels, so conv1 input channels set to 512
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # Convolutional layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation
        self.ps = nn.PixelUnshuffle(2)  # Pixel unshuffle with a scaling factor of 2

        # NEW: Expansion layer to provide features for twice as many kernels
        # Based on your architecture, we double the channel capacity after pixel unshuffle
        self.feat_expand = nn.Conv2d(256, 512, kernel_size=1)

        # Define an MLP for vector generation for gaussian dict
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        # Define an MLP for color prediction: for grayscale images
        # Input dim = 128 (512 channels / 4 channel-groups per spatial position for 16 kernels)
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 128, 'out_dim': 1, 'hidden_list': [256, 512, 256, 128, 64]}}
        self.mlp = models.make(mlp_spec)
        
        # Define an MLP for offset prediction
        # Input dim = 128 (512 channels / 4 channel-groups per spatial position for 16 kernels)
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 128, 'out_dim': 2, 'hidden_list': [256, 512, 256, 128, 64]}}
        self.mlp_offset = models.make(mlp_spec)
        
        # Initialize pre-defined Gaussian convariance parameter dictionaries
        # Dictionary 1: from natural images (stable, smooth results)
        cho1_natural = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]).cuda()
        cho2_natural = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]).cuda()
        cho3_natural = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]).cuda()

        # Dictionary 2: from MRI data (textured results, may be unstable)
        cho1_mri = torch.tensor([0.56, 0.77, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()
        cho2_mri = torch.tensor([-0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()
        cho3_mri = torch.tensor([0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05, 2.26]).cuda()
        
        # Selective merge - combine key components
        # Take small variance from natural images (cho1/cho3 with 0) and large variance from MRI
        cho1_hybrid = torch.tensor([0, 0.41, 0.62, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()  # Extended
        cho2_hybrid = torch.tensor([-0.86, -0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()  # Wider range
        cho3_hybrid = torch.tensor([0, 0.33, 0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05]).cuda()  # Extended
        self.gau_dict = torch.tensor(list(product(cho1_hybrid, cho2_hybrid, cho3_hybrid))).cuda()
        
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1, 3).cuda()), dim=0)  # Add zeros
        print(f"Gaussian dictionary initialized with {self.gau_dict.shape[0]} entries (hybrid strategy)")

        self.last_size = (self.H, self.W)
        self.background = torch.ones(1).cuda()  # Default background color for 1-ch output

    def gen_feat(self, inp):
        """
        Generate features from the input image using the encoder.
        
        Args:
            inp (torch.Tensor): Input image.
        
        Returns:
            torch.Tensor: Extracted low-resolution features.
        """
        self.inp = inp  # Store the input image
        feat = self.encoder(inp)  # Encode the input image
        # Apply pixel unshuffle to the encoded features (increases channels by 4)
        self.feat = self.ps(feat)
        # Apply expansion to support the increased number of kernels
        # Replace self.feat with the expanded features so downstream modules
        # operate on the expanded (512-channel) representation.
        self.feat = self.feat_expand(self.feat)
        return self.feat

    def query_output(self, inp, scale):
        """
        Generate the high-resolution image output for a given scale.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        feat = self.feat

        # Process the scaling factors
        if scale.shape == (1, 2):  # Handle cases with two scaling factors
            scale1 = float(scale[0, 0])
            scale2 = float(scale[0, 1])
        else:  # Handle uniform scaling
            scale1 = float(scale[0])
            scale2 = float(scale[0])

        # Compute dimensions of the low-resolution and high-resolution images
        lr_h = self.inp.shape[-2]  # Low-resolution height
        lr_w = self.inp.shape[-1]  # Low-resolution width
        H = round(int(lr_h) * scale1)  # High-resolution height
        W = round(int(lr_w) * scale2)  # High-resolution width

        # Determine the number of tiles for rasterization
        self.tile_bounds = (
            (W + self.BLOCK_W - 1) // self.BLOCK_W,
            (H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )

        window_size = 1  # Window size for Gaussian position adjustments
        pred = []  # List to store predictions
        bs, _, feat_h, feat_w = feat.shape  # Batch size and feature dimensions (feat_h, feat_w are after unshuffle)

        # Process features for color prediction
        # feat has 512 channels after expansion, shape [bs, 512, feat_h, feat_w]
        # Note: feat_h = lr_h * 2, feat_w = lr_w * 2 due to PixelUnshuffle(2)
        # Each LR pixel corresponds to a 2x2 block in feat space, and we want k_total=16 kernels per LR pixel
        # Since k_h=4, k_w=4, we have 16 kernels arranged as 4x4 grid per LR pixel
        # The 2x2 spatial block in feat gives us 4 spatial positions, each needs 4 kernels (to get 16 total)
        
        # Reshape to group by LR pixels: [bs, 512, lr_h, 2, lr_w, 2] -> [bs, 512, lr_h, lr_w, 4]
        para_c = self.feat.reshape(bs, 512, lr_h, 2, lr_w, 2).permute(0, 1, 2, 4, 3, 5).reshape(bs, 512, lr_h, lr_w, 4)
        # Now reshape to [bs, lr_h, lr_w, 4, 512] then split 512 channels into per-kernel features
        para_c = para_c.permute(0, 2, 3, 4, 1).reshape(bs, lr_h * lr_w * 4, 512)  # [bs, lr_h*lr_w*4, 512]
        # We need k_total=16 with 4 spatial positions per LR pixel
        # Split 512 channels into 4 groups of 128 to get 16 kernels: 4 positions * 4 channel-groups = 16 kernels
        para_c = para_c.reshape(bs, lr_h * lr_w * 4, 4, 128).reshape(bs, lr_h * lr_w * self.k_total, 128)  # [bs, lr_h*lr_w*16, 128]
        
        color = self.mlp(para_c.reshape(-1, 128))  # [bs*lr_h*lr_w*16, 128] -> [bs*lr_h*lr_w*16, 1]
        color = color.reshape(bs, lr_h * lr_w * self.k_total, -1)

        # Process features for Gaussian convariance parameter estimation
        para_c_conv = self.leaky_relu(self.feat)
        para = self.conv1(para_c_conv)  # [bs, 512, feat_h, feat_w]
        vector = self.mlp_vector(self.gau_dict.to(para.device))  # [1001, 512]
        
        # Reshape para to [512, bs*lr_h*lr_w] for per-LR-pixel dictionary matching
        # Note: para has spatial size feat_h x feat_w = (lr_h*2) x (lr_w*2)
        # We need to pool/aggregate to lr_h x lr_w resolution
        para = para.reshape(bs, 512, lr_h, 2, lr_w, 2).mean(dim=(3, 5))  # Average over 2x2 blocks -> [bs, 512, lr_h, lr_w]
        para = para.reshape(bs, 512, lr_h * lr_w).reshape(512, bs * lr_h * lr_w)  # [512, bs*lr_h*lr_w]
        para = vector @ para  # [1001, 512] @ [512, bs*lr_h*lr_w] = [1001, bs*lr_h*lr_w]
        
        # CRITICAL: Add temperature scaling to prevent overconfident selection
        temperature = 2.0  # Adjust this: higher = more diverse selection (try 2.0-5.0 if unstable)
        para = para / temperature
        
        # Monitor softmax sharpness BEFORE normalization
        para_max = para.max(dim=0)[0].mean()
        para_std = para.std(dim=0).mean()
        
        para = torch.softmax(para, dim=0)  # Normalize the similarity scores to produce weights using softmax
        
        # Check if selection is too concentrated (potential instability indicator)
        max_weight = para.max(dim=0)[0].mean()
        if max_weight > 0.95:  # If any position has >95% weight on one dict entry
            print(f"WARNING: Softmax over-concentrated! Max weight: {max_weight:.4f}, consider increasing temperature")
        
        para = para.permute(1, 0) @ self.gau_dict.to(para.device)  # [bs*lr_h*lr_w, 1001] @ [1001, 3] = [bs*lr_h*lr_w, 3]
        # Expand to k_total kernels: replicate for each kernel at same spatial location
        para = para.reshape(bs, lr_h * lr_w, 3).unsqueeze(2).expand(bs, lr_h * lr_w, self.k_total, 3)  # [bs, lr_h*lr_w, 16, 3]
        para = para.reshape(bs, lr_h * lr_w * self.k_total, 3)  # [bs, lr_h*lr_w*16, 3]

        # Process features for offset prediction
        # Reuse para_c which has shape [bs, lr_h*lr_w*16, 128]
        offset = self.mlp_offset(para_c.reshape(-1, 128))  # [bs*lr_h*lr_w*16, 128] -> [bs*lr_h*lr_w*16, 2]
        offset = torch.tanh(offset).reshape(bs, lr_h * lr_w * self.k_total, -1)
        
        # STABILITY: Ensure para has minimum values to prevent singular covariance matrices
        # This prevents cho1=0 AND cho3=0 simultaneously (which creates degenerate gaussians)
        # Use fully out-of-place operations: rebuild tensor with torch.stack to avoid any inplace modification
        eps = 1e-6
        cho1_clamped = torch.clamp(para[:, :, 0], min=eps)  # cho1 >= eps
        cho2_unchanged = para[:, :, 1]  # cho2 keeps original
        cho3_clamped = torch.clamp(para[:, :, 2], min=eps)  # cho3 >= eps
        para = torch.stack([cho1_clamped, cho2_unchanged, cho3_clamped], dim=2)

        # Generate output predictions for each image in the batch
        for i in range(bs):
            offset_ = offset[i, :, :].squeeze(0)
            color_ = color[i, :, :].squeeze(0)
            para_ = para[i, :, :].squeeze(0)

            # Generate coordinate grid for the high-resolution image
            get_xyz = torch.tensor(get_coord(lr_h * self.k_h, lr_w * self.k_w)).reshape(lr_h * self.k_h, lr_w * self.k_w, 2).cuda()
            get_xyz = get_xyz.reshape(-1, 2)

            # # Adjust coordinates using offsets
            # xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / wf - 1 / W
            # xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / hf - 1 / H

            # Adjust coordinates using offsets
            xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / lr_w - 1 / W
            xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / lr_h - 1 / H
            get_xyz = torch.cat((xyz1, xyz2), dim=1)

            # Adjust Gaussian parameters
            weighted_cholesky = para_ / 4
            weighted_opacity = torch.ones(color_.shape[0], 1).cuda()
            
            # Apply scale factors and stability constraints using fully out-of-place operations
            # Rebuild tensor with torch.stack to avoid any inplace modification
            min_cholesky = 0.001  # Minimum standard deviation in normalized coordinates
            
            cho1_scaled = torch.clamp(weighted_cholesky[:, 0] * scale2, min=min_cholesky)
            cho2_scaled = weighted_cholesky[:, 1] * scale1
            cho3_scaled = torch.clamp(weighted_cholesky[:, 2] * scale1, min=min_cholesky)
            
            weighted_cholesky = torch.stack([cho1_scaled, cho2_scaled, cho3_scaled], dim=1)

            # Perform Gaussian projection and rasterization
            xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
                get_xyz, weighted_cholesky, H, W, self.tile_bounds
            )
            
            # SAFETY CHECK: If too many gaussians are culled, this indicates a problem
            valid_ratio = (radii > 0).float().mean()
            if valid_ratio < 0.1:  # Less than 10% valid gaussians
                print(f"CRITICAL WARNING: Only {valid_ratio*100:.1f}% gaussians valid! Potential collapse imminent.")
                print(f"  weighted_cholesky stats: min={weighted_cholesky.min():.6f}, max={weighted_cholesky.max():.6f}")
            
            out_img = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit, color_, weighted_opacity,
                H, W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False
            )
            out_img = out_img.permute(2, 0, 1).unsqueeze(0)
            pred.append(out_img)

        # Combine outputs for the batch
        out_img = torch.cat(pred)
        return out_img

    def forward(self, inp, scale):
        """
        Forward pass for the ContinuousGaussian module.
        
        Args:
            inp (torch.Tensor): Input image.
            scale (torch.Tensor): Scaling factor.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        self.gen_feat(inp)  # Generate low-resolution features
        image = self.query_output(inp, scale)  # Generate high-resolution output
        return image

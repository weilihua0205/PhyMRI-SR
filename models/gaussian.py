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


def mask_guided_sample_coords(mask, grid_h, grid_w, density_weights=None):
    """
    根据分割 mask 生成非均匀采样坐标，前景高密度区域分配更多坐标点。

    Args:
        mask (torch.Tensor): [1, H, W] 或 [H, W]，整数标签（0=背景,1=CSF,2=GM,3=WM）
        grid_h (int): 目标格点高度
        grid_w (int): 目标格点宽度
        density_weights (dict, optional): {标签值(int): 权重(float)}
                        默认: {0: 0.5, 1: 2.7, 2: 1.8, 3: 1.0}

    Returns:
        sample_coords (torch.Tensor): [N, 2]，归一化到 [-1, 1] 的 (x, y) 坐标，
                                      N = grid_h * grid_w，与均匀格点总数相同。
                                      格式与 get_coord() 输出一致（先 y/col，后 x/row）。
    """
    if density_weights is None:
        # density_weights = {0: 0.5, 1: 2.7, 2: 1.8, 3: 1.0}
        density_weights = {0: 0.5, 1: 1.0, 2: 2.7, 3: 1.8}
        #density_weights = {0: 0.5, 1: 0.99, 2: 0.85, 3: 0.70}

    device = mask.device
    if mask.dim() == 3:
        mask_2d = mask[0]   # [H, W]
    else:
        mask_2d = mask       # [H, W]

    mask_np = mask_2d.cpu().numpy().astype(np.int32)
    H_mask, W_mask = mask_np.shape
    N_total = grid_h * grid_w

    labels = np.unique(mask_np)
    # 计算各标签加权像素数
    weighted_counts = {}
    for lbl in labels:
        w = density_weights.get(int(lbl), 0.0)
        weighted_counts[lbl] = float(np.sum(mask_np == lbl)) * w

    total_weighted = sum(weighted_counts.values())

    all_row_coords = []  # 像素行坐标
    all_col_coords = []  # 像素列坐标

    if total_weighted <= 0:
        # 所有权重为 0（全背景），退化为均匀格点
        coords = get_coord(grid_w, grid_h).to(device)
        return coords

    # 按比例分配采样数量到各标签区域
    allocated = 0
    label_allocations = {}
    for lbl in labels:
        if weighted_counts[lbl] <= 0:
            label_allocations[lbl] = 0
            continue
        n_k = int(round(N_total * weighted_counts[lbl] / total_weighted))
        label_allocations[lbl] = n_k
        allocated += n_k

    # 修正舍入误差，保证总数恰好为 N_total
    diff = N_total - allocated
    if diff != 0:
        # 找到权重最大的前景标签进行补偿
        fg_labels = [l for l in labels if label_allocations[l] > 0]
        if fg_labels:
            top_lbl = max(fg_labels, key=lambda l: weighted_counts[l])
            label_allocations[top_lbl] += diff

    # 在各标签区域内有放回地随机采样像素坐标
    rng = np.random.default_rng()
    for lbl in labels:
        n_k = label_allocations.get(lbl, 0)
        if n_k <= 0:
            continue
        pixel_indices = np.argwhere(mask_np == lbl)  # [M, 2]: (row, col)
        if len(pixel_indices) == 0:
            continue
        chosen = rng.choice(len(pixel_indices), size=n_k, replace=True)
        chosen_coords = pixel_indices[chosen]   # [n_k, 2]
        all_row_coords.append(chosen_coords[:, 0])
        all_col_coords.append(chosen_coords[:, 1])

    if len(all_row_coords) == 0:
        coords = get_coord(grid_w, grid_h).to(device)
        return coords

    rows = np.concatenate(all_row_coords)   # 像素行（对应 y / height 轴）
    cols = np.concatenate(all_col_coords)   # 像素列（对应 x / width  轴）

    # 将像素坐标映射到 mask 分辨率的 [-1, 1]，再重新映射到 grid 分辨率的 [-1, 1]
    # mask 空间 → [-1, 1]（中心对齐）
    y_norm = (rows + 0.5) / H_mask * 2.0 - 1.0   # height 轴
    x_norm = (cols + 0.5) / W_mask * 2.0 - 1.0   # width  轴

    # 对齐到 grid 分辨率的格点中心（轻微量化，保证与 grid_sample 一致）
    y_snapped = np.round((y_norm + 1.0) / 2.0 * grid_h - 0.5).clip(0, grid_h - 1)
    x_snapped = np.round((x_norm + 1.0) / 2.0 * grid_w - 0.5).clip(0, grid_w - 1)
    y_out = (y_snapped + 0.5) / grid_h * 2.0 - 1.0
    x_out = (x_snapped + 0.5) / grid_w * 2.0 - 1.0

    # 输出格式与 get_coord() 一致：每行 [x(col方向), y(row方向)]
    sample_coords = torch.from_numpy(
        np.stack([x_out, y_out], axis=1).astype(np.float32)
    ).to(device)   # [N, 2]

    return sample_coords



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
        
        # Create the encoder module based on the given specifications
        self.encoder = models.make(encoder_spec)
        
        # Initialize placeholders for various attributes
        self.feat = None  # Low-resolution (LR) features
        self.inp = None  # Input image
        self.feat_coord = None  # Feature coordinates
        self.init_num_points = None
        self.H, self.W = None, None  # Image height and width
        self.BLOCK_H, self.BLOCK_W = 16, 16  # Block size for tiling

        # Define additional convolutional and activation layers
        # After feature expansion we operate on 512 channels
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # Convolutional layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation
        self.ps = nn.PixelUnshuffle(2)  # Pixel unshuffle with a scaling factor of 2

        # Increase per-pixel kernel density to 16 (4x4 grid per LR pixel)
        self.k_h = 4
        self.k_w = 4
        self.k_total = self.k_h * self.k_w  # 16 kernels per LR pixel

        # Expand features from 256 -> 512 to allow splitting into per-kernel groups
        self.feat_expand = nn.Conv2d(256, 512, kernel_size=1)

        # Define an MLP for vector generation for gaussian dict
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        # ============================================================
        # T2 Physics-informed intensity prediction (v2)
        # Signal = ρ * exp(-1/R2) + δ
        #   ρ: proton density (via softplus, ≥0)
        #   R2: T2/TE ratio (via softplus + eps, >0)
        #   δ: learned residual correction (unbounded, full-capacity MLP)
        # 
        # Design: physics term provides structural prior (tissue contrast),
        #         residual term compensates value range and fine details.
        #         No hard clamping on δ — let gradient flow freely.
        # ============================================================
        # MLP for physics parameters: predicts raw_ρ and raw_R2 (2 outputs)
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 128, 'out_dim': 2, 'hidden_list': [256, 128, 64]}}
        self.mlp_physics = models.make(mlp_spec)
        
        # Full-capacity MLP for residual correction δ (same as original color MLP)
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 128, 'out_dim': 1, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp_residual = models.make(mlp_spec)
        
        # Softplus for non-negative activation
        self.softplus = nn.Softplus()
        
        # Define an MLP for offset prediction (per-kernel)
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 128, 'out_dim': 2, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp_offset = models.make(mlp_spec)
        
        # Initialize pre-defined Gaussian convariance parameter dictionaries
        # Dictionary 1: from natural images (stable, smooth results)
        cho1_natural = torch.tensor([0, 0.41, 0.62, 0.98, 1.13, 1.29, 1.64, 1.85, 2.36]).cuda()
        cho2_natural = torch.tensor([-0.86, -0.36, -0.16, 0.19, 0.34, 0.49, 0.84, 1.04, 1.54]).cuda()
        cho3_natural = torch.tensor([0, 0.33, 0.53, 0.88, 1.03, 1.18, 1.53, 1.73, 2.23]).cuda()

        # Dictionary 2: from 5T MRI data (textured results, may be unstable)
        cho1_mri = torch.tensor([0.56, 0.77, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()
        cho2_mri = torch.tensor([-0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()
        cho3_mri = torch.tensor([0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05, 2.26]).cuda()

        # dictionary:5T (gau quantile)
        cho1_mri_5t = torch.tensor([0.42, 0.71, 0.86, 1.24, 1.40, 1.57, 1.95, 2.10, 2.39])
        cho2_mri_5t = torch.tensor([-0.54, -0.29, -0.17, 0.15, 0.29, 0.43, 0.75, 0.88, 1.12])
        cho3_mri_5t = torch.tensor([0.40, 0.69, 0.85, 1.23, 1.40, 1.57, 1.95, 2.11, 2.40])

        #Dictionary 3: from 3T MRI data
        cho1_3t = torch.tensor([0.29, 0.49, 0.61, 0.88, 1.00, 1.12, 1.39, 1.50, 1.71]).cuda()
        cho2_3t = torch.tensor([-0.18, 0.03, 0.14, 0.41, 0.52, 0.64, 0.91, 1.02, 1.23]).cuda()
        cho3_3t = torch.tensor([0.26, 0.47, 0.58, 0.86, 0.98, 1.10, 1.38, 1.49, 1.71]).cuda()
        
        # Selective merge - combine key components
        # Take small variance from natural images (cho1/cho3 with 0) and large variance from MRI
        cho1_hybrid = torch.tensor([0, 0.41, 0.62, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()  # Extended
        cho2_hybrid = torch.tensor([-0.86, -0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()  # Wider range
        cho3_hybrid = torch.tensor([0, 0.33, 0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05]).cuda()  # Extended

        #hybird 3t+5t
        cho1_hybrid_mri = torch.tensor([0, 0.42, 0.71, 0.86, 1.24, 1.40, 1.57, 1.95, 2.10, 2.39]).cuda()  # Extended
        cho2_hybrid_mri = torch.tensor([-0.18, -0.54, -0.29, -0.17, 0.15, 0.29, 0.43, 0.75, 0.88, 1.12]).cuda()  # Wider range
        cho3_hybrid_mri = torch.tensor([0, 0.26, 0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05]).cuda()  # Extended

        cho1_hybrid_mri_small = torch.tensor([0, 0.05, 0.13, 0.42, 0.71, 0.86, 1.24, 1.40, 1.57, 1.95]).cuda()  # Extended
        cho2_hybrid_mri_small = torch.tensor([-0.18, -0.54, -0.29, -0.17, 0.15, 0.29, 0.43, 0.75, 0.88, 1.12]).cuda()  # Wider range
        cho3_hybrid_mri_small = torch.tensor([0, 0.05, 0.14, 0.26, 0.54, 0.75, 0.97, 1.19, 1.40, 1.62]).cuda()  # Extended

        # 非均匀字典：在小 scale 区间（[0, 0.5]）密集采样（7个点），大 scale 区间（[0.5, 1.95]）稀疏采样（3个点）
        # 目的：让网络有更多小 scale 高斯核可选，同时保留大 scale 的梯度通道，避免表达能力退化
        # cho1 和 cho3 独立设计：cho1 控制 x 方向宽度，cho3 控制 y 方向宽度
        # cho1_hybrid_mri_small = torch.tensor([0, 0.05, 0.10, 0.18, 0.28, 0.40, 0.50, 0.86, 1.40, 1.95]).cuda()  # 前7个点在[0,0.5]加密
        # cho2_hybrid_mri_small = torch.tensor([-0.18, -0.54, -0.29, -0.17, 0.15, 0.29, 0.43, 0.75, 0.88, 1.12]).cuda()  # Wider range
        # cho3_hybrid_mri_small = torch.tensor([0, 0.05, 0.10, 0.18, 0.28, 0.40, 0.50, 0.75, 1.19, 1.62]).cuda()  # 前7个点在[0,0.5]加密
        
        # #hybird seg 3t+5t
        # cho1_seg = torch.tensor([0, 0.13, 0.15, 0.16, 0.18, 0.19, 0.20, 0.23, 0.24, 0.26], dtype=torch.float32).cuda()
        # cho2_seg = torch.tensor([-0.19, -0.18, -0.10, 0.15, 0.38, 0.60, 0.82, 1.31, 1.51, 1.89], dtype=torch.float32).cuda()
        # cho3_seg = torch.tensor([0, 0.14, 0.15, 0.16, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23], dtype=torch.float32).cuda()

        # cho1_seg = torch.tensor([0, 0.13, 0.15, 0.16, 0.18, 0.19, 0.20, 0.42, 0.71, 0.86], dtype=torch.float32).cuda()
        # cho2_seg = torch.tensor([-0.19, -0.18, -0.10, 0.15, 0.38, 0.60, 0.82, 1.31, 1.51, 1.89], dtype=torch.float32).cuda()
        # cho3_seg = torch.tensor([0, 0.14, 0.15, 0.16, 0.18, 0.19, 0.20, 0.54, 0.75, 0.97], dtype=torch.float32).cuda()

        self.gau_dict = torch.tensor(list(product(cho1_hybrid_mri_small.cpu(), cho2_hybrid_mri_small.cpu(), cho3_hybrid_mri_small.cpu())), dtype=torch.float32).cuda()
        
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1, 3, dtype=torch.float32).cuda()), dim=0)  # Add zeros
        print(f"Gaussian dictionary initialized with {self.gau_dict.shape[0]} entries (hybrid strategy)")

        self.last_size = (self.H, self.W)
        self.background = torch.ones(1).cuda()  # Default background color for 1-ch output

        # mask 引导的非均匀采样坐标（路径一）：在 gen_feat() 时生成，query_output() 时使用
        # 默认为 None，表示退化为均匀格点
        self.sample_coords = None  # [N, 2] or None

        # 各组织标签的密度权重（可通过配置覆盖）
        # 0=背景(低密度), 1=CSF(最高), 2=GM(中), 3=WM(低)
        self.density_weights = kwargs.get('density_weights', {0: 0.5, 1: 2.7, 2: 1.8, 3: 1.0})

    def gen_feat(self, inp, mask=None):
        """
        Generate features from the input image using the encoder.
        If mask is provided, also generate mask-guided non-uniform sample_coords
        for use in query_output().
        
        Args:
            inp  (torch.Tensor): Input image [bs, 1, H, W].
            mask (torch.Tensor | None): Segmentation mask [bs, 1, H_hr, W_hr],
                                        integer labels as float32. None = uniform grid.
        
        Returns:
            torch.Tensor: Extracted low-resolution features.
        """
        self.inp = inp  # Store the input image
        feat = self.encoder(inp)  # Encode the input image
        # Apply pixel unshuffle to the encoded features (increases channels by 4)
        self.feat = self.ps(feat)
        # Apply expansion to support the increased number of kernels
        # Replace self.feat with the expanded (512-channel) representation.
        self.feat = self.feat_expand(self.feat)

        # 生成非均匀采样坐标（每张图独立，存为列表供 query_output 按 batch 索引）
        lr_h = inp.shape[-2]
        lr_w = inp.shape[-1]
        grid_h = lr_h * self.k_h
        grid_w = lr_w * self.k_w

        if mask is not None:
            # 为 batch 中每张图生成各自的 sample_coords
            self.sample_coords = []
            for b in range(inp.shape[0]):
                coords_b = mask_guided_sample_coords(
                    mask[b],          # [1, H_hr, W_hr]
                    grid_h, grid_w,
                    density_weights=self.density_weights
                )  # [N, 2]
                self.sample_coords.append(coords_b)
        else:
            # 无 mask：退化为均匀格点（与原始行为完全一致）
            self.sample_coords = None

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
        # Note: feat_h = lr_h/2, feat_w = lr_w/2 due to PixelUnshuffle(2)
        # We want k_total=16 kernels per LR pixel (4x4 grid)
        # Total kernels = lr_h * lr_w * 16
        # feat positions = (lr_h/2) * (lr_w/2) = lr_h * lr_w / 4
        # Each feat position needs to produce 16 * 4 = 64 kernels worth of features
        # With 512 channels, each kernel gets 512 / 64 = 8 dims... too few!
        # 
        # Alternative approach: use same structure as original but with 4x more kernels
        # Original: feat [bs, 256, feat_h, feat_w] -> para_c [bs, 256, lr_h*lr_w*4] (treating each of 4 sub-pixels)
        # For 16x: we keep spatial structure same but use channel splitting
        #
        # Actually, let's follow original pattern more closely:
        # Original uses 256 channels for lr_h*lr_w*4 positions, each position gets 256-dim feature
        # For 16 kernels, we have lr_h*lr_w*16 positions with 512 channels
        # Each kernel gets 512 * (lr_h/2 * lr_w/2) / (lr_h * lr_w * 16) = 512 / 64 = 8 dims
        #
        # Better approach: Upsample feat spatially to match kernel grid
        # Kernel grid is lr_h*k_h x lr_w*k_w = lr_h*4 x lr_w*4
        # Upsample feat from [bs, 512, lr_h/2, lr_w/2] to [bs, 512, lr_h*4, lr_w*4] using interpolation
        
        feat_h, feat_w = feat_h, feat_w  # lr_h/2, lr_w/2
        target_h, target_w = lr_h * self.k_h, lr_w * self.k_w  # lr_h*4, lr_w*4
        
        # Upsample features to match kernel grid resolution
        feat_upsampled = F.interpolate(self.feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # feat_upsampled: [bs, 512, lr_h*4, lr_w*4]
        
        # ================================================================
        # 核心修改：支持 mask 引导的非均匀采样
        # ================================================================
        def sample_feat(feat_map, coords_list, target_h, target_w):
            """
            feat_map:   [bs, C, gh, gw]
            coords_list: list of [N, 2] (x, y) in [-1,1], or None for uniform
            返回: [bs, C, N] 按坐标采样的特征，N = target_h * target_w
            """
            bs_f, C, gh, gw = feat_map.shape
            N = target_h * target_w
            if coords_list is None:
                # 均匀展平（原始逻辑）
                return feat_map.reshape(bs_f, C, N)
            # 非均匀 grid_sample
            sampled_list = []
            for b in range(bs_f):
                # grid_sample 要求 grid 形状 [1, N, 1, 2]，坐标顺序 (x, y)
                grid = coords_list[b].to(feat_map.device)   # [N, 2]
                grid = grid.view(1, N, 1, 2)                # [1, N, 1, 2]
                feat_b = feat_map[b:b+1]                    # [1, C, gh, gw]
                # grid_sample 输出 [1, C, N, 1]
                sampled = F.grid_sample(feat_b, grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')
                sampled_list.append(sampled.squeeze(0).squeeze(-1))  # [C, N]
            return torch.stack(sampled_list, dim=0)  # [bs, C, N]

        # 采样颜色特征
        N = lr_h * lr_w * self.k_total
        para_c = sample_feat(feat_upsampled, self.sample_coords, target_h, target_w)  # [bs, 512, N]
        para_c = para_c.reshape(bs, 4, 128, N)  # [bs, 4, 128, N]
        para_c = para_c[:, 0, :, :]  # Take first 128-channel group: [bs, 128, N]
        para_c = para_c.permute(0, 2, 1).reshape(bs * N, 128)  # [bs*N, 128]
        
        # T2 Physics-informed intensity prediction (v2)
        # Step 1: Predict physics parameters (ρ, R2)
        physics_raw = self.mlp_physics(para_c)  # [bs*N, 128] -> [bs*N, 2]
        raw_rho = physics_raw[:, 0:1]   # [bs*N, 1]
        raw_r2  = physics_raw[:, 1:2]   # [bs*N, 1]
        
        # Step 2: Apply constraints
        rho = self.softplus(raw_rho) + 0.1     # ρ ≥ 0.1 (proton density下限):
                                               #   防止ρ→0导致物理项永久失效（梯度消失陷阱）
                                               #   softplus(x)+0.1 保证最小值0.1，exp(-1/R2)≈0.007
                                               #   物理项最小值≈0.1*0.007=0.0007，始终有非零梯度
        r2  = self.softplus(raw_r2) + 0.2      # R2 = T2/TE > 0 (minimum 0.2:
                                               #   exp(-1/0.2)=exp(-5)≈0.007，物理项仍可趋近于0
                                               #   1/R2²最大=25，梯度稳定；原0.01时1/R2²=10000导致爆炸)
        
        # Step 3: Physics signal: ρ * exp(-1/R2)
        # 额外 clamp：防止在极端 R2 下指数值的数值溢出
        exp_term = torch.exp(torch.clamp(-1.0 / r2, min=-20.0))  # exp值下限 exp(-20)≈2e-9，避免梯度消失区
        signal_physics = rho * exp_term  # [bs*N, 1]
        
        # Step 4: Residual correction δ
        # 用 soft clamp 替代 tanh：在正常范围(-3,3)内梯度接近线性（≈原始MLP输出），
        # 仅在极端值时平滑截断，避免梯度消失（tanh饱和区梯度→0会损害mlp_residual的表达能力）
        # 正常训练时|δ|≈0.19，崩溃时暴涨至1.13；3.0的上限既覆盖正常范围又防止失控
        delta_raw = self.mlp_residual(para_c)  # [bs*N, 1]
        delta = 3.0 * torch.tanh(delta_raw / 3.0)  # soft clamp: 近线性区间约(-3,3)，边界平滑截断
        
        # Step 5: Final signal = physics + residual
        color = signal_physics + delta  # [bs*N, 1]
        color = color.reshape(bs, N, -1)
        
        # 暴露 phy_ratio 供外部损失使用
        # phy_ratio = |signal_physics| / (|signal_physics| + |δ| + ε)
        self.last_phy_ratio = (signal_physics.abs() / 
                               (signal_physics.abs() + delta.abs() + 1e-8)).reshape(bs, N, 1)
        
        # Expose physics parameters for monitoring
        self.last_rho = rho.reshape(bs, N, 1)            # [bs, N, 1]
        self.last_r2  = r2.reshape(bs, N, 1)              # [bs, N, 1]
        self.last_delta = delta.reshape(bs, N, 1)          # [bs, N, 1]
        self.last_signal_physics = signal_physics.reshape(bs, N, 1)  # [bs, N, 1]

        # Process features for Gaussian convariance parameter estimation
        para_c_conv = self.leaky_relu(self.feat)
        para = self.conv1(para_c_conv)  # [bs, 512, feat_h, feat_w] where feat_h=lr_h/2, feat_w=lr_w/2
        vector = self.mlp_vector(self.gau_dict.to(para.device))  # [dict_size, 512]
        
        # Upsample para to kernel grid resolution for per-kernel covariance
        para_upsampled = F.interpolate(para, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # para_upsampled: [bs, 512, lr_h*4, lr_w*4]
        
        # 采样协方差特征（使用非均匀采样）
        para_sampled = sample_feat(para_upsampled, self.sample_coords, target_h, target_w)  # [bs, 512, N]
        para_sampled = para_sampled.permute(0, 2, 1).reshape(bs * N, 512)  # [bs*N, 512]
        
        para_sim = vector @ para_sampled.permute(1, 0)  # [dict_size, 512] @ [512, bs*N] = [dict_size, bs*N]
        
        # CRITICAL: Add temperature scaling to prevent overconfident selection
        temperature = 2.0  # Adjust this: higher = more diverse selection (try 2.0-5.0 if unstable)
        para_sim = para_sim / temperature
        
        # Monitor softmax sharpness BEFORE normalization
        para_max = para_sim.max(dim=0)[0].mean()
        para_std = para_sim.std(dim=0).mean()
        
        para_sim = torch.softmax(para_sim, dim=0)  # Normalize the similarity scores to produce weights using softmax
        
        # Check if selection is too concentrated (potential instability indicator)
        max_weight = para_sim.max(dim=0)[0].mean()
        if max_weight > 0.95:  # If any position has >95% weight on one dict entry
            print(f"WARNING: Softmax over-concentrated! Max weight: {max_weight:.4f}, consider increasing temperature")
        
        para = para_sim.permute(1, 0) @ self.gau_dict.to(para_sim.device)  # [bs*N, dict_size] @ [dict_size, 3] = [bs*N, 3]
        para = para.reshape(bs, N, 3)  # [bs, N, 3]

        # Process features for offset prediction
        # Use upsampled features for offset (second 128-channel group)
        offset_feat = feat_upsampled.reshape(bs, 4, 128, target_h, target_w)[:, 1, :, :, :]  # [bs, 128, H, W]
        offset_feat = offset_feat.view(bs, 128, target_h, target_w)  # [bs, 128, H, W]
        offset_sampled = sample_feat(offset_feat, self.sample_coords, target_h, target_w)  # [bs, 128, N]
        offset_sampled = offset_sampled.permute(0, 2, 1).reshape(bs * N, 128)  # [bs*N, 128]
        
        offset = self.mlp_offset(offset_sampled)  # [bs*N, 128] -> [bs*N, 2]
        offset = torch.tanh(offset).reshape(bs, N, -1)
        
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

            # 使用 sample_coords 作为初始格点坐标（与特征采样坐标完全对应）
            if self.sample_coords is not None:
                get_xyz = self.sample_coords[i].to(feat.device)  # [N, 2]
            else:
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

        # 暴露每个高斯核的 cho1/cho3（已 clamp、未除以4），供外部计算 scale 软约束损失
        # para shape: [bs, N, 3]，其中 para[...,0]=cho1，para[...,2]=cho3
        self.last_para = para  # [bs, N, 3]

        return out_img

    def forward(self, inp, scale, mask=None):
        """
        Forward pass for the ContinuousGaussian module.
        
        Args:
            inp  (torch.Tensor): Input image [bs, 1, H, W].
            scale (torch.Tensor): Scaling factor.
            mask (torch.Tensor | None): Segmentation mask [bs, 1, H_hr, W_hr],
                                        integer labels as float32. None = uniform grid.
        
        Returns:
            torch.Tensor: High-resolution output image.
        """
        self.gen_feat(inp, mask=mask)  # Generate low-resolution features + compute sample_coords
        image = self.query_output(inp, scale)  # Generate high-resolution output
        return image

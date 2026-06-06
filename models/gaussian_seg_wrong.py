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
import importlib

try:
    _backend_mod = importlib.import_module('gsplat.cuda._backend')
    GSPLAT_AVAILABLE = getattr(_backend_mod, '_C', None) is not None
except Exception:
    GSPLAT_AVAILABLE = False


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
    与 GaussianImage 的 _init_xyz_from_mask() 逻辑一致。

    Args:
        mask (torch.Tensor): [1, H, W] 或 [H, W]，整数标签（0=背景,1=CSF,2=GM,3=WM）
        grid_h (int): 目标格点高度（= lr_h * 2）
        grid_w (int): 目标格点宽度（= lr_w * 2）
        density_weights (dict, optional): {标签值(int): 权重(float)}
                        默认: {0: 0.0, 1: 3.0, 2: 2.0, 3: 1.0}

    Returns:
        sample_coords (torch.Tensor): [N, 2]，归一化到 [-1, 1] 的 (x, y) 坐标，
                                      N = grid_h * grid_w，与均匀格点总数相同。
                                      格式与 get_coord() 输出一致（先 y/col，后 x/row）。
    """
    if density_weights is None:
        density_weights = {0: 0.5, 1: 2.7, 2: 1.8, 3: 1.0}

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
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Convolutional layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation
        self.ps = nn.PixelUnshuffle(2)  # Pixel unshuffle with a scaling factor of 2

        # Define an MLP for vector generation for gaussian dict
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 3, 'out_dim': 512, 'hidden_list': [256, 512, 512, 512]}}
        self.mlp_vector = models.make(mlp_spec)
        
        # Define an MLP for color prediction: for grayscale images
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 1, 'hidden_list': [512, 1024, 256, 128, 64]}}
        self.mlp = models.make(mlp_spec)
        
        # Define an MLP for offset prediction
        mlp_spec = {'name': 'mlp', 'args': {'in_dim': 256, 'out_dim': 2, 'hidden_list': [512, 1024, 256, 128, 64]}}
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

        # Strategy: Hybrid Dictionary - merge selected components from both
        # Option A: Full concatenation (large dict: 729+729+1 = 1459 entries)
        # gau_dict_natural = torch.tensor(list(product(cho1_natural, cho2_natural, cho3_natural))).cuda()
        # gau_dict_mri = torch.tensor(list(product(cho1_mri, cho2_mri, cho3_mri))).cuda()
        # self.gau_dict = torch.cat([gau_dict_natural, gau_dict_mri, torch.zeros(1, 3).cuda()], dim=0)

        # Option C: Use original MRI dict only (comment out lines 179-181, uncomment below)
        # cho1 = torch.tensor([0.56, 0.77, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()
        # cho2 = torch.tensor([-0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()
        # cho3 = torch.tensor([0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05, 2.26]).cuda()
        # self.gau_dict = torch.tensor(list(product(cho1, cho2, cho3))).cuda()
        
        # Selective merge - combine key components
        # Take small variance from natural images (cho1/cho3 with 0) and large variance from MRI
        cho1_hybrid = torch.tensor([0, 0.41, 0.62, 0.98, 1.19, 1.40, 1.62, 1.83, 2.04, 2.25]).cuda()  # Extended
        cho2_hybrid = torch.tensor([-0.86, -0.42, -0.24, -0.07, 0.11, 0.29, 0.47, 0.65, 0.83, 1.01]).cuda()  # Wider range
        cho3_hybrid = torch.tensor([0, 0.33, 0.54, 0.75, 0.97, 1.19, 1.40, 1.62, 1.83, 2.05]).cuda()  # Extended
        self.gau_dict = torch.tensor(list(product(cho1_hybrid, cho2_hybrid, cho3_hybrid))).cuda()
        
        self.gau_dict = torch.cat((self.gau_dict, torch.zeros(1, 3).cuda()), dim=0)  # Add zeros
        print(f"Gaussian dictionary initialized with {self.gau_dict.shape[0]} entries (hybrid strategy)")

        self.last_size = (self.H, self.W)
        self.background = torch.zeros(1).cuda()  # MRI background is black (0)
        if not GSPLAT_AVAILABLE:
            raise RuntimeError(
                "gsplat CUDA backend is unavailable (_C is None). "
                "Please ensure CUDA Toolkit is installed and PATH contains nvcc "
                "(e.g., export CUDA_HOME=/usr/local/cuda; export PATH=$CUDA_HOME/bin:$PATH)."
            )

        # mask 引导的非均匀采样坐标（路径一）：在 gen_feat() 时生成，query_output() 时使用
        # 默认为 None，表示退化为均匀格点
        self.sample_coords = None  # [N, 2] or None

        # 各组织标签的密度权重（可通过配置覆盖）
        # 0=背景(不放核), 1=CSF(最高), 2=GM, 3=WM
        self.density_weights = kwargs.get('density_weights', {0: 0.8, 1: 2.6, 2: 1.4, 3: 1.0})

    def gen_feat(self, inp, mask=None):
        """
        Generate features from the input image using the encoder.
        If mask is provided, also generate mask-guided non-uniform sample_coords
        for use in query_output().

        Args:
            inp  (torch.Tensor): Input LR image [bs, 1, H, W].
            mask (torch.Tensor | None): Segmentation mask [bs, 1, H_hr, W_hr],
                                        integer labels as float32. None = uniform grid.
        Returns:
            torch.Tensor: Extracted low-resolution features.
        """
        self.inp = inp
        feat = self.encoder(inp)
        self.feat = self.ps(feat)  # [bs, C, lr_h*2, lr_w*2]

        # 生成非均匀采样坐标（每张图独立，存为列表供 query_output 按 batch 索引）
        lr_h = inp.shape[-2]
        lr_w = inp.shape[-1]
        grid_h = lr_h * 2
        grid_w = lr_w * 2

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
        bs, _, _, _ = feat.shape  # Batch size and feature dimensions
        N = lr_h * lr_w * 4  # 每张图的高斯核数量（= lr_h*2 * lr_w*2）

        # ----------------------------------------------------------------
        # 路径一核心：按 sample_coords 从特征图采样，替代原来的 reshape 展平
        # 有 mask 时：sample_coords[b] 是非均匀坐标，前景区域密集
        # 无 mask 时：退化为均匀格点（与原始行为完全一致）
        # ----------------------------------------------------------------
        def sample_feat(feat_map, coords_list):
            """
            feat_map:   [bs, C, gh, gw]
            coords_list: list of [N, 2] (x, y) in [-1,1], or None for uniform
            返回: [bs, C, N]，每张图按各自 coords 从特征图采样
            """
            bs_f, C, gh, gw = feat_map.shape
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
        feat_for_color = sample_feat(self.feat, self.sample_coords)          # [bs, C_ps, N]
        para_c_color = feat_for_color.permute(1, 0, 2)                       # [C_ps, bs, N]
        color = self.mlp(para_c_color.reshape(-1, bs * N).permute(1, 0))     # [bs*N, 1]
        color = color.reshape(bs, N, -1)                                     # [bs, N, 1]

        # 采样协方差 / offset 特征
        para_c_leaky = self.leaky_relu(self.feat)                            # [bs, 256, gh, gw]
        feat_for_para = sample_feat(para_c_leaky, self.sample_coords)        # [bs, 256, N]
        para_conv_input = para_c_leaky                                        # conv1 仍在空间域操作
        para = self.conv1(para_conv_input)                                    # [bs, 512, gh, gw]
        feat_para_sampled = sample_feat(para, self.sample_coords)             # [bs, 512, N]

        # 字典查询（Cholesky 参数估计）
        vector = self.mlp_vector(self.gau_dict.to(feat.device))              # [D, 512]
        feat_para_flat = feat_para_sampled.permute(1, 0, 2).reshape(-1, bs * N)  # [512, bs*N]
        para_sim = vector @ feat_para_flat                                    # [D, bs*N]

        # CRITICAL: temperature scaling
        temperature = 2.0
        para_sim = para_sim / temperature
        para_sim = torch.softmax(para_sim, dim=0)                            # [D, bs*N]

        # 监控 softmax 集中度
        max_weight = para_sim.max(dim=0)[0].mean()
        if max_weight > 0.95:
            print(f"WARNING: Softmax over-concentrated! Max weight: {max_weight:.4f}")

        para = para_sim.permute(1, 0) @ self.gau_dict.to(feat.device)        # [bs*N, 3]
        para = para.reshape(bs, N, -1)                                       # [bs, N, 3]

        # offset 预测
        feat_offset_flat = feat_for_para.permute(1, 0, 2).reshape(-1, bs * N)  # [256, bs*N]
        offset = self.mlp_offset(feat_offset_flat.permute(1, 0))              # [bs*N, 2]
        offset = torch.tanh(offset).reshape(bs, N, -1)                       # [bs, N, 2]

        # 稳定性约束
        eps = 1e-6
        cho1_clamped = torch.clamp(para[:, :, 0], min=eps)
        cho2_unchanged = para[:, :, 1]
        cho3_clamped = torch.clamp(para[:, :, 2], min=eps)
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
                get_xyz = get_coord(lr_h * 2, lr_w * 2).to(feat.device)  # [N, 2]

            # # Adjust coordinates using offsets
            # xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / wf - 1 / W
            # xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / hf - 1 / H

            # Adjust coordinates using offsets
            xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / lr_w - 1 / W
            xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / lr_h - 1 / H
            get_xyz = torch.cat((xyz1, xyz2), dim=1)

            # Adjust Gaussian parameters
            weighted_cholesky = para_ / 4
            weighted_opacity = torch.ones(color_.shape[0], 1, device=feat.device)
            
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

    def forward(self, inp, scale, mask=None):
        """
        Forward pass for the ContinuousGaussian module.
        
        Args:
            inp  (torch.Tensor): Input LR image [bs, 1, H, W].
            scale (torch.Tensor): Scaling factor.
            mask (torch.Tensor | None): Segmentation mask [bs, 1, H_hr, W_hr],
                                        integer labels as float32. None = uniform grid.
        Returns:
            torch.Tensor: High-resolution output image.
        """
        self.gen_feat(inp, mask=mask)      # 生成特征 + 计算 sample_coords
        image = self.query_output(inp, scale)
        return image

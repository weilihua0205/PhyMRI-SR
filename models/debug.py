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
    bs, C, hf, wf = feat.shape  # Batch size, channels, and feature dimensions
    num_pixels = hf * wf  # Total number of pixels in feature map
    num_gaussians = num_pixels * 4  # Each pixel corresponds to 4 Gaussians

    # Process features for color prediction
    # Reshape from (bs, C, hf, wf) to (bs, num_gaussians, C)
    # First reshape to (bs, C, num_pixels), then duplicate 4 times for 4 Gaussians per pixel
    para_c = self.feat.reshape(bs, C, num_pixels).permute(0, 2, 1)  # (bs, num_pixels, C)
    para_c = para_c.unsqueeze(2).expand(bs, num_pixels, 4, C).reshape(bs, num_gaussians, C)
    # Reshape to (bs*num_gaussians, C) and feed to MLP
    color = self.mlp(para_c.reshape(-1, C))
    color = color.reshape(bs, num_gaussians, -1)

    # Process features for Gaussian convariance parameter estimation
    para_c = self.leaky_relu(self.feat)
    para = self.conv1(para_c)  # para shape: (bs, 512, hf, wf)
    vector = self.mlp_vector(self.gau_dict.to(para.device)) # Transform Gaussian covariance dictionary to increase dimensions
    # Reshape para from (bs, 512, hf, wf) to (512, bs*num_gaussians)
    para = para.reshape(bs, 512, num_pixels).permute(0, 2, 1)  # (bs, num_pixels, 512)
    para = para.unsqueeze(2).expand(bs, num_pixels, 4, 512).reshape(bs, num_gaussians, 512)
    para = para.permute(2, 0, 1).reshape(512, -1)  # (512, bs*num_gaussians)
    para = vector @ para  # This calculates the similarity between para and each element in the dictionary
    para = torch.softmax(para, dim=0)  # Normalize the similarity scores to produce weights using softmax
    para = para.permute(1, 0) @ self.gau_dict.to(para.device) # Compute the weighted sum of dictionary elements to get the final covariance
    para = para.reshape(bs, num_gaussians, -1)

    # Process features for offset prediction
    # para_c shape: (bs, 256, hf, wf), reshape to (bs*num_gaussians, C)
    para_c_for_offset = self.leaky_relu(self.feat)
    para_c_for_offset = para_c_for_offset.reshape(bs, C, num_pixels).permute(0, 2, 1)  # (bs, num_pixels, C)
    para_c_for_offset = para_c_for_offset.unsqueeze(2).expand(bs, num_pixels, 4, C).reshape(bs, num_gaussians, C)
    offset = self.mlp_offset(para_c_for_offset.reshape(-1, C))
    offset = torch.tanh(offset).reshape(bs, num_gaussians, -1)

    # Generate output predictions for each image in the batch
    for i in range(bs):
        offset_ = offset[i, :, :].squeeze(0)
        color_ = color[i, :, :].squeeze(0)
        para_ = para[i, :, :].squeeze(0)

        # Generate coordinate grid for the feature map
        # Each feature pixel generates 4 Gaussians at sub-pixel positions
        get_xyz = torch.tensor(get_coord(hf, wf)).reshape(hf, wf, 2).cuda()
        get_xyz = get_xyz.reshape(num_pixels, 2)
        # Duplicate each coordinate 4 times for 4 Gaussians per pixel
        get_xyz = get_xyz.unsqueeze(1).expand(num_pixels, 4, 2).reshape(num_gaussians, 2)

        # Adjust coordinates using offsets
        xyz1 = get_xyz[:, 0:1] + 2 * window_size * offset_[:, 0:1] / wf - 1 / W
        xyz2 = get_xyz[:, 1:2] + 2 * window_size * offset_[:, 1:2] / hf - 1 / H
        get_xyz = torch.cat((xyz1, xyz2), dim=1)

        # Adjust Gaussian parameters
        weighted_cholesky = para_ / 4
        weighted_opacity = torch.ones(color_.shape[0], 1).cuda()
        weighted_cholesky[:, 0] *= scale2
        weighted_cholesky[:, 1] *= scale1
        weighted_cholesky[:, 2] *= scale1

        # Perform Gaussian projection and rasterization
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            get_xyz, weighted_cholesky, H, W, self.tile_bounds
        )
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
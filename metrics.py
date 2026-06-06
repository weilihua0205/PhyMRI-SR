"""
Extended evaluation metrics for MRI super-resolution validity analysis.

The metrics in this module are used to evaluate the effectiveness and validity
of mask-guided sampling and T2 physics-prior modeling. They include tissue-aware
PSNR/SSIM, LPIPS, NMSE, VIF, FID, HFEN, and DISTS.

Usage:
    from metrics import MetricsCalculator
    calc = MetricsCalculator(device='cuda')
    results = calc.compute_sample(pred, gt, mask=mask)
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. Tissue-specific PSNR / SSIM
# ============================================================

TISSUE_NAMES = {0: 'BG', 1: 'CSF', 2: 'GM', 3: 'WM'}


def calc_tissue_psnr(pred, gt, mask, tissue_labels=None):
    """
    Compute PSNR separately for each tissue label in the segmentation mask.

    Args:
        pred: Predicted image tensor [B, 1, H, W], range [0, 1].
        gt: Ground-truth image tensor [B, 1, H, W], range [0, 1].
        mask: Segmentation label tensor [B, 1, H, W], with labels such as
            0=BG, 1=CSF, 2=GM, 3=WM.
        tissue_labels: Optional list of labels to evaluate.

    Returns:
        Dictionary mapping tissue names to PSNR values. Labels with too few
        pixels are skipped to avoid unstable measurements.
    """
    if tissue_labels is None:
        tissue_labels = [0, 1, 2, 3]

    results = {}
    pred_flat = pred.float()
    gt_flat = gt.float()
    mask_flat = mask.float()

    # Resize the mask to prediction resolution when evaluating different scales.
    if mask_flat.shape[-2:] != pred_flat.shape[-2:]:
        mask_flat = F.interpolate(mask_flat, size=pred_flat.shape[-2:], mode='nearest')

    for lbl in tissue_labels:
        name = TISSUE_NAMES.get(lbl, f'label_{lbl}')
        region = mask_flat == lbl

        num_pixels = region.sum().item()
        if num_pixels < 10:
            continue

        diff_sq = ((pred_flat - gt_flat) ** 2) * region.float()
        mse = diff_sq.sum() / num_pixels

        if mse < 1e-10:
            results[name] = 100.0
        else:
            psnr = -10 * torch.log10(mse)
            results[name] = torch.clamp(psnr, 0.0, 100.0).item()

    return results


def calc_tissue_ssim(pred, gt, mask, tissue_labels=None, window_size=7, sigma=1.5):
    """
    Compute SSIM separately for each tissue label in the segmentation mask.

    A full SSIM map is computed first, then averaged only over pixels belonging
    to each tissue region.

    Args:
        pred: Predicted image tensor [B, 1, H, W], range [0, 1].
        gt: Ground-truth image tensor [B, 1, H, W], range [0, 1].
        mask: Segmentation label tensor [B, 1, H, W].
        tissue_labels: Optional list of labels to evaluate.
        window_size: SSIM Gaussian window size.
        sigma: Standard deviation of the Gaussian SSIM window.

    Returns:
        Dictionary mapping tissue names to SSIM values.
    """
    if tissue_labels is None:
        tissue_labels = [0, 1, 2, 3]

    pred = pred.float().clamp(0, 1)
    gt = gt.float().clamp(0, 1)
    mask_r = mask.float()

    if mask_r.shape[-2:] != pred.shape[-2:]:
        mask_r = F.interpolate(mask_r, size=pred.shape[-2:], mode='nearest')

    ssim_map = _ssim_map(pred, gt, window_size=window_size, sigma=sigma)

    results = {}
    for lbl in tissue_labels:
        name = TISSUE_NAMES.get(lbl, f'label_{lbl}')
        region = (mask_r == lbl).float()

        # Remove window-border pixels because SSIM is less reliable near padding.
        pad = window_size // 2
        if pad > 0 and region.shape[-1] > 2 * pad and region.shape[-2] > 2 * pad:
            region_inner = region[..., pad:-pad, pad:-pad]
            ssim_inner = ssim_map[..., pad:-pad, pad:-pad]
        else:
            region_inner = region
            ssim_inner = ssim_map

        num_pixels = region_inner.sum().item()
        if num_pixels < 10:
            continue

        weighted_ssim = (ssim_inner * region_inner).sum() / num_pixels
        results[name] = torch.clamp(weighted_ssim, 0.0, 1.0).item()

    return results


def _ssim_map(sr, hr, window_size=7, sigma=1.5):
    """Compute a per-pixel SSIM map with shape [B, C, H, W]."""
    channel = sr.size(1)
    window = _create_window(window_size, channel, sigma).to(sr.device)

    pad = window_size // 2
    mu1 = F.conv2d(sr, window, padding=pad, groups=channel)
    mu2 = F.conv2d(hr, window, padding=pad, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=pad, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    return numerator / (denominator + 1e-12)


def _create_window(window_size, channel, sigma):
    """Create a channel-wise 2D Gaussian convolution window."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    window_2d = window_2d / window_2d.sum()
    window = window_2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# ============================================================
# 2. LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================

_lpips_net = None


def get_lpips_net(device='cuda'):
    """Return a cached LPIPS network instance."""
    global _lpips_net
    if _lpips_net is None:
        try:
            import lpips
            _lpips_net = lpips.LPIPS(net='alex', verbose=False).to(device)
            _lpips_net.eval()
        except ImportError:
            raise ImportError(
                "LPIPS requires the 'lpips' package. Install with: pip install lpips"
            )
    return _lpips_net


def calc_lpips(pred, gt, lpips_net=None):
    """
    Compute LPIPS; lower is better.

    LPIPS expects 3-channel inputs. Single-channel MRI images are repeated across
    RGB channels, and [0, 1] inputs are passed with normalize=True.
    """
    if lpips_net is None:
        lpips_net = get_lpips_net(pred.device)

    if pred.size(1) == 1:
        pred_in = pred.repeat(1, 3, 1, 1)
        gt_in = gt.repeat(1, 3, 1, 1)
    else:
        pred_in = pred
        gt_in = gt

    with torch.no_grad():
        lpips_val = lpips_net(pred_in, gt_in, normalize=True)

    if lpips_val.dim() > 0:
        lpips_val = lpips_val.mean()

    return lpips_val.item()


# ============================================================
# 3. NMSE (Normalized Mean Squared Error)
# ============================================================

def calc_nmse(pred, gt):
    """
    Compute NMSE, a standard MRI reconstruction metric; lower is better.

    NMSE = ||pred - gt||^2 / ||gt||^2
    """
    diff_sq = ((pred - gt) ** 2).sum()
    gt_sq = (gt ** 2).sum()

    if gt_sq < 1e-10:
        return 0.0

    nmse = (diff_sq / gt_sq).item()
    return nmse


# ============================================================
# 4. VIF (Visual Information Fidelity)
# ============================================================

def calc_vif(pred, gt):
    """
    Compute VIF-P (Visual Information Fidelity); higher is better.

    This implementation uses piq.vif_p and supports single-channel images.
    """
    try:
        import piq
    except ImportError:
        raise ImportError(
            "VIF requires the 'piq' package. Install with: pip install piq"
        )

    pred_in = pred.float().clamp(0, 1)
    gt_in = gt.float().clamp(0, 1)

    with torch.no_grad():
        vif_val = piq.vif_p(pred_in, gt_in, data_range=1.0)

    if vif_val.dim() > 0:
        vif_val = vif_val.mean()

    return vif_val.item()


# ============================================================
# 5. FID (Frechet Inception Distance)
# ============================================================

class FIDCalculator:
    """
    Batch-level FID calculator.

    FID is a distribution-level metric, so features are collected across all
    samples and computed once at the end. Results from fewer than 50 samples are
    useful only as a rough reference.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.pred_features = []
        self.gt_features = []
        self._inception = None

    def _get_inception(self):
        """Lazily load the InceptionV3 feature extractor."""
        if self._inception is None:
            try:
                from torchvision.models import inception_v3
                model = inception_v3(pretrained=True, transform_input=False)
                # Remove the classification head and use 2048-D features.
                model.fc = torch.nn.Identity()
                model = model.to(self.device)
                model.eval()
                self._inception = model
            except Exception as e:
                warnings.warn(f"FID: Cannot load InceptionV3: {e}. FID will not be computed.")
                return None
        return self._inception

    def _extract_features(self, images):
        """
        Extract Inception features from images.

        Args:
            images: Tensor [B, C, H, W], range [0, 1].

        Returns:
            Numpy array [B, 2048], or None if Inception is unavailable.
        """
        model = self._get_inception()
        if model is None:
            return None

        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        if images.shape[-1] != 299 or images.shape[-2] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to ImageNet statistics expected by InceptionV3.
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            feats = model(images)

        if isinstance(feats, tuple):
            feats = feats[0]

        return feats.cpu().numpy()

    def add_batch(self, pred, gt):
        """Collect features for one prediction/ground-truth batch."""
        pred_feat = self._extract_features(pred)
        gt_feat = self._extract_features(gt)

        if pred_feat is not None and gt_feat is not None:
            self.pred_features.append(pred_feat)
            self.gt_features.append(gt_feat)

    def compute(self):
        """
        Compute FID; lower is better and 0 means identical feature distributions.

        Returns -1 when FID cannot be computed.
        """
        if len(self.pred_features) == 0 or len(self.gt_features) == 0:
            warnings.warn("FID: No features collected, returning -1")
            return -1.0

        pred_feats = np.concatenate(self.pred_features, axis=0)
        gt_feats = np.concatenate(self.gt_features, axis=0)

        n_samples = pred_feats.shape[0]
        if n_samples < 2:
            warnings.warn("FID: Need at least 2 samples, returning -1")
            return -1.0

        if n_samples < 50:
            warnings.warn(f"FID: Only {n_samples} samples (recommended >= 50). Result is reference only.")

        mu1 = np.mean(pred_feats, axis=0)
        mu2 = np.mean(gt_feats, axis=0)
        sigma1 = np.cov(pred_feats, rowvar=False)
        sigma2 = np.cov(gt_feats, rowvar=False)

        return self._frechet_distance(mu1, sigma1, mu2, sigma2)

    @staticmethod
    def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Compute the Frechet distance between two multivariate Gaussians."""
        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        # Numerical precision may introduce a tiny imaginary component.
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                warnings.warn(f"FID: Imaginary component {np.max(np.abs(covmean.imag)):.4f}")
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)

    def reset(self):
        """Clear collected features."""
        self.pred_features = []
        self.gt_features = []


# ============================================================
# 6. HFEN (High-Frequency Error Norm)
# ============================================================

def calc_hfen(pred, gt, sigma=1.5, kernel_size=15):
    """
    Compute HFEN (High-Frequency Error Norm); lower is better.

    HFEN applies a Laplacian-of-Gaussian filter to emphasize high-frequency
    structures, then compares prediction and ground truth in that filtered space.
    It is often more sensitive than PSNR/SSIM to edge and detail fidelity in MRI
    super-resolution.
    """
    pred_f = pred.float().clamp(0, 1)
    gt_f = gt.float().clamp(0, 1)

    channel = pred_f.size(1)
    log_kernel = _make_log_kernel(kernel_size, sigma, pred_f.device)
    log_kernel = log_kernel.expand(channel, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    pred_hf = F.conv2d(pred_f, log_kernel, padding=pad, groups=channel)
    gt_hf = F.conv2d(gt_f, log_kernel, padding=pad, groups=channel)

    diff_norm = torch.norm(pred_hf - gt_hf)
    gt_norm = torch.norm(gt_hf)

    if gt_norm < 1e-10:
        return 0.0

    return (diff_norm / gt_norm).item()


def _make_log_kernel(size, sigma, device):
    """
    Create a normalized Laplacian-of-Gaussian convolution kernel.

    Args:
        size: Odd kernel size.
        sigma: Gaussian standard deviation.
        device: Torch device.

    Returns:
        Tensor [1, 1, size, size].
    """
    half = size // 2
    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        torch.arange(-half, half + 1, dtype=torch.float32, device=device),
        indexing='ij',
    )
    r2 = x ** 2 + y ** 2
    s2 = sigma ** 2
    kernel = -(1.0 / (np.pi * s2 ** 2)) * (1.0 - r2 / (2.0 * s2)) * torch.exp(-r2 / (2.0 * s2))
    # Zero-mean normalization makes flat regions produce near-zero response.
    kernel = kernel - kernel.mean()
    return kernel.unsqueeze(0).unsqueeze(0)


# ============================================================
# 7. DISTS wrapper
# ============================================================

_dists_metric = None


def get_dists_metric(device='cuda'):
    """Return a cached DISTS metric instance."""
    global _dists_metric
    if _dists_metric is None:
        try:
            from DISTS_pytorch import DISTS
            _dists_metric = DISTS().to(device)
            _dists_metric.eval()
        except ImportError:
            raise ImportError(
                "DISTS requires the 'DISTS_pytorch' package. "
                "Install with: pip install DISTS_pytorch"
            )
    return _dists_metric


def calc_dists(pred, gt, dists_metric=None):
    """Compute DISTS; lower is better."""
    if dists_metric is None:
        dists_metric = get_dists_metric(pred.device)

    if pred.size(1) == 1:
        pred_in = pred.repeat(1, 3, 1, 1)
        gt_in = gt.repeat(1, 3, 1, 1)
    else:
        pred_in = pred
        gt_in = gt

    pred_in = pred_in.clamp(0, 1)
    gt_in = gt_in.clamp(0, 1)

    with torch.no_grad():
        dists_val = dists_metric(pred_in, gt_in)

    if dists_val.dim() > 0:
        dists_val = dists_val.mean()

    return dists_val.item()


# ============================================================
# 8. Unified metrics calculator
# ============================================================

class MetricsCalculator:
    """
    Unified metric calculator for MRI SR evaluation validity checks.

    FID is accumulated during compute_sample() and should be finalized after all
    samples have been processed.
    """

    def __init__(self, device='cuda', use_lpips=True, use_dists=True,
                 use_vif=True, use_fid=True, use_hfen=True):
        self.device = device
        self.use_lpips = use_lpips
        self.use_dists = use_dists
        self.use_vif = use_vif
        self.use_fid = use_fid
        self.use_hfen = use_hfen

        # Lazily initialized metric backends.
        self._lpips_net = None
        self._dists_metric = None
        self._fid_calc = None

        # Availability flags for optional dependencies.
        self.lpips_available = True
        self.dists_available = True
        self.vif_available = True
        self.fid_available = True

        self._check_dependencies()

    def _check_dependencies(self):
        """Check whether optional metric dependencies are available."""
        if self.use_lpips:
            try:
                import lpips
            except ImportError:
                warnings.warn("lpips not installed. LPIPS will be skipped. Install: pip install lpips")
                self.lpips_available = False

        if self.use_dists:
            try:
                from DISTS_pytorch import DISTS
            except ImportError:
                warnings.warn("DISTS_pytorch not installed. DISTS will be skipped. Install: pip install DISTS_pytorch")
                self.dists_available = False

        if self.use_vif:
            try:
                import piq
            except ImportError:
                warnings.warn("piq not installed. VIF will be skipped. Install: pip install piq")
                self.vif_available = False

        if self.use_fid:
            try:
                from scipy import linalg
                from torchvision.models import inception_v3
            except ImportError:
                warnings.warn("scipy/torchvision not available. FID will be skipped.")
                self.fid_available = False

    def _get_lpips(self):
        if self._lpips_net is None and self.lpips_available:
            self._lpips_net = get_lpips_net(self.device)
        return self._lpips_net

    def _get_dists(self):
        if self._dists_metric is None and self.dists_available:
            self._dists_metric = get_dists_metric(self.device)
        return self._dists_metric

    def _get_fid_calc(self):
        if self._fid_calc is None and self.fid_available:
            self._fid_calc = FIDCalculator(device=self.device)
        return self._fid_calc

    def compute_sample(self, pred, gt, mask=None):
        """
        Compute all enabled per-sample or per-batch metrics.

        Args:
            pred: Predicted image tensor [B, C, H, W], range [0, 1].
            gt: Ground-truth image tensor [B, C, H, W], range [0, 1].
            mask: Optional segmentation label tensor [B, 1, H, W].

        Returns:
            Dictionary containing all successfully computed metrics.
        """
        results = {}

        results['nmse'] = calc_nmse(pred, gt)

        if self.use_lpips and self.lpips_available:
            try:
                results['lpips'] = calc_lpips(pred, gt, self._get_lpips())
            except Exception as e:
                warnings.warn(f"LPIPS computation failed: {e}")
                results['lpips'] = -1.0

        if self.use_dists and self.dists_available:
            try:
                results['dists'] = calc_dists(pred, gt, self._get_dists())
            except Exception as e:
                warnings.warn(f"DISTS computation failed: {e}")
                results['dists'] = -1.0

        if self.use_vif and self.vif_available:
            try:
                results['vif'] = calc_vif(pred, gt)
            except Exception as e:
                warnings.warn(f"VIF computation failed: {e}")
                results['vif'] = -1.0

        if self.use_hfen:
            try:
                results['hfen'] = calc_hfen(pred, gt)
            except Exception as e:
                warnings.warn(f"HFEN computation failed: {e}")
                results['hfen'] = -1.0

        if mask is not None:
            try:
                results['tissue_psnr'] = calc_tissue_psnr(pred, gt, mask)
                results['tissue_ssim'] = calc_tissue_ssim(pred, gt, mask)
            except Exception as e:
                warnings.warn(f"Tissue metrics computation failed: {e}")
                results['tissue_psnr'] = {}
                results['tissue_ssim'] = {}

        # FID is accumulated here and computed later by compute_fid().
        if self.use_fid and self.fid_available:
            try:
                fid_calc = self._get_fid_calc()
                if fid_calc is not None:
                    fid_calc.add_batch(pred, gt)
            except Exception as e:
                warnings.warn(f"FID feature extraction failed: {e}")

        return results

    def compute_fid(self):
        """Compute accumulated FID after all samples have been processed."""
        if not self.use_fid or not self.fid_available:
            return -1.0

        fid_calc = self._get_fid_calc()
        if fid_calc is None:
            return -1.0

        return fid_calc.compute()

    def reset_fid(self):
        """Reset accumulated FID features."""
        if self._fid_calc is not None:
            self._fid_calc.reset()

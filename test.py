"""
Testing SR model on:
- Simulated multi-resolution 3T → 64mT dataset
- Real paired 64mT–3T dataset
- Real paired 3T-5T multi-resolution dataset
output-dir:
    #   sr_npy/*.npy: model super-resolved outputs saved as float32 arrays.
    #   visualizations/*.png: per-sample panels showing upsampled LR, SR, GT, and error.
    #   per_sample_metrics.csv: PSNR, SSIM, DISTS, HFEN, and effective scales per sample.
    #   summary.json: mean metrics and absolute paths to the generated result folders/files.
    #   metrics_overview.png: overview plots of all metrics across the test set.
"""
import argparse
import csv
import inspect
import json
import math
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import models
from datasets.wrappers import resize_fn
from metrics import calc_dists, calc_hfen


# Use a fixed seed because metric comparisons should be reproducible across runs.
def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path: str):
    # Some checkpoints may contain numpy scalar metadata saved by older training code.
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f'Checkpoint missing "model": {checkpoint_path}')
    return checkpoint


def load_array(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    return np.array(Image.open(path))


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    # Convert image-like arrays to channel-first tensors expected by the SR model.
    if arr.ndim == 2:
        tensor = torch.from_numpy(arr).float().unsqueeze(0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            tensor = torch.from_numpy(arr).float()
        else:
            tensor = torch.from_numpy(arr).float().permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        tensor = tensor / 255.0
    elif tensor.max().item() > 1.0:
        # Treat floating-point images outside [0, 1] as 8-bit intensity values.
        tensor = tensor / 255.0
    return tensor.clamp(0, 1)


class PairedRealDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, mask_dir=None):
        self.lr_map = self._scan_dir(lr_dir)
        self.hr_map = self._scan_dir(hr_dir)
        self.mask_map = self._scan_dir(mask_dir) if mask_dir else None

        # Match LR, HR, and optional mask files by filename stem to keep slices aligned.
        stems = sorted(set(self.lr_map) & set(self.hr_map))
        if self.mask_map is not None:
            stems = sorted(set(stems) & set(self.mask_map))
        if not stems:
            raise ValueError("No matched samples found in LR/HR/(mask) directories.")
        self.stems = stems

    @staticmethod
    def _scan_dir(root):
        mapping = {}
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() in {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                mapping[stem] = path
        return mapping

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        sample = {
            "name": stem,
            "inp": to_tensor(load_array(self.lr_map[stem])),
            "gt": to_tensor(load_array(self.hr_map[stem])),
        }
        if self.mask_map is not None:
            sample["mask"] = to_tensor(load_array(self.mask_map[stem]))
        return sample


def collate_one(batch):
    # Keep one sample per batch so the original sample name stays a plain string.
    sample = batch[0]
    output = {"name": sample["name"]}
    for key in ("inp", "gt", "mask"):
        if key in sample:
            output[key] = sample[key].unsqueeze(0)
    return output


def build_norm_tensors(data_norm, device):
    if not data_norm:
        return None
    # Store training-time normalization constants in broadcastable NCHW form.
    return {
        "inp_sub": torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device),
        "inp_div": torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device),
        "gt_sub_img": torch.FloatTensor(data_norm["gt"]["sub"]).view(1, -1, 1, 1).to(device),
        "gt_div_img": torch.FloatTensor(data_norm["gt"]["div"]).view(1, -1, 1, 1).to(device),
    }


def normalize_inp(inp, norm):
    if norm is None:
        return inp
    return (inp - norm["inp_sub"]) / norm["inp_div"]


def denorm_img(pred, norm):
    if norm is None:
        return pred
    return pred * norm["gt_div_img"] + norm["gt_sub_img"]


def prepare_input_for_sr_scale(inp, gt, sr_scale=None):
    if sr_scale is None:
        return inp
    if sr_scale <= 0:
        raise ValueError(f"sr_scale must be > 0, got {sr_scale}")

    gt_h, gt_w = gt.shape[-2:]
    # sr_scale is defined as HR/LR, so the LR target is derived from the GT size.
    target_h = max(1, int(round(gt_h / sr_scale)))
    target_w = max(1, int(round(gt_w / sr_scale)))

    if inp.shape[-2:] == (target_h, target_w):
        return inp

    if inp.dim() == 4:
        resized = torch.stack([resize_fn(sample, (target_h, target_w)) for sample in inp], dim=0)
    elif inp.dim() == 3:
        resized = resize_fn(inp, (target_h, target_w)).unsqueeze(0)
    else:
        raise ValueError(f"Expected inp to be 3D or 4D tensor, got shape {tuple(inp.shape)}")

    return resized.to(dtype=inp.dtype, device=inp.device).clamp(0, 1)


def calc_psnr(sr, hr):
    diff = sr - hr
    mse = diff.pow(2).mean()
    if mse < 1e-10:
        return 100.0
    return torch.clamp(-10 * torch.log10(mse), 0.0, 100.0).item()


def calc_ssim(sr, hr, window_size=11, sigma=1.5):
    sr = sr.float().clamp(0, 1)
    hr = hr.float().clamp(0, 1)
    channel = sr.size(1)
    # Build the Gaussian SSIM window directly on the active device.
    coords = torch.arange(window_size, dtype=torch.float32, device=sr.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    window_2d = window_2d / window_2d.sum()
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(sr, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(hr, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(sr * sr, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )
    return torch.clamp(ssim_map.mean(), 0.0, 1.0).item()


def infer_scale_model(model, batch, norm):
    inp = batch["inp"]
    gt = batch["gt"]
    mask = batch.get("mask")

    inp_model = normalize_inp(inp, norm)
    h_ratio = gt.shape[-2] / inp.shape[-2]
    w_ratio = gt.shape[-1] / inp.shape[-1]
    if abs(h_ratio - w_ratio) > 1e-6:
        raise ValueError(f"Non-uniform scale: h={h_ratio}, w={w_ratio}")
    # ContinuousSR receives a scalar HR/LR scale alongside the normalized input.
    scale = torch.tensor([float(h_ratio)], device=inp.device)

    forward_params = inspect.signature(model.forward).parameters
    # Older checkpoints may not expose mask conditioning, so detect it at runtime.
    supports_mask = "mask" in forward_params

    with torch.no_grad():
        if supports_mask and mask is not None:
            pred = model(inp_model, scale, mask=mask)
        else:
            pred = model(inp_model, scale)

    pred = denorm_img(pred, norm).clamp(0, 1)
    return pred, gt


def tensor_to_save_numpy(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.float32)


def tensor_to_show_numpy(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu()
    if arr.dim() == 4:
        arr = arr[0]
    if arr.size(0) == 1:
        return arr[0].numpy()
    return arr.permute(1, 2, 0).contiguous().numpy()


def try_calc_dists(pred: torch.Tensor, gt: torch.Tensor):
    try:
        return float(calc_dists(pred, gt))
    except Exception as exc:
        warnings.warn(f"DISTS computation failed, will be saved as NaN: {exc}")
        return math.nan


def save_case_visualization(inp, pred, gt, save_path, sample_name, metrics_row):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Upsample LR only for visual comparison; the model prediction is saved separately.
    lr_up = F.interpolate(inp, size=gt.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    lr_np = tensor_to_show_numpy(lr_up)
    pred_np = tensor_to_show_numpy(pred)
    gt_np = tensor_to_show_numpy(gt)
    err_np = np.abs(pred_np - gt_np)

    if err_np.ndim == 3:
        # Collapse RGB/channel errors into a single heatmap for readability.
        err_show = err_np.mean(axis=-1)
    else:
        err_show = err_np

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("Input (upsampled)", lr_np, "gray"),
        ("SR", pred_np, "gray"),
        ("GT", gt_np, "gray"),
        ("|SR-GT|", err_show, "hot"),
    ]

    for ax, (title, img, cmap) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if title != "|SR-GT|" else None)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    metric_text = (
        f"{sample_name}\n"
        f"PSNR={metrics_row['psnr']:.4f}  "
        f"SSIM={metrics_row['ssim']:.4f}  "
        f"DISTS={metrics_row['dists']:.4f}\n"
        f"HFEN={metrics_row['hfen']:.4f}"
    )
    fig.suptitle(metric_text, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(rows, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample_names = [row["sample"] for row in rows]
    x = np.arange(len(rows))
    metrics = {
        "PSNR": [row["psnr"] for row in rows],
        "SSIM": [row["ssim"] for row in rows],
        "DISTS": [row["dists"] for row in rows],
        "HFEN": [row["hfen"] for row in rows],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    for ax, (title, values) in zip(axes, metrics.items()):
        arr = np.asarray(values, dtype=np.float32)
        ax.plot(x, arr, marker="o", linewidth=1.5, markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        if len(sample_names) <= 30:
            ax.set_xticks(x)
            ax.set_xticklabels(sample_names, rotation=60, ha="right", fontsize=8)
        else:
            # For large test sets, show a sparse set of labels to avoid unreadable plots.
            tick_idx = np.linspace(0, len(sample_names) - 1, min(10, len(sample_names)), dtype=int)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([sample_names[i] for i in tick_idx], rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_test_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config is None:
        return {}
    if "test" in config and isinstance(config["test"], dict):
        config = config["test"]
    return config


def resolve_arg(args, config, name, required=False):
    value = getattr(args, name)
    if value is None:
        value = config.get(name)
    if required and value is None:
        cli_name = name.replace("_", "-")
        raise ValueError(f"Missing required option: --{cli_name} or config key '{name}'")
    return value


def resolve_data_paths(args, config):
    lr_dir = resolve_arg(args, config, "lr_dir")
    hr_dir = resolve_arg(args, config, "hr_dir")
    mask_dir = resolve_arg(args, config, "mask_dir")

    data_root = config.get("data_root")
    if data_root:
        lr_dir = lr_dir or os.path.join(data_root, config.get("lr_subdir", "LR"))
        hr_dir = hr_dir or os.path.join(data_root, config.get("hr_subdir", "HR"))
        mask_subdir = config.get("mask_subdir", "mask")
        if mask_dir is None and mask_subdir:
            mask_dir = os.path.join(data_root, mask_subdir)

    if lr_dir is None:
        raise ValueError("Missing required option: --lr-dir, config key 'lr_dir', or config key 'data_root'")
    if hr_dir is None:
        raise ValueError("Missing required option: --hr-dir, config key 'hr_dir', or config key 'data_root'")
    return lr_dir, hr_dir, mask_dir


def main():
    parser = argparse.ArgumentParser(
        description="Test ContinuousSR on real paired 64mT-3T data and save SR outputs, metrics, and visualizations."
    )
    parser.add_argument("--config", default=None, help="Optional test YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint path")
    parser.add_argument("--lr-dir", default=None, help="Real 64mT folder")
    parser.add_argument("--hr-dir", default=None, help="Real 3T folder")
    parser.add_argument("--mask-dir", default=None, help="Optional mask folder aligned to HR")
    parser.add_argument("--output-dir", default=None, help="Directory to save outputs")
    parser.add_argument("--sr-scale", type=float, default=None, help="Override input scale by online resizing LR")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    test_config = load_test_config(args.config)
    checkpoint_path = resolve_arg(args, test_config, "checkpoint", required=True)
    lr_dir, hr_dir, mask_dir = resolve_data_paths(args, test_config)
    output_dir = resolve_arg(args, test_config, "output_dir", required=True)
    sr_scale = resolve_arg(args, test_config, "sr_scale")
    device_name = resolve_arg(args, test_config, "device") or "cuda"
    num_workers = resolve_arg(args, test_config, "num_workers")
    seed = resolve_arg(args, test_config, "seed")
    if num_workers is None:
        num_workers = 0
    if seed is None:
        seed = 1234

    setup_seed(seed)
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    norm = build_norm_tensors(config.get("data_norm"), device)

    dataset = PairedRealDataset(lr_dir, hr_dir, mask_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_one,
    )

    model = models.make(checkpoint["model"], load_sd=True).to(device)
    model.eval()

    sr_dir = os.path.join(output_dir, "sr_npy")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    rows = []
    for batch in loader:
        name = batch["name"]
        batch["inp"] = batch["inp"].to(device)
        batch["gt"] = batch["gt"].to(device)
        if "mask" in batch:
            batch["mask"] = batch["mask"].to(device)

        # Optionally reshape LR online to evaluate arbitrary HR/LR scale factors.
        batch["inp"] = prepare_input_for_sr_scale(batch["inp"], batch["gt"], sr_scale=sr_scale)
        pred, gt = infer_scale_model(model, batch, norm)

        psnr = calc_psnr(pred, gt)
        ssim = calc_ssim(pred, gt)
        dists = try_calc_dists(pred, gt)
        hfen = float(calc_hfen(pred, gt))
        scale_h = gt.shape[-2] / batch["inp"].shape[-2]
        scale_w = gt.shape[-1] / batch["inp"].shape[-1]

        row = {
            "sample": name,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "dists": float(dists),
            "hfen": float(hfen),
            "scale_h": float(scale_h),
            "scale_w": float(scale_w),
        }
        rows.append(row)

        # Save both machine-readable SR arrays and per-sample visual diagnostics.
        np.save(os.path.join(sr_dir, name + ".npy"), tensor_to_save_numpy(pred))
        save_case_visualization(
            batch["inp"],
            pred,
            gt,
            os.path.join(vis_dir, name + ".png"),
            name,
            row,
        )

    csv_path = os.path.join(output_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample", "psnr", "ssim", "dists", "hfen", "scale_h", "scale_w"],
        )
        writer.writeheader()
        writer.writerows(rows)

    def safe_mean(key):
        vals = np.asarray([row[key] for row in rows], dtype=np.float32)
        # DISTS can be NaN if its backend fails, so summaries must ignore NaNs.
        return float(np.nanmean(vals)) if len(vals) else math.nan

    summary = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "num_samples": len(rows),
        "psnr_mean": safe_mean("psnr"),
        "ssim_mean": safe_mean("ssim"),
        "dists_mean": safe_mean("dists"),
        "hfen_mean": safe_mean("hfen"),
        "avg_scale_h": safe_mean("scale_h"),
        "avg_scale_w": safe_mean("scale_w"),
        "sr_dir": os.path.abspath(sr_dir),
        "vis_dir": os.path.abspath(vis_dir),
        "metrics_csv": os.path.abspath(csv_path),
    }

    summary_json = os.path.join(output_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_plot = os.path.join(output_dir, "metrics_overview.png")
    save_summary_plot(rows, summary_plot)

    print(f"Samples: {summary['num_samples']}")
    print(f"PSNR mean: {summary['psnr_mean']:.4f}")
    print(f"SSIM mean: {summary['ssim_mean']:.4f}")
    print(f"DISTS mean: {summary['dists_mean']:.4f}")
    print(f"HFEN mean: {summary['hfen_mean']:.4f}")
    print(f"Saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

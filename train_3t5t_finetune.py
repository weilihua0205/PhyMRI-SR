# i"""
# Fine-tune a pretrained ContinuousSR model on real 3T-5T multi-resolution paired data.

# The script uses real 3T MRI slices as low-resolution inputs and matched 5T MRI
# slices as high-resolution targets. It starts from an existing checkpoint, then
# adapts the model to the 3T-to-5T paired reconstruction task.
# """

import argparse
import inspect
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from evaluate import calc_psnr, calc_ssim
from losses import make as make_loss
from schedulers import make_scheduler


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_collate(batch):
    # Clone tensors during collation to avoid non-contiguous batch tensors from wrapped datasets.
    if not batch:
        return batch
    elem = batch[0]
    if isinstance(elem, dict):
        out = {}
        for key in elem:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.stack([v.clone().contiguous() for v in values], dim=0)
            else:
                out[key] = values
        return out
    if isinstance(elem, torch.Tensor):
        return torch.stack([x.clone().contiguous() for x in batch], dim=0)
    return batch


def make_paired_dataset(lr_dir, hr_dir, mask_dir=None, inp_size=64, augment=False, cache="in_memory", repeat=1):
    # Build either a plain paired dataset or a mask-conditioned paired dataset.
    if mask_dir:
        dataset_spec = {
            "name": "paired-npy-folders-with-mask",
            "args": {
                "root_path_lr": lr_dir,
                "root_path_hr": hr_dir,
                "root_path_mask": mask_dir,
                "cache": cache,
                "repeat": repeat,
            },
        }
    else:
        dataset_spec = {
            "name": "paired-npy-folders",
            "args": {
                "root_path_1": lr_dir,
                "root_path_2": hr_dir,
                "cache": cache,
                "repeat": repeat,
            },
        }

    wrapper_spec = {
        "name": "sr-implicit-paired",
        "args": {
            "inp_size": inp_size,
            "augment": augment,
        },
    }
    base = datasets.make(dataset_spec)
    if not mask_dir:
        # Plain paired folders rely on sorted filenames, so explicitly reject mismatched pairs.
        names_1 = getattr(base.dataset_1, "filenames", None)
        names_2 = getattr(base.dataset_2, "filenames", None)
        if names_1 is not None and names_2 is not None and names_1 != names_2:
            raise ValueError(
                "3T/5T filenames are not aligned. "
                "Please make sure both folders contain the same sorted .npy filenames."
            )
    return datasets.make(wrapper_spec, args={"dataset": base}), dataset_spec, wrapper_spec


def load_pretrained_model(pretrain_path, device):
    print(f"==> Loading pretrained weights: {pretrain_path}")
    # Some training checkpoints contain numpy scalar metadata.
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f'Checkpoint missing "model": {pretrain_path}')

    model = models.make(checkpoint["model"], load_sd=True).to(device)
    model.train()

    pre_epoch = checkpoint.get("epoch", None)
    pre_metric = checkpoint.get("best_metric", None)
    if pre_epoch is not None:
        print(f"==> Source checkpoint epoch: {pre_epoch}")
    if pre_metric is not None:
        print(f"==> Source checkpoint best_metric: {pre_metric:.4f}")
    return model, checkpoint


def build_config(args, source_checkpoint):
    # Save a self-contained config snapshot that records both source weights and fine-tune data.
    model_spec = {
        "name": source_checkpoint["model"]["name"],
        "args": source_checkpoint["model"]["args"],
    }
    return {
        "pretrain": args.pretrain,
        "train_dataset": {
            "lr_dir": args.train_3t_dir,
            "hr_dir": args.train_5t_dir,
            "mask_dir": args.train_mask_dir,
            "inp_size": args.inp_size,
            "augment": args.augment,
            "cache": args.cache,
            "repeat": args.repeat,
        },
        "val_dataset": {
            "lr_dir": args.val_3t_dir,
            "hr_dir": args.val_5t_dir,
            "mask_dir": args.val_mask_dir,
            "inp_size": args.val_inp_size,
            "augment": False,
            "cache": args.cache,
            "repeat": 1,
        } if args.val_3t_dir and args.val_5t_dir else None,
        "model": model_spec,
        "optimizer": {
            "name": "adam",
            "args": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "lr_scheduler": {
            "name": "MultiStepLR",
            "args": {
                "milestones": args.milestones,
                "gamma": args.gamma,
            },
        } if args.milestones else None,
        "loss": {
            "name": "CombinedLoss",
            "args": {
                "losses_dict": {
                    "L1Loss": args.l1_weight,
                    "FrequencyLoss": args.frequency_weight,
                    "GradientLoss": args.gradient_weight,
                },
            },
        },
        "data_norm": {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        },
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "num_epochs": args.epochs,
        "val_interval": args.val_interval,
        "save_interval": args.save_interval,
        "grad_clip": args.grad_clip,
        "eval_type": args.eval_type,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
    }


def move_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def train_one_epoch(model, loader, criterion, optimizer, device, config, epoch):
    model.train()
    loss_avg = utils.Averager()
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}", ncols=100)
    data_norm = config.get("data_norm")
    # The pretrained checkpoint may or may not support mask conditioning.
    supports_mask = "mask" in inspect.signature(model.forward).parameters

    if data_norm:
        # Keep input and target normalization consistent with the original training convention.
        inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device)
        inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device)
        gt_sub = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, -1, 1, 1).to(device)
        gt_div = torch.FloatTensor(data_norm["gt"]["div"]).view(1, -1, 1, 1).to(device)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        if data_norm:
            inp = (batch["inp"] - inp_sub) / inp_div
            gt = (batch["gt"] - gt_sub) / gt_div
        else:
            inp = batch["inp"]
            gt = batch["gt"]

        scale = batch["scale"]
        mask = batch.get("mask")

        # Preserve compatibility with older non-mask ContinuousSR checkpoints.
        if supports_mask:
            pred = model(inp, scale, mask=mask)
        else:
            pred = model(inp, scale)

        loss = criterion(pred, gt)
        optimizer.zero_grad()
        loss.backward()

        if config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()
        loss_avg.add(loss.item())
        pbar.set_postfix({"loss": f"{loss_avg.item():.4f}"})

    return loss_avg.item()


def validate_one_epoch(model, loader, device, config):
    model.eval()
    psnr_values = []
    ssim_values = []
    data_norm = config.get("data_norm")
    # Validation follows the same mask compatibility path as training.
    supports_mask = "mask" in inspect.signature(model.forward).parameters

    if data_norm:
        # Only the prediction is denormalized before metric computation; GT stays in image space.
        inp_sub = torch.FloatTensor(data_norm["inp"]["sub"]).view(1, -1, 1, 1).to(device)
        inp_div = torch.FloatTensor(data_norm["inp"]["div"]).view(1, -1, 1, 1).to(device)
        gt_sub = torch.FloatTensor(data_norm["gt"]["sub"]).view(1, -1, 1, 1).to(device)
        gt_div = torch.FloatTensor(data_norm["gt"]["div"]).view(1, -1, 1, 1).to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", ncols=100):
            batch = move_batch_to_device(batch, device)
            if data_norm:
                inp = (batch["inp"] - inp_sub) / inp_div
            else:
                inp = batch["inp"]
            gt = batch["gt"]
            scale = batch["scale"]
            mask = batch.get("mask")

            if supports_mask:
                pred = model(inp, scale, mask=mask)
            else:
                pred = model(inp, scale)
            if data_norm:
                pred = pred * gt_div + gt_sub
            pred = pred.clamp(0, 1)

            # Track standard reconstruction quality metrics for model selection.
            psnr_values.append(float(calc_psnr(pred, gt, dataset=config.get("eval_type")).item()))
            ssim_values.append(float(calc_ssim(pred, gt, dataset=config.get("eval_type")).item()))

    model.train()
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def selection_score(psnr, ssim, metric):
    if metric == "psnr":
        return float(psnr)
    if metric == "ssim":
        return float(ssim)
    raise ValueError(f"Unsupported selection_metric: {metric}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_score, save_dir, config, name):
    # Store model architecture args together with weights so the checkpoint is reloadable.
    model_spec = {
        "name": config["model"]["name"],
        "args": config["model"]["args"],
        "sd": model.state_dict(),
    }
    checkpoint = {
        "model": model_spec,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_score,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(save_dir, name))


def apply_config_defaults(args):
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}

    train_dataset = cfg.get("train_dataset", {})
    train_args = train_dataset.get("dataset", {}).get("args", {})
    train_wrapper_args = train_dataset.get("wrapper", {}).get("args", {})
    val_dataset = cfg.get("val_dataset", {})
    val_args = val_dataset.get("dataset", {}).get("args", {})
    val_wrapper_args = val_dataset.get("wrapper", {}).get("args", {})

    args.train_3t_dir = args.train_3t_dir or train_args.get("root_path_1") or train_args.get("root_path_lr")
    args.train_5t_dir = args.train_5t_dir or train_args.get("root_path_2") or train_args.get("root_path_hr")
    args.train_mask_dir = args.train_mask_dir or train_args.get("root_path_mask")
    args.val_3t_dir = args.val_3t_dir or val_args.get("root_path_1") or val_args.get("root_path_lr")
    args.val_5t_dir = args.val_5t_dir or val_args.get("root_path_2") or val_args.get("root_path_hr")
    args.val_mask_dir = args.val_mask_dir or val_args.get("root_path_mask")

    args.batch_size = cfg.get("batch_size", args.batch_size)
    args.val_batch_size = val_dataset.get("batch_size", args.val_batch_size)
    args.num_workers = cfg.get("num_workers", args.num_workers)
    args.epochs = cfg.get("num_epochs", args.epochs)
    args.inp_size = train_wrapper_args.get("inp_size", args.inp_size)
    args.val_inp_size = val_wrapper_args.get("inp_size", args.val_inp_size)
    args.augment = bool(train_wrapper_args.get("augment", args.augment))
    args.cache = train_args.get("cache", args.cache)
    args.repeat = train_args.get("repeat", args.repeat)

    optimizer_args = cfg.get("optimizer", {}).get("args", {})
    args.lr = optimizer_args.get("lr", args.lr)
    args.weight_decay = optimizer_args.get("weight_decay", args.weight_decay)

    scheduler_args = cfg.get("lr_scheduler", {}).get("args", {})
    args.milestones = scheduler_args.get("milestones", args.milestones)
    args.gamma = scheduler_args.get("gamma", args.gamma)

    loss_weights = cfg.get("loss", {}).get("args", {}).get("losses_dict", {})
    args.l1_weight = loss_weights.get("L1Loss", args.l1_weight)
    args.frequency_weight = loss_weights.get("FrequencyLoss", args.frequency_weight)
    args.gradient_weight = loss_weights.get("GradientLoss", args.gradient_weight)

    args.val_interval = cfg.get("val_interval", args.val_interval)
    args.save_interval = cfg.get("save_interval", args.save_interval)
    args.selection_metric = cfg.get("selection_metric", args.selection_metric)
    args.eval_type = cfg.get("eval_type", args.eval_type)
    args.seed = cfg.get("seed", args.seed)
    args.grad_clip = cfg.get("grad_clip", args.grad_clip)

    if args.train_3t_dir is None or args.train_5t_dir is None:
        raise ValueError("Fine-tune config must provide train_dataset dataset args root_path_1/root_path_2.")
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune 64mT-3T ContinuousSR weights on paired 3T-5T data.")
    parser.add_argument("--config", default=None, help="Optional fine-tuning YAML config path.")
    parser.add_argument("--pretrain", default=r"save\meta_learning\real_mix_4sim_2real_real_mask\checkpoint_best.pth")
    parser.add_argument("--train-3t-dir", default=None, help="3T input .npy folder for training")
    parser.add_argument("--train-5t-dir", default=None, help="5T target .npy folder for training")
    parser.add_argument("--val-3t-dir", default=None, help="3T input .npy folder for validation")
    parser.add_argument("--val-5t-dir", default=None, help="5T target .npy folder for validation")
    parser.add_argument("--train-mask-dir", default=None, help="Optional training mask .npy folder aligned to 5T")
    parser.add_argument("--val-mask-dir", default=None, help="Optional validation mask .npy folder aligned to 5T")
    parser.add_argument("--save-name", default="finetune_3t5t_from_64mt3t")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--inp-size", type=int, default=64)
    parser.add_argument("--val-inp-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--milestones", type=int, nargs="*", default=[40, 80])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--l1-weight", type=float, default=0.7)
    parser.add_argument("--frequency-weight", type=float, default=0.15)
    parser.add_argument("--gradient-weight", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache", choices=["none", "bin", "in_memory"], default="in_memory")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["psnr", "ssim"], default="ssim")
    parser.add_argument("--eval-type", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return apply_config_defaults(parser.parse_args())


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Device: {device}")

    model, source_checkpoint = load_pretrained_model(args.pretrain, device)
    config = build_config(args, source_checkpoint)

    # Each fine-tuning run writes its config, logs, and checkpoints under save/<save-name>.
    save_dir = os.path.join("save", args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    with open(os.path.join(save_dir, "finetune_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"pretrain: {args.pretrain}\n")
        f.write(f"started_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    train_dataset, train_dataset_spec, train_wrapper_spec = make_paired_dataset(
        args.train_3t_dir,
        args.train_5t_dir,
        args.train_mask_dir,
        inp_size=args.inp_size,
        augment=args.augment,
        cache=args.cache,
        repeat=args.repeat,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate,
    )
    print(f"==> Training samples: {len(train_dataset)}")

    val_loader = None
    if args.val_3t_dir and args.val_5t_dir:
        # Validation is optional; when present, the best checkpoint is selected by PSNR or SSIM.
        val_dataset, _, _ = make_paired_dataset(
            args.val_3t_dir,
            args.val_5t_dir,
            args.val_mask_dir,
            inp_size=args.val_inp_size,
            augment=False,
            cache=args.cache,
            repeat=1,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=safe_collate,
        )
        print(f"==> Validation samples: {len(val_dataset)}")

    optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
    scheduler = make_scheduler(optimizer, config["lr_scheduler"]) if config.get("lr_scheduler") else None
    criterion = make_loss(config["loss"]).to(device)
    print(f"==> Parameters: {utils.compute_num_params(model, text=True)}")
    print(f"==> Loss: {config['loss']['args']['losses_dict']}")
    print(f"==> Save dir: {save_dir}")

    best_score = -float("inf")
    best_desc = "none"
    log_path = os.path.join(save_dir, "log.txt")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, epoch)
        lr = optimizer.param_groups[0]["lr"]
        line = f"Epoch {epoch + 1}/{args.epochs} loss={train_loss:.6f} lr={lr:.6g}"

        if scheduler is not None:
            scheduler.step()

        # Without validation, only latest/final checkpoints are meaningful.
        is_best = False
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            val_psnr, val_ssim = validate_one_epoch(model, val_loader, device, config)
            score = selection_score(val_psnr, val_ssim, args.selection_metric)
            is_best = score > best_score
            if is_best:
                best_score = score
                best_desc = f"PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}"
            line += f" val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f} best={best_desc}"

        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        save_checkpoint(model, optimizer, scheduler, epoch, best_score, save_dir, config, "checkpoint_latest.pth")
        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, best_score, save_dir, config, "checkpoint_best.pth")
        if (epoch + 1) % args.save_interval == 0:
            # Periodic snapshots make it possible to inspect or roll back intermediate training states.
            save_checkpoint(model, optimizer, scheduler, epoch, best_score, save_dir, config, f"checkpoint_epoch{epoch + 1:04d}.pth")

    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_score, save_dir, config, "checkpoint_final.pth")
    if val_loader is None:
        save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_score, save_dir, config, "checkpoint_best.pth")
    print(f"==> Fine-tuning done. Best {args.selection_metric}: {best_score:.4f} ({best_desc})")


if __name__ == "__main__":
    main()

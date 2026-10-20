#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune the restoration network on projector-distortion triplets.

    python train.py --epochs 30
    python train.py --data-root datasets/SampleData/sample_train --epochs 5
    python train.py --data-root /path/to/train --val-root /path/to/validation

Only the training split updates weights. After every epoch, the complete validation
split is restored at model.input_size without random crops or gradients. The best
checkpoint maximizes mean validation PSNR; sample_test is reserved for final scoring.
Validation uses surface targets, not detection labels. Its metrics and restoration
previews are saved under the run's validation/ directory.

The architecture comes from model.backend in configs/restoration.yaml: naf_se_unet
uses the ablation block; plain_cnn uses the plain_cnn block. A resumed checkpoint
carries its own architecture and config. Run directories identify the variant.

Loss: L1 + VGG perceptual + (1 - SSIM) + high-frequency Haar, weights from
configs/restoration.yaml (tuned by an Optuna sweep on this dataset).

The network predicts the residual, so the supervised quantity is
`(distorted - residual).clamp(-1, 1)` against the surface image. Checkpoints embed
their architecture config, so demo.py reconstructs the right architecture with no extra
flags.

Needs the training extra:  pip install -e .[train]
"""

import argparse
import csv
import gc
import math
import os
import sys
from datetime import datetime

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from projector_distortion.cli import SAMPLE_EVAL, SAMPLE_TRAIN, resolve_device  # noqa: E402
from projector_distortion.config import load_config, pick, resolve_path  # noqa: E402
from projector_distortion.data import (  # noqa: E402
    RESEARCH_DIRS, TripletPatchDataset, index_triplets, surface_id,
)
from projector_distortion.models.restoration import (  # noqa: E402
    architecture_name, config_from_settings, build_network, count_parameters,
    load_checkpoint, save_checkpoint,
)
from projector_distortion.utils.image import (  # noqa: E402
    bgr_to_tensor, psnr, read_bgr, resize, ssim, tensor_to_bgr,
)


class HighFrequencyWaveletLoss(nn.Module):
    """
    L1 on the high-frequency Haar sub-bands (LH, HL, HH).

    Targets edges and texture directly, replacing a separate gradient + laplacian
    pair. pixel_unshuffle plus a 1x1 grouped conv makes it nearly free.
    """

    def __init__(self, c=3):
        super().__init__()
        self.c = c
        w = torch.tensor([[0.5, 0.5, 0.5, 0.5],
                          [0.5, 0.5, -0.5, -0.5],
                          [0.5, -0.5, 0.5, -0.5],
                          [0.5, -0.5, -0.5, 0.5]], dtype=torch.float32)
        self.register_buffer("haar", w.view(4, 4, 1, 1).repeat(c, 1, 1, 1))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        p = F.conv2d(F.pixel_unshuffle(pred, 2), self.haar, groups=self.c)
        t = F.conv2d(F.pixel_unshuffle(target, 2), self.haar, groups=self.c)
        b, _, h, w = p.shape
        p = p.view(b, self.c, 4, h, w)
        t = t.view(b, self.c, 4, h, w)
        return self.l1(p[:, :, 1:], t[:, :, 1:])       # band 0 is LL, skip it


class PerceptualLoss(nn.Module):
    """VGG19 relu3_3 feature L1. Frozen, always eval, always FP32."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)])
        for p in self.slice.parameters():
            p.requires_grad_(False)
        self.slice.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(False)      # stay frozen even when the parent switches to train()
        self.slice.eval()
        return self

    def forward(self, x, y):
        with torch.amp.autocast(x.device.type, enabled=False):
            x = ((x.float() + 1.0) / 2.0 - self.mean) / self.std
            y = ((y.float() + 1.0) / 2.0 - self.mean) / self.std
            return torch.mean(torch.abs(self.slice(x) - self.slice(y.detach())))


DEFAULT_DATA_ROOT = SAMPLE_TRAIN

# Fixed so two runs of the same config index the same triplets in the same order;
# an ablation is only comparable against a baseline that saw the same data.
SEED = 42

# train.data keys, and the pre-rename spelling each one replaced.
DATA_KEYS = {"distorted": "pro", "light": "beam", "surface": "clean"}


def _configured_data(data):
    """The three train.data directories, still accepting the pre-rename key names."""
    return {new: (data or {}).get(new, (data or {}).get(old))
            for new, old in DATA_KEYS.items()}


def _ssim_module(device):
    try:
        import pytorch_msssim
    except ImportError as e:
        raise SystemExit("training needs pytorch-msssim: pip install -e .[train]") from e
    return pytorch_msssim.SSIM(data_range=2.0, size_average=True, channel=3).to(device)


class ValidationDataset(Dataset):
    """Full, deterministic images; training's random patch sampler is not used."""

    def __init__(self, triplets, size):
        self.triplets = list(triplets)
        self.size = tuple(size)  # width, height

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        distorted, surface, light = self.triplets[index]

        def image(path):
            return bgr_to_tensor(read_bgr(path), size=self.size).squeeze(0)

        name = os.path.splitext(os.path.basename(distorted))[0]
        reference = resize(read_bgr(surface), self.size)
        target = bgr_to_tensor(reference).squeeze(0)
        return image(distorted), image(light), target, name, reference


def _check_scene_separation(train_triplets, val_triplets):
    train_ids = {surface_id(t[0]) for t in train_triplets}
    val_ids = {surface_id(t[0]) for t in val_triplets}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise SystemExit("training and validation share surfaceId(s): "
                         + ", ".join(overlap))


def _preview_names(triplets, limit):
    """Prefer one example per scene before filling the remaining preview slots."""
    if limit < -1:
        raise ValueError("validation save_images must be -1, 0, or a positive integer")
    names = [os.path.splitext(os.path.basename(t[0]))[0] for t in triplets]
    if limit == -1:
        return set(names)
    selected, seen = [], set()
    for name in names:
        sid = surface_id(name)
        if sid not in seen:
            selected.append(name)
            seen.add(sid)
    selected.extend(name for name in names if name not in selected)
    return set(selected[:limit])


def _loss_parts(restored, surface, l1, percep, wavelet, ssim_fn):
    return {"l1": l1(restored, surface),
            "perceptual": percep(restored, surface),
            "ssim": 1 - ssim_fn(restored, surface),
            "wavelet": wavelet(restored, surface)}


def _save_validation_preview(directory, name, distorted, restored, surface):
    restored_dir = os.path.join(directory, "restored")
    comparison_dir = os.path.join(directory, "comparison")
    os.makedirs(restored_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    panel = np.concatenate([distorted, restored, surface], axis=1)
    width = restored.shape[1]
    for offset, label in enumerate(("distorted", "restored", "surface")):
        cv2.putText(panel, label, (offset * width + 5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (offset * width + 5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for path, image in ((os.path.join(restored_dir, name + ".png"), restored),
                        (os.path.join(comparison_dir, name + ".jpg"), panel)):
        if not cv2.imwrite(path, image):
            raise OSError(f"could not save validation image: {path}")


@torch.inference_mode()
def validate_epoch(net, loader, device, weights, l1, percep, wavelet, ssim_fn,
                   directory, preview_names):
    """Mean per-image metrics, no weight/statistics updates, full validation set."""
    was_training = net.training
    rows = []
    net.eval()
    os.makedirs(directory, exist_ok=True)
    try:
        for distorted, light, surface, names, references in loader:
            # Validation uses batch size one so every image has equal weight.
            distorted = distorted.to(device)
            light = light.to(device)
            surface = surface.to(device)
            residual = net(torch.cat([distorted, light], dim=1))
            if not torch.isfinite(residual).all():
                raise RuntimeError(f"non-finite validation prediction: {names[0]}")
            restored = (distorted - residual).clamp(-1, 1)
            parts = _loss_parts(restored, surface, l1, percep, wavelet, ssim_fn)
            total = sum(weights.get(k, 0.0) * v for k, v in parts.items())
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite validation loss: {names[0]}")
            # Match evaluate.py's RGB->uint8 BGR conversion and image metrics.
            # Keep the original uint8 target: a tensor round-trip can truncate a
            # pixel by one and would disagree with evaluate.py's ground truth.
            reference = references[0].numpy()
            result = tensor_to_bgr(restored)
            row = {"name_id": names[0], "surface_id": surface_id(names[0]),
                   "val_loss": float(total),
                   **{f"val_{k}_loss": float(v) for k, v in parts.items()},
                   "val_psnr": psnr(reference, result),
                   "val_ssim": ssim(reference, result)}
            rows.append(row)
            if names[0] in preview_names:
                _save_validation_preview(directory, names[0],
                                         tensor_to_bgr(distorted), result, reference)
    finally:
        net.train(was_training)
    if not rows:
        raise RuntimeError("validation dataset is empty")
    with open(os.path.join(directory, "metrics.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {key: float(np.mean([row[key] for row in rows]))
            for key in rows[0] if key.startswith("val_")}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None,
                   help=f"folder with distorted/ light/ surface/, or the research "
                        f"layout (default: the three directories under train.data in "
                        f"configs/restoration.yaml, else {DEFAULT_DATA_ROOT}). Passing "
                        f"this overrides train.data")
    p.add_argument("--gt", default=None,
                   help="where surface/ lives when it is not under --data-root")
    p.add_argument("--val-root", default=None,
                   help="complete validation split (default: train.validation.root)")
    p.add_argument("--val-gt", default=None,
                   help="validation surface targets when stored outside --val-root")
    p.add_argument("--val-save-images", type=int, default=None,
                   help="previews per epoch: -1 all, 0 none; default from validation config")
    p.add_argument("--out", default="runs", help="parent folder for run directories")
    p.add_argument("--epochs", type=int, default=None,
                   help="default: train.epochs in configs/restoration.yaml")
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: train.batch_size in configs/restoration.yaml")
    p.add_argument("--lr", type=float, default=None,
                   help="default: train.lr in configs/restoration.yaml")
    p.add_argument("--sample", type=int, default=0, help="cap the triplet count")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if present)")
    return p


def seed_everything(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _surface_root(data_root, gt_root):
    """
    Directory holding the surface targets.

    They sit either under data_root (research and flat layouts) or under
    gt_root/surface (the bundled sample set). The pre-rename `clean/` name is still
    accepted so an already-collected session keeps training.
    """
    candidates = [os.path.join(data_root, RESEARCH_DIRS["surface"]),
                  os.path.join(data_root, "surface"),
                  os.path.join(gt_root or "", "surface"),
                  os.path.join(data_root, "clean"),
                  os.path.join(gt_root or "", "clean")]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise SystemExit(
        "no surface/ targets found. training needs the un-projected screen images.\n"
        + "".join(f"    looked in {p}\n" for p in candidates))


def _optimizer_step(net, opt, scaler):
    if scaler is not None:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed_everything()

    rest_cfg = load_config("restoration")
    tcfg = rest_cfg.get("train", {})
    device = resolve_device(args.device)
    epochs = int(pick(args.epochs, tcfg, "epochs", default=30))
    batch_size = int(pick(args.batch_size, tcfg, "batch_size", default=4))
    accum = int(pick(None, tcfg, "accum_steps", default=4))
    lr = float(pick(args.lr, tcfg, "lr", default=2e-4))
    workers = 4 if batch_size <= 4 else 8
    weights = {k: float(v) for k, v in (tcfg.get("loss") or {}).items()}
    weights.setdefault("l1", 1.0)
    vcfg = tcfg.get("validation") or {}
    val_root = resolve_path(args.val_root or vcfg.get("root") or SAMPLE_EVAL)
    val_gt = resolve_path(args.val_gt or vcfg.get("gt"))
    val_save_images = int(pick(args.val_save_images, vcfg, "save_images", default=10))
    if val_save_images < -1:
        raise SystemExit("--val-save-images must be -1, 0, or a positive integer")

    # A resumed checkpoint carries its own architecture, and that wins over the
    # `ablation:` block - continuing a run must not silently change the network.
    net = None
    if args.resume:
        # A mistyped --resume is a typo, not a bug, and every other entry point
        # answers one with a line rather than a traceback.
        try:
            net, cfg, meta = load_checkpoint(resolve_path(args.resume), device=device)
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(f"could not resume from {args.resume}\n    {e}") from e
        print(f"resuming from {args.resume} (cfg via {meta['cfg_source']})")
    else:
        try:
            cfg = config_from_settings(rest_cfg)
        except (ValueError, TypeError) as e:
            raise SystemExit(f"invalid restoration model config: {e}") from e

    tag = cfg.tag()
    run = os.path.join(resolve_path(args.out) or args.out,
                       f"{datetime.now():%m%d_%H%M}_{epochs}ep_{tag}")
    os.makedirs(run, exist_ok=True)

    print("=" * 74)
    print(f"train {architecture_name(cfg)} [{tag}] | {epochs} epochs | device={device}")
    print(f"    {cfg.describe()}")
    print(f"    batch {batch_size} x accum {accum} | lr {lr} | workers {workers}")
    print(f"    loss weights {weights}")
    print(f"    run dir {run}")
    print("=" * 74)

    # train.data names the three directories separately, because captured sets are
    # date-partitioned and their light frames live somewhere else entirely. An
    # explicit --data-root replaces that: the config ships with all three filled in,
    # so without this the flag could never take effect.
    configured = _configured_data(tcfg.get("data"))
    if not args.data_root and any(configured.values()):
        data_root = configured
        triplets = index_triplets(sample=args.sample, seed=SEED, **configured)
    else:
        root = args.data_root or DEFAULT_DATA_ROOT
        data_root = resolve_path(root) or root
        surface_root = _surface_root(data_root, resolve_path(args.gt))
        triplets = index_triplets(data_root, surface_root=surface_root,
                                  sample=args.sample, seed=SEED)
    patch = tuple(pick(None, tcfg, "patch_size", default=[180, 320]))
    big = tuple(pick(None, tcfg, "resize_to", default=[360, 640]))
    dataset = TripletPatchDataset(triplets, small_size=patch, big_size=big)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=(device == "cuda"))

    val_surface = _surface_root(val_root, val_gt)
    val_triplets = index_triplets(val_root, surface_root=val_surface)
    _check_scene_separation(triplets, val_triplets)
    val_size = tuple(pick(None, rest_cfg, "model", "input_size", default=[640, 360]))
    val_dataset = ValidationDataset(val_triplets, val_size)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    previews = _preview_names(val_triplets, val_save_images)
    print(f"    validation: {len(val_triplets)} images at {val_size[0]}x{val_size[1]}"
          f" from {val_root}; {len(previews)} previews/epoch")
    print("    best checkpoint: highest mean validation PSNR (dB)")

    if net is None:
        net = build_network(cfg).to(device)
    print(f"trainable params: {count_parameters(net):,}")

    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)

    l1 = nn.L1Loss()
    percep = PerceptualLoss().to(device)
    wavelet = HighFrequencyWaveletLoss(3).to(device)
    ssim_fn = _ssim_module(device)

    # AMP follows the device: it is a speed knob with no effect on CPU, and
    # nothing in a run needs it turned off selectively.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device) if use_amp else None
    history, best, best_epoch = [], -float("inf"), None

    for epoch in range(epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        ep = {k: [] for k in ("total", "l1", "perceptual", "ssim", "wavelet", "residual")}
        steps = 0

        loop = loader
        try:
            from tqdm import tqdm
            loop = tqdm(loader, desc=f"[{tag}][{epoch + 1}/{epochs}]")
        except ImportError:
            pass

        for distorted, light, surface in loop:
            distorted = distorted.to(device, non_blocking=True)
            light = light.to(device, non_blocking=True)
            surface = surface.to(device, non_blocking=True)

            ctx = (torch.amp.autocast(device_type="cuda") if use_amp
                   else torch.enable_grad())
            with ctx:
                residual = net(torch.cat([distorted, light], dim=1))
                restored = (distorted - residual).clamp(-1, 1)
                parts = _loss_parts(restored, surface, l1, percep, wavelet, ssim_fn)
                total = sum(weights.get(k, 0.0) * v for k, v in parts.items())

            scaled = total / accum
            if use_amp:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            steps += 1
            if steps % accum == 0:
                _optimizer_step(net, opt, scaler)

            ep["total"].append(float(total.detach()))
            for k, v in parts.items():
                ep[k].append(float(v.detach()))
            ep["residual"].append(float(residual.detach().abs().mean()))
            if hasattr(loop, "set_postfix"):
                loop.set_postfix({k: f"{np.mean(ep[k]):.3f}"
                                  for k in ("total", "l1", "perceptual")})

        if steps % accum:               # flush a partial accumulation window
            _optimizer_step(net, opt, scaler)

        row = {"epoch": epoch + 1, **{k: round(float(np.mean(v)), 5)
                                      for k, v in ep.items()}}
        validation = validate_epoch(
            net, val_loader, device, weights, l1, percep, wavelet, ssim_fn,
            os.path.join(run, "validation", f"epoch_{epoch + 1:03d}"), previews)
        row.update({k: round(v, 5) for k, v in validation.items()})
        history.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

        score = validation["val_psnr"]
        if math.isnan(score):
            raise RuntimeError("validation PSNR is NaN; cannot select a checkpoint")
        improved = best_epoch is None or score > best
        if improved:
            best, best_epoch = score, epoch + 1
        extra = dict(epoch=epoch + 1, epochs=epochs, loss=row["total"],
                     loss_weights=weights, batch_size=batch_size, accum_steps=accum,
                     lr=lr, data_root=data_root, triplets=len(triplets), seed=SEED,
                     validation_root=val_root, validation_surface=val_surface,
                     validation_triplets=len(val_triplets), validation_size=val_size,
                     selection_metric="val_psnr", selection_mode="max",
                     best_val_psnr=best, best_epoch=best_epoch, **validation)
        save_checkpoint(os.path.join(run, f"epoch_{epoch + 1}.pt"), net, cfg, **extra)
        if improved:
            save_checkpoint(os.path.join(run, f"restorer_{tag}_best.pt"), net, cfg,
                            **extra)
            print(f"  new best validation PSNR: {best:.5f} dB (epoch {best_epoch})")

        _write_log(history, run)
        sched.step()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\ndone. best validation PSNR {best:.5f} dB at epoch {best_epoch}\n  -> {run}")
    print(f"  use it:  set model.backend to {architecture_name(cfg)} and "
          f"model.weights in configs/restoration.yaml to\n"
          f"           {os.path.join(run, f'restorer_{tag}_best.pt')}\n"
          f"           then run: python demo.py")
    return 0


def _write_log(history, run):
    with open(os.path.join(run, "loss_log.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5))
        xs = [r["epoch"] for r in history]
        a1.plot(xs, [r["total"] for r in history], label="total", linewidth=2)
        a1.plot(xs, [r["val_loss"] for r in history], label="validation total",
                linewidth=2)
        for k in ("l1", "perceptual", "ssim"):
            a1.plot(xs, [r[k] for r in history], label=k, linestyle=":")
        a1.set_xlabel("epoch")
        a1.legend()
        a1.grid(True)
        a1.set_title("losses")
        a2.plot(xs, [r["wavelet"] for r in history], label="wavelet HF", color="purple")
        a2.plot(xs, [r["residual"] for r in history], label="mean |residual|",
                color="green")
        a2.set_xlabel("epoch")
        a2.legend()
        a2.grid(True)
        a2.set_title("auxiliary")
        a3.plot(xs, [r["val_psnr"] for r in history], label="validation PSNR",
                color="tab:blue")
        a3.set_xlabel("epoch")
        a3.set_ylabel("PSNR (dB)", color="tab:blue")
        quality = a3.twinx()
        quality.plot(xs, [r["val_ssim"] for r in history], label="validation SSIM",
                     color="tab:orange")
        quality.set_ylabel("SSIM", color="tab:orange")
        a3.grid(True)
        a3.set_title("validation quality (best = maximum PSNR)")
        fig.tight_layout()
        fig.savefig(os.path.join(run, "loss_plots.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    raise SystemExit(main())

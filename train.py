#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune the restoration network on projector-distortion triplets.

    python train.py --epochs 30
    python train.py --data-root data/sample_input --epochs 5 --batch-size 2
    python train.py --no-ca                       # ablate channel attention
    python train.py --data-root /path/to/full/dataset --epochs 30

Loss: L1 + VGG perceptual + (1 - SSIM) + high-frequency Haar, weights from
configs/restoration.yaml (tuned by an Optuna sweep on this dataset).

The network predicts the residual, so the supervised quantity is
`(pro - residual).clamp(-1, 1)` against the clean image. Checkpoints embed their
RestorationConfig, so demo.py reconstructs the right architecture with no extra flags.

Needs the training extra:  pip install -e .[train]
"""

import argparse
import gc
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from projector_distortion.cli import add_common_args, resolve_device  # noqa: E402
from projector_distortion.config import load_config, pick, resolve_path  # noqa: E402
from projector_distortion.data import (  # noqa: E402
    RESEARCH_DIRS, TripletPatchDataset, index_triplets,
)
from projector_distortion.models.restoration import (  # noqa: E402
    add_ablation_args, build_network, config_from_args, count_parameters,
    load_checkpoint, save_checkpoint,
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


def _ssim_module(device):
    try:
        import pytorch_msssim
    except ImportError as e:
        raise SystemExit("training needs pytorch-msssim: pip install -e .[train]") from e
    return pytorch_msssim.SSIM(data_range=2.0, size_average=True, channel=3).to(device)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/sample_input",
                   help="folder with pro/ beam/ clean/ (or the research layout)")
    p.add_argument("--gt", default="data/sample_gt",
                   help="where clean/ lives when it is not under --data-root")
    p.add_argument("--out", default="runs", help="parent folder for run directories")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sample", type=int, default=0, help="cap the triplet count")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--save-every", type=int, default=1, help="epochs between saves")
    add_common_args(p)
    add_ablation_args(p, capacity=True)
    return p


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _clean_root(data_root, gt_root):
    """
    Directory holding the clean targets.

    They sit either under data_root (research and flat layouts) or under
    gt_root/clean (the bundled sample set).
    """
    candidates = [os.path.join(data_root, RESEARCH_DIRS["clean"]),
                  os.path.join(data_root, "clean"),
                  os.path.join(gt_root or "", "clean")]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise SystemExit(
        "no clean/ targets found. training needs the un-projected screen images.\n"
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
    seed_everything(args.seed)

    rest_cfg = load_config("restoration", args.restoration_config)
    tcfg = rest_cfg.get("train", {})
    device = resolve_device(args.device)
    epochs = int(pick(args.epochs, tcfg, "epochs", default=30))
    batch_size = int(pick(args.batch_size, tcfg, "batch_size", default=4))
    accum = int(pick(args.accum_steps, tcfg, "accum_steps", default=4))
    lr = float(pick(args.lr, tcfg, "lr", default=2e-4))
    workers = args.num_workers if args.num_workers is not None else (
        4 if batch_size <= 4 else 8)
    weights = {k: float(v) for k, v in (tcfg.get("loss") or {}).items()}
    weights.setdefault("l1", 1.0)

    # A resumed checkpoint carries its own architecture, and that wins over --no-*.
    net = None
    cfg = config_from_args(args)
    if args.resume:
        net, cfg, meta = load_checkpoint(resolve_path(args.resume), device=device)
        print(f"resuming from {args.resume} (cfg via {meta['cfg_source']})")

    tag = cfg.tag()
    run = os.path.join(resolve_path(args.out) or args.out,
                       f"{datetime.now():%m%d_%H%M}_{epochs}ep_{tag}")
    os.makedirs(run, exist_ok=True)

    print("=" * 74)
    print(f"train restormer_like [{tag}] | {epochs} epochs | device={device}")
    print(f"    {cfg.describe()}")
    print(f"    batch {batch_size} x accum {accum} | lr {lr} | workers {workers}")
    print(f"    loss weights {weights}")
    print(f"    run dir {run}")
    print("=" * 74)

    data_root = resolve_path(args.data_root) or args.data_root
    clean_root = _clean_root(data_root, resolve_path(args.gt))
    triplets = index_triplets(data_root, clean_root=clean_root,
                              sample=args.sample, seed=args.seed)
    patch = tuple(pick(None, tcfg, "patch_size", default=[180, 320]))
    big = tuple(pick(None, tcfg, "resize_to", default=[360, 640]))
    dataset = TripletPatchDataset(triplets, small_size=patch, big_size=big)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=(device == "cuda"))

    if net is None:
        net = build_network(cfg).to(device)
    print(f"trainable params: {count_parameters(net):,}")

    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)

    l1 = nn.L1Loss()
    percep = PerceptualLoss().to(device)
    wavelet = HighFrequencyWaveletLoss(3).to(device)
    ssim_fn = _ssim_module(device)

    use_amp = (not args.no_amp) and device == "cuda"
    scaler = torch.amp.GradScaler(device) if use_amp else None
    history, best = [], float("inf")

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

        for pro, beam, clean in loop:
            pro = pro.to(device, non_blocking=True)
            beam = beam.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            ctx = (torch.amp.autocast(device_type="cuda") if use_amp
                   else torch.enable_grad())
            with ctx:
                residual = net(torch.cat([pro, beam], dim=1))
                restored = (pro - residual).clamp(-1, 1)
                parts = {
                    "l1": l1(restored, clean),
                    "perceptual": percep(restored, clean),
                    "ssim": 1 - ssim_fn(restored, clean),
                    "wavelet": wavelet(restored, clean),
                }
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
        history.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

        extra = dict(epoch=epoch + 1, epochs=epochs, loss=row["total"],
                     loss_weights=weights, batch_size=batch_size, accum_steps=accum,
                     lr=lr, data_root=data_root, triplets=len(triplets))
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(os.path.join(run, f"epoch_{epoch + 1}.pt"), net, cfg, **extra)
        if row["total"] < best:
            best = row["total"]
            save_checkpoint(os.path.join(run, f"restorer_{tag}_best.pt"), net, cfg,
                            **extra)
            print(f"  new best: {best:.5f}")

        _write_log(history, run)
        sched.step()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\ndone. best loss {best:.5f}\n  -> {run}")
    print(f"  use it:  python demo.py --restorer-weights "
          f"{os.path.join(run, f'restorer_{tag}_best.pt')}")
    return 0


def _write_log(history, run):
    import csv
    with open(os.path.join(run, "loss_log.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        xs = [r["epoch"] for r in history]
        a1.plot(xs, [r["total"] for r in history], label="total", linewidth=2)
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
        fig.tight_layout()
        fig.savefig(os.path.join(run, "loss_plots.png"))
        plt.close(fig)
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    raise SystemExit(main())

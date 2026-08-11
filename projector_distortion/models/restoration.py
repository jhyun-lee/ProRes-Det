"""
Restoration network, its config, and the BaseRestorer wrapper.

A 3-level U-Net of NAFSEBlock (LayerNorm2d -> NAFBlock -> CALayer):

    input  (B, 6, H, W) = cat([distorted, light]) in [-1, 1]
    output (B, 3, H, W) = residual
    restored = (distorted - residual).clamp(-1, 1)

There is no MDTA attention and no depthwise-gated FFN here, so this is NAFNet blocks
plus squeeze-excite channel attention in a U-Net, not Restormer - which is what the
`naf_se_unet` name says. It is fully convolutional, which is why the 180x320 training
patch and the 640x360 inference size behave identically.

Checkpoints written before the rename carry `arch: "restormer_like"`. Nothing reads
that field to rebuild the network - the architecture comes from the embedded `cfg` -
so an old checkpoint still loads unchanged.
"""

import os
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.image import bgr_to_tensor, residual_to_bgr, tensor_to_bgr
from .base import BaseRestorer, register_restorer

CHECKPOINT_FORMAT = 2
ARCH_NAME = "naf_se_unet"
LEGACY_ARCH_NAME = "restormer_like"


@dataclass
class RestorationConfig:
    """
    Structural toggles + capacity; the defaults reproduce the reference model.

    Turning a `use_*` flag off and retraining measures that component's
    contribution. The config travels inside the checkpoint, so inference never has
    to guess which variant produced the weights. TOGGLE_HELP describes each flag.
    """

    use_prenorm: bool = True
    use_naf_norm: bool = True
    use_simple_gate: bool = True
    use_naf_scale: bool = True
    use_ca: bool = True

    use_skip1: bool = True
    use_skip2: bool = True
    use_skip3: bool = True
    use_bottleneck: bool = True
    use_tanh: bool = True

    base_dim: int = 48
    enc_depth: Tuple[int, int, int] = (2, 2, 3)
    dec_depth: Tuple[int, int, int] = (2, 2, 2)
    bottleneck_depth: int = 2
    dw_expand: int = 2
    ffn_expand: int = 2
    ca_reduction: int = 16
    in_ch: int = 6
    out_ch: int = 3

    def __post_init__(self):
        # torch.save/load and YAML round-trips turn tuples into lists
        self.enc_depth = tuple(int(v) for v in self.enc_depth)
        self.dec_depth = tuple(int(v) for v in self.dec_depth)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RestorationConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def is_default(self) -> bool:
        return self == RestorationConfig()

    def tag(self) -> str:
        """'FULL' when nothing is ablated, else e.g. 'NoCA-NoSkip3'."""
        off = [t for a, _, t in TOGGLES if not getattr(self, a)]
        return "-".join(off) if off else "FULL"

    def describe(self) -> str:
        on = [a for a, _, _ in TOGGLES if getattr(self, a)]
        off = [a for a, _, _ in TOGGLES if not getattr(self, a)]
        return (f"base_dim={self.base_dim} enc={self.enc_depth} dec={self.dec_depth} "
                f"bott={self.bottleneck_depth}\n"
                f"    ON  : {', '.join(on) or '(none)'}\n"
                f"    OFF : {', '.join(off) or '(none)'}")


# (config attribute, CLI flag, short tag)
TOGGLES = [
    ("use_prenorm", "no-prenorm", "NoPre"),
    ("use_naf_norm", "no-naf-norm", "NoNorm"),
    ("use_simple_gate", "no-simple-gate", "NoGate"),
    ("use_naf_scale", "no-naf-scale", "NoScale"),
    ("use_ca", "no-ca", "NoCA"),
    ("use_skip1", "no-skip1", "NoSkip1"),
    ("use_skip2", "no-skip2", "NoSkip2"),
    ("use_skip3", "no-skip3", "NoSkip3"),
    ("use_bottleneck", "no-bottleneck", "NoBott"),
    ("use_tanh", "no-tanh", "NoTanh"),
]

TOGGLE_HELP = {
    "no-prenorm": "disable the pre-LayerNorm of NAFSEBlock",
    "no-naf-norm": "disable LayerNorm2d inside NAFBlock",
    "no-simple-gate": "replace SimpleGate (x1*x2) with GELU (no channel halving)",
    "no-naf-scale": "disable learnable residual scales beta / gamma",
    "no-ca": "disable channel attention; the block becomes a plain NAFBlock",
    "no-skip1": "disable U-Net skip enc1 -> dec1 (full resolution)",
    "no-skip2": "disable U-Net skip enc2 -> dec2 (1/2 resolution)",
    "no-skip3": "disable U-Net skip enc3 -> dec3 (1/4 resolution)",
    "no-bottleneck": "replace the 1/8-resolution bottleneck with Identity",
    "no-tanh": "disable the output tanh (unbounded residual)",
}


def add_ablation_args(parser):
    """
    Register the --no-* toggles and the capacity overrides. Training only.

    Inference has no use for them: every checkpoint carries the config it was trained
    with, so demo.py and evaluate.py rebuild the right architecture with no flags.
    """
    g = parser.add_argument_group("restoration ablation")
    for _, flag, _ in TOGGLES:
        g.add_argument(f"--{flag}", action="store_true", help=TOGGLE_HELP[flag])
    c = parser.add_argument_group("restoration capacity")
    c.add_argument("--base-dim", type=int, default=48)
    c.add_argument("--enc-depth", default="2,2,3", help="3 comma separated ints")
    c.add_argument("--dec-depth", default="2,2,2", help="3 comma separated ints")
    c.add_argument("--bottleneck-depth", type=int, default=2)
    c.add_argument("--dw-expand", type=int, default=2)
    c.add_argument("--ffn-expand", type=int, default=2)
    c.add_argument("--ca-reduction", type=int, default=16)
    return parser


def _parse_depth(text, name):
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    if len(parts) != 3:
        raise ValueError(f"--{name} needs exactly 3 comma separated ints, got {text!r}")
    return tuple(int(p) for p in parts)


def config_from_args(args) -> RestorationConfig:
    kw = {attr: not getattr(args, flag.replace("-", "_"), False)
          for attr, flag, _ in TOGGLES}
    for name in ("base_dim", "bottleneck_depth", "dw_expand", "ffn_expand",
                 "ca_reduction"):
        if getattr(args, name, None) is not None:
            kw[name] = getattr(args, name)
    for name in ("enc_depth", "dec_depth"):
        if getattr(args, name, None) is not None:
            kw[name] = _parse_depth(getattr(args, name), name.replace("_", "-"))
    return RestorationConfig(**kw)


# Attribute names below must not change - checkpoints key off them.

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies, so the output has C/2 channels."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CALayer(nn.Module):
    """Squeeze-and-excite channel attention."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(self.avg(x))


class NAFBlock(nn.Module):
    """NAFNet-style block. Without SimpleGate, conv3/conv5 widen to match GELU."""

    def __init__(self, c, dw_expand=2, ffn_expand=2, cfg: RestorationConfig = None):
        super().__init__()
        cfg = cfg or RestorationConfig()
        dwc = c * dw_expand
        gate_div = 2 if cfg.use_simple_gate else 1

        self.norm1 = LayerNorm2d(c) if cfg.use_naf_norm else nn.Identity()
        self.conv1 = nn.Conv2d(c, dwc, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dwc, dwc, 3, 1, 1, groups=dwc, bias=True)
        self.sg = SimpleGate() if cfg.use_simple_gate else nn.GELU()
        self.conv3 = nn.Conv2d(dwc // gate_div, c, 1, 1, 0, bias=True)

        self.norm2 = LayerNorm2d(c) if cfg.use_naf_norm else nn.Identity()
        ffc = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffc, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffc // gate_div, c, 1, 1, 0, bias=True)

        if cfg.use_naf_scale:
            self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        else:
            self.register_parameter("beta", None)
            self.register_parameter("gamma", None)

    def forward(self, x):
        y = self.conv3(self.sg(self.conv2(self.conv1(self.norm1(x)))))
        x = x + (y * self.beta if self.beta is not None else y)
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + (y * self.gamma if self.gamma is not None else y)


class NAFSEBlock(nn.Module):
    """LayerNorm -> NAFBlock -> squeeze-excite channel attention."""

    def __init__(self, c, cfg: RestorationConfig = None):
        super().__init__()
        cfg = cfg or RestorationConfig()
        self.pre = LayerNorm2d(c) if cfg.use_prenorm else nn.Identity()
        self.body = NAFBlock(c, cfg.dw_expand, cfg.ffn_expand, cfg=cfg)
        self.ca = CALayer(c, cfg.ca_reduction) if cfg.use_ca else nn.Identity()

    def forward(self, x):
        return self.ca(x + self.body(self.pre(x)))


class RestorationNet(nn.Module):
    """3-level encoder / bottleneck / 3-level decoder over NAFSEBlock."""

    def __init__(self, cfg: RestorationConfig = None):
        super().__init__()
        cfg = cfg or RestorationConfig()
        self.cfg = cfg
        bd, ed, dd = cfg.base_dim, cfg.enc_depth, cfg.dec_depth

        def stage(channels, depth):
            return nn.Sequential(*[NAFSEBlock(channels, cfg)
                                   for _ in range(depth)])

        self.stem = nn.Conv2d(cfg.in_ch, bd, 3, 1, 1)

        self.enc1 = stage(bd, ed[0])
        self.down1 = nn.Conv2d(bd, bd * 2, 2, 2)
        self.enc2 = stage(bd * 2, ed[1])
        self.down2 = nn.Conv2d(bd * 2, bd * 4, 2, 2)
        self.enc3 = stage(bd * 4, ed[2])
        self.down3 = nn.Conv2d(bd * 4, bd * 8, 2, 2)

        self.bottleneck = (stage(bd * 8, cfg.bottleneck_depth)
                           if cfg.use_bottleneck else nn.Identity())

        self.up3 = nn.ConvTranspose2d(bd * 8, bd * 4, 2, 2)
        self.j3 = nn.Conv2d(bd * 4 * (2 if cfg.use_skip3 else 1), bd * 4, 1)
        self.dec3 = stage(bd * 4, dd[0])

        self.up2 = nn.ConvTranspose2d(bd * 4, bd * 2, 2, 2)
        self.j2 = nn.Conv2d(bd * 2 * (2 if cfg.use_skip2 else 1), bd * 2, 1)
        self.dec2 = stage(bd * 2, dd[1])

        self.up1 = nn.ConvTranspose2d(bd * 2, bd, 2, 2)
        self.j1 = nn.Conv2d(bd * (2 if cfg.use_skip1 else 1), bd, 1)
        self.dec1 = stage(bd, dd[2])

        self.out = nn.Sequential(
            nn.Conv2d(bd, bd // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(bd // 2, cfg.out_ch, 3, 1, 1),
        )
        self.tanh = nn.Tanh() if cfg.use_tanh else nn.Identity()

    @staticmethod
    def _merge(up, skip, use_skip):
        """Align to the encoder feature's size, then concat when the skip is on."""
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear",
                               align_corners=False)
        return torch.cat([up, skip], dim=1) if use_skip else up

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.bottleneck(self.down3(s3))

        d3 = self.dec3(self.j3(self._merge(self.up3(b), s3, self.cfg.use_skip3)))
        d2 = self.dec2(self.j2(self._merge(self.up2(d3), s2, self.cfg.use_skip2)))
        d1 = self.dec1(self.j1(self._merge(self.up1(d2), s1, self.cfg.use_skip1)))
        return self.tanh(self.out(d1))


def build_network(cfg: RestorationConfig = None) -> RestorationNet:
    return RestorationNet(cfg or RestorationConfig())


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model: nn.Module, cfg: RestorationConfig, **extra):
    """Write weights with the config embedded, so loading never has to guess."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    blob = {"format": CHECKPOINT_FORMAT, "arch": ARCH_NAME,
            "cfg": cfg.to_dict(), "state_dict": model.state_dict()}
    blob.update(extra)
    torch.save(blob, path)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


def _strip_prefixes(state_dict):
    for prefix in ("module.", "_orig_mod."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path, device="cpu", cfg: Optional[RestorationConfig] = None,
                    strict=True):
    """
    Returns (network, cfg, meta).

    An explicit `cfg` wins; otherwise the config embedded in the checkpoint is used.
    A bare state_dict (the legacy format) falls back to the defaults.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"restoration checkpoint not found: {path}")

    blob = _torch_load(path, device)
    if isinstance(blob, dict) and "state_dict" in blob:
        state_dict = blob["state_dict"]
        embedded = blob.get("cfg")
        used = cfg or (RestorationConfig.from_dict(embedded) if embedded
                       else RestorationConfig())
        meta = {k: v for k, v in blob.items() if k != "state_dict"}
        meta["cfg_source"] = "embedded" if embedded else "defaults"
    else:
        state_dict = blob
        used = cfg or RestorationConfig()
        meta = {"format": 1, "arch": ARCH_NAME, "cfg_source": "legacy-raw"}

    model = build_network(used).to(device)
    try:
        missing, unexpected = model.load_state_dict(_strip_prefixes(state_dict),
                                                    strict=strict)
    except RuntimeError as e:
        raise RuntimeError(_mismatch_help(path, used, meta["cfg_source"])) from e
    if missing or unexpected:
        meta["missing_keys"], meta["unexpected_keys"] = list(missing), list(unexpected)
    return model, used, meta


def _mismatch_help(path, cfg: RestorationConfig, cfg_source: str) -> str:
    """
    Turn torch's wall of missing keys into the one thing that fixes it.

    The common cause is a checkpoint that carries no config of its own: it gets
    rebuilt from the defaults, so an ablated variant never matches. Nothing in the
    key list says that, and the ablation flags are training-only, so the answer is
    the `ablation:` block of a config file.
    """
    lines = [f"{os.path.basename(str(path))} does not fit the architecture it was "
             f"loaded into (tag {cfg.tag()}, config from {cfg_source})."]
    if cfg_source in ("legacy-raw", "defaults"):
        lines.append(
            "    This checkpoint carries no config of its own, so it was rebuilt from\n"
            "    the defaults. If it is an ablated variant, describe it in the\n"
            "    'ablation:' block of a YAML and pass --restoration-config <file>.")
    else:
        lines.append("    The embedded config does not match the stored weights, so "
                     "the file is likely corrupt or hand-edited.")
    return "\n".join(lines)


@register_restorer(ARCH_NAME)
class NAFSEUNetRestorer(BaseRestorer):
    """Owns normalisation and the numpy <-> tensor round trip."""

    name = ARCH_NAME

    def __init__(self, weights, device="cpu", cfg: RestorationConfig = None,
                 input_size=(640, 360), **_):
        self.device = device
        self.input_size = tuple(input_size)
        self.net, self.cfg, self.meta = load_checkpoint(weights, device=device, cfg=cfg)
        self.net.eval()
        self.weights = weights
        self.params = count_parameters(self.net)

    @torch.no_grad()
    def _forward(self, distorted_bgr, light_bgr):
        distorted = bgr_to_tensor(distorted_bgr, self.device, self.input_size)
        light = bgr_to_tensor(light_bgr, self.device, self.input_size)
        inp = torch.cat([distorted, light], dim=1)
        residual = self.net(inp).float()
        restored = (distorted - residual).clamp(-1, 1)
        return restored, residual

    def restore(self, distorted_bgr, light_bgr):
        restored, residual = self._forward(distorted_bgr, light_bgr)
        return (tensor_to_bgr(restored, self.input_size),
                residual_to_bgr(residual, self.input_size))

    def restore_full(self, distorted_bgr, light_bgr):
        """restore() plus mean |residual|, from one forward pass rather than two."""
        restored, residual = self._forward(distorted_bgr, light_bgr)
        return (tensor_to_bgr(restored, self.input_size),
                residual_to_bgr(residual, self.input_size),
                float(residual.abs().mean()))

    def describe(self):
        return (f"{self.name} [{self.cfg.tag()}] {os.path.basename(str(self.weights))} "
                f"({self.params:,} params, cfg from {self.meta.get('cfg_source')})")

    def info(self) -> Dict:
        return {"name": self.name, "weights": os.path.abspath(str(self.weights)),
                "params": self.params, "cfg": self.cfg.to_dict(),
                "cfg_source": self.meta.get("cfg_source"),
                "input_size": list(self.input_size), "device": str(self.device)}


def build_restorer(kind=ARCH_NAME, weights=None, device="cpu",
                   cfg: RestorationConfig = None,
                   input_size=(640, 360)) -> BaseRestorer:
    """
    Instantiate a registered restoration backend, the mirror of `build_detector`.

    `cfg` is this architecture's ablation config; a backend that has no use for it
    absorbs it through `**_`, exactly as `ssd` does with the yolo-only `imgsz`.
    """
    from .base import get_restorer_class

    cls = get_restorer_class(kind)
    if not weights:
        raise ValueError(f"restorer '{kind}' needs a weights path")
    return cls(weights, device=device, cfg=cfg, input_size=input_size)

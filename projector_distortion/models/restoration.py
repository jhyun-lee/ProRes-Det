# projector_distortion/models/restoration.py
"""
Restoration network, its config, and the BaseRestorer wrapper.

Architecture is a 3-level U-Net of RestormerLikeBlock, ported from
`Another_ModelDefine.build_restor_like()` in the original research repo. With the
default config the module graph, parameter names and tensor shapes are identical to
that reference, so checkpoints trained there load unchanged.

    input  (B, 6, H, W) = cat([pro, beam]) in [-1, 1]
    output (B, 3, H, W) = residual
    restored = (pro - residual).clamp(-1, 1)

NOTE ON THE NAME. `RestormerLikeBlock` is LayerNorm2d -> NAFBlock -> CALayer. It has
no MDTA attention, so it is *not* Restormer; the upstream file calls these
"lightweight proxies ... baselines for measurement only". Functionally it is closer to
NAFNet + squeeze-excite. It is fully convolutional, which is why the 180x320 training
patch and the 640x360 inference size behave identically.

Every structural piece can be switched off through `RestorationConfig`, which is how
the ablation study was run.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.image import bgr_to_tensor, residual_to_bgr, tensor_to_bgr
from .base import BaseRestorer

CHECKPOINT_FORMAT = 2
ARCH_NAME = "restormer_like"


# =============================================================================
# Config
# =============================================================================

@dataclass
class RestorationConfig:
    """
    Structural toggles + capacity. Defaults reproduce the reference model exactly.

    The `use_*` flags exist so a component's contribution can be measured by turning
    it off and retraining; they also travel inside the checkpoint so inference never
    has to guess which variant produced the weights.
    """

    # RestormerLikeBlock internals
    use_prenorm: bool = True        # pre-LayerNorm in front of the NAF body
    use_naf_norm: bool = True       # LayerNorm2d at NAFBlock norm1 / norm2
    use_simple_gate: bool = True    # SimpleGate (x1*x2) vs plain GELU
    use_naf_scale: bool = True      # learnable residual scales beta / gamma
    use_ca: bool = True             # channel attention after the block

    # U-Net skeleton
    use_skip1: bool = True          # skip enc1 -> dec1 (full res)
    use_skip2: bool = True          # skip enc2 -> dec2 (1/2 res)
    use_skip3: bool = True          # skip enc3 -> dec3 (1/4 res)
    use_bottleneck: bool = True     # bottleneck stage at 1/8 res
    use_tanh: bool = True           # output tanh (residual bounded to [-1, 1])

    # capacity
    base_dim: int = 48
    enc_depth: Tuple[int, int, int] = (2, 2, 3)
    dec_depth: Tuple[int, int, int] = (2, 2, 2)
    bottleneck_depth: int = 2
    dw_expand: int = 2
    ffn_expand: int = 2
    ca_reduction: int = 16
    in_ch: int = 6                  # [pro | beam]
    out_ch: int = 3                 # residual

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
        """'FULL' when nothing is ablated, else 'NoCA-NoSkip3'."""
        off = [t for a, _, t in TOGGLES if not getattr(self, a)]
        return "-".join(off) if off else "FULL"

    def describe(self) -> str:
        on = [a for a, _, _ in TOGGLES if getattr(self, a)]
        off = [a for a, _, _ in TOGGLES if not getattr(self, a)]
        return (f"base_dim={self.base_dim} enc={self.enc_depth} dec={self.dec_depth} "
                f"bott={self.bottleneck_depth}\n"
                f"    ON  : {', '.join(on) or '(none)'}\n"
                f"    OFF : {', '.join(off) or '(none)'}")


#: (config attribute, CLI flag, short tag)
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
    "no-prenorm": "disable the pre-LayerNorm of RestormerLikeBlock",
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


def add_ablation_args(parser, capacity: bool = True):
    """Register --no-* toggles (and optionally the capacity overrides)."""
    g = parser.add_argument_group("restoration ablation")
    for _, flag, _ in TOGGLES:
        g.add_argument(f"--{flag}", action="store_true", help=TOGGLE_HELP[flag])
    if capacity:
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


# =============================================================================
# Blocks  (attribute names must not change - checkpoints key off them)
# =============================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies -> output has C/2 channels."""

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
    """
    NAFNet-style block.

    With use_simple_gate off, GELU keeps the channel count instead of halving it, so
    conv3 / conv5 widen to match - a different shape, hence a different checkpoint.
    """

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


class RestormerLikeBlock(nn.Module):
    """
    LayerNorm -> NAFBlock -> channel attention.

    With use_prenorm and use_ca both off this degenerates to a plain NAFBlock.
    """

    def __init__(self, c, cfg: RestorationConfig = None):
        super().__init__()
        cfg = cfg or RestorationConfig()
        self.pre = LayerNorm2d(c) if cfg.use_prenorm else nn.Identity()
        self.body = NAFBlock(c, cfg.dw_expand, cfg.ffn_expand, cfg=cfg)
        self.ca = CALayer(c, cfg.ca_reduction) if cfg.use_ca else nn.Identity()

    def forward(self, x):
        return self.ca(x + self.body(self.pre(x)))


# =============================================================================
# Network
# =============================================================================

class RestorationNet(nn.Module):
    """3-level encoder / bottleneck / 3-level decoder over RestormerLikeBlock."""

    def __init__(self, cfg: RestorationConfig = None):
        super().__init__()
        cfg = cfg or RestorationConfig()
        self.cfg = cfg
        bd, ed, dd = cfg.base_dim, cfg.enc_depth, cfg.dec_depth
        block = lambda c: RestormerLikeBlock(c, cfg)  # noqa: E731

        self.stem = nn.Conv2d(cfg.in_ch, bd, 3, 1, 1)

        self.enc1 = nn.Sequential(*[block(bd) for _ in range(ed[0])])
        self.down1 = nn.Conv2d(bd, bd * 2, 2, 2)
        self.enc2 = nn.Sequential(*[block(bd * 2) for _ in range(ed[1])])
        self.down2 = nn.Conv2d(bd * 2, bd * 4, 2, 2)
        self.enc3 = nn.Sequential(*[block(bd * 4) for _ in range(ed[2])])
        self.down3 = nn.Conv2d(bd * 4, bd * 8, 2, 2)

        self.bottleneck = (
            nn.Sequential(*[block(bd * 8) for _ in range(cfg.bottleneck_depth)])
            if cfg.use_bottleneck else nn.Identity())

        self.up3 = nn.ConvTranspose2d(bd * 8, bd * 4, 2, 2)
        self.j3 = nn.Conv2d(bd * 4 * (2 if cfg.use_skip3 else 1), bd * 4, 1)
        self.dec3 = nn.Sequential(*[block(bd * 4) for _ in range(dd[0])])

        self.up2 = nn.ConvTranspose2d(bd * 4, bd * 2, 2, 2)
        self.j2 = nn.Conv2d(bd * 2 * (2 if cfg.use_skip2 else 1), bd * 2, 1)
        self.dec2 = nn.Sequential(*[block(bd * 2) for _ in range(dd[1])])

        self.up1 = nn.ConvTranspose2d(bd * 2, bd, 2, 2)
        self.j1 = nn.Conv2d(bd * (2 if cfg.use_skip1 else 1), bd, 1)
        self.dec1 = nn.Sequential(*[block(bd) for _ in range(dd[2])])

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


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_checkpoint(path, model: nn.Module, cfg: RestorationConfig, **extra):
    """Write weights with the config embedded, so loading never has to guess."""
    import os
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

    An explicit `cfg` wins. Otherwise the config embedded in the checkpoint is used;
    for a bare state_dict (the legacy format) it falls back to the defaults, which
    reproduce the reference model.
    """
    import os
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
    missing, unexpected = model.load_state_dict(_strip_prefixes(state_dict),
                                                strict=strict)
    if missing or unexpected:
        meta["missing_keys"], meta["unexpected_keys"] = list(missing), list(unexpected)
    return model, used, meta


# =============================================================================
# BaseRestorer wrapper
# =============================================================================

class RestormerLikeRestorer(BaseRestorer):
    """
    Glue between RestorationNet and the pipeline.

    Handles normalisation, the residual convention, autocast and the numpy <-> tensor
    round trip so nothing downstream deals with tensors.
    """

    name = "restormer_like"

    def __init__(self, weights, device="cpu", cfg: RestorationConfig = None,
                 input_size=(640, 360), fp16=False):
        self.device = device
        self.input_size = tuple(input_size)
        self.fp16 = bool(fp16) and str(device).startswith("cuda")
        self.net, self.cfg, self.meta = load_checkpoint(weights, device=device, cfg=cfg)
        self.net.eval()
        self.weights = weights
        self.params = count_parameters(self.net)

    @torch.no_grad()
    def _forward(self, pro_bgr, beam_bgr):
        pro = bgr_to_tensor(pro_bgr, self.device, self.input_size)
        beam = bgr_to_tensor(beam_bgr, self.device, self.input_size)
        inp = torch.cat([pro, beam], dim=1)

        if self.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                residual = self.net(inp)
        else:
            residual = self.net(inp)

        residual = residual.float()
        restored = (pro - residual).clamp(-1, 1)
        return restored, residual

    def restore(self, pro_bgr, beam_bgr):
        restored, residual = self._forward(pro_bgr, beam_bgr)
        return (tensor_to_bgr(restored, self.input_size),
                residual_to_bgr(residual, self.input_size))

    def restore_full(self, pro_bgr, beam_bgr):
        """restore() plus mean |residual| - one forward pass, not two."""
        restored, residual = self._forward(pro_bgr, beam_bgr)
        return (tensor_to_bgr(restored, self.input_size),
                residual_to_bgr(residual, self.input_size),
                float(residual.abs().mean()))

    def describe(self):
        import os
        return (f"{self.name} [{self.cfg.tag()}] {os.path.basename(str(self.weights))} "
                f"({self.params:,} params, cfg from {self.meta.get('cfg_source')})")

    def info(self) -> Dict:
        import os
        return {"name": self.name, "weights": os.path.abspath(str(self.weights)),
                "params": self.params, "cfg": self.cfg.to_dict(),
                "cfg_source": self.meta.get("cfg_source"), "fp16": self.fp16,
                "input_size": list(self.input_size), "device": str(self.device)}


def build_restorer(weights, device="cpu", cfg: RestorationConfig = None,
                   input_size=(640, 360), fp16=False) -> RestormerLikeRestorer:
    return RestormerLikeRestorer(weights, device=device, cfg=cfg,
                                 input_size=input_size, fp16=fp16)

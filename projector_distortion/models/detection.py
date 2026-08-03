"""
Detection backends.

Both wrappers return `list[Detection]` with a 0-based class id, so the pipeline never
branches on which model is running. Label conventions differ upstream and are
normalised here: ultralytics YOLO is already 0-based, torchvision heads emit the
COCO category id (1..N, with 0 = background).

Adding a backend is a subclass plus `@register_detector("key")`.
"""

import os
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from .base import BaseDetector, Detection, register_detector

# Default class list; the YAML config and YOLO checkpoints can override it.
CLASS_NAMES = [
    "Apple", "BlueBerry", "GreenApple", "Kiwi", "Lemon", "Melon", "Orange",
    "Peach", "Pear", "Punica", "Watermelon", "Cat", "Elephant", "Giraffe",
    "Monkey", "Shark", "Snake",
]


def _load(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"detector weights not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


@register_detector("yolo")
class YoloDetector(BaseDetector):
    """ultralytics YOLO. Needs `pip install ultralytics`."""

    name = "yolo"

    def __init__(self, weights, class_names: Sequence[str] = None, conf=0.25,
                 device="cpu", imgsz=None, **_):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "the yolo backend needs ultralytics: pip install 'projector-distortion"
                "-framework[yolo]'"
            ) from e
        if not os.path.exists(weights):
            raise FileNotFoundError(f"detector weights not found: {weights}")

        self.model = YOLO(weights)
        self.weights = weights
        self.imgsz = imgsz

        # A YOLO checkpoint carries its own names; prefer them unless the caller
        # was explicit, since checkpoints differ in how many classes they cover.
        names = getattr(self.model, "names", None)
        if class_names is None and names:
            class_names = ([names[k] for k in sorted(names)]
                           if isinstance(names, dict) else list(names))
        super().__init__(class_names or CLASS_NAMES, conf, device)

    def detect(self, bgr) -> List[Detection]:
        kw = {"verbose": False, "conf": self.conf, "device": self.device}
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        out = []
        for result in self.model(bgr, **kw):
            for box in result.boxes:
                cls_id = int(box.cls)
                out.append(Detection(cls_id, self.label_of(cls_id), float(box.conf),
                                     box.xyxy[0].cpu().numpy().astype(int).tolist()))
        return out

    def describe(self):
        return f"yolo {os.path.basename(self.weights)} ({super().describe()})"

    def info(self) -> Dict:
        d = super().info()
        d.update(weights=os.path.abspath(self.weights), imgsz=self.imgsz)
        return d


@register_detector("ssd")
class SsdDetector(BaseDetector):
    """
    torchvision `ssdlite320_mobilenet_v3_large`; no dependency beyond torchvision.

    The classification head is rebuilt for the fine-tuned class count before the
    weights load, matching how the checkpoint was produced.
    """

    name = "ssd"

    def __init__(self, weights, class_names: Sequence[str] = None, conf=0.25,
                 device="cpu", **_):
        super().__init__(class_names or CLASS_NAMES, conf, device)
        self.weights = weights
        self.net = self._build(len(self.class_names) + 1)   # +1 background
        sd = _load(weights, device)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"warning: ssd weights had {len(missing)} missing / "
                  f"{len(unexpected)} unexpected keys")
        self.net.to(device).eval()

    @staticmethod
    def _build(num_classes):
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        from torchvision.models.detection.ssd import SSDHead

        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        in_channels = []
        for module in model.head.classification_head.module_list:
            conv = next((m for m in module.modules() if isinstance(m, torch.nn.Conv2d)),
                        None)
            in_channels.append(conv.in_channels if conv is not None
                               else module[0][0].in_channels)
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDHead(
            in_channels, num_anchors, num_classes).classification_head
        return model

    @torch.no_grad()
    def detect(self, bgr) -> List[Detection]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device)
        output = self.net([tensor])[0]

        out = []
        for box, label, score in zip(output["boxes"].cpu(), output["labels"].cpu(),
                                     output["scores"].cpu()):
            score = float(score)
            if score < self.conf:
                continue
            cls_id = int(label) - 1          # torchvision label is the COCO id, 1..N
            if cls_id < 0:
                continue
            out.append(Detection(cls_id, self.label_of(cls_id), score,
                                 box.numpy().astype(int).tolist()))
        return out

    def describe(self):
        return f"ssd {os.path.basename(self.weights)} ({super().describe()})"

    def info(self) -> Dict:
        d = super().info()
        d["weights"] = os.path.abspath(self.weights)
        return d


def build_detector(kind, weights=None, class_names=None, conf=0.25, device="cpu",
                   imgsz=None) -> BaseDetector:
    """Instantiate a registered backend. 'none' needs no weights."""
    from .base import get_detector_class
    cls = get_detector_class(kind)
    if kind == "none":
        return cls(class_names or CLASS_NAMES, conf, device)
    if not weights:
        raise ValueError(f"detector '{kind}' needs a weights path")
    return cls(weights, class_names=class_names, conf=conf, device=device, imgsz=imgsz)


def filter_detections(detections, min_width=20, min_height=20, min_area=500,
                      best_per_class=False):
    """
    Drop boxes below the size gate; optionally keep only the top box per class.

    best_per_class mirrors the original demo. It defaults to False because metric
    code needs every box.
    """
    kept = [d for d in detections
            if (d.box[2] - d.box[0]) >= min_width
            and (d.box[3] - d.box[1]) >= min_height
            and d.area >= min_area]
    if not best_per_class:
        return kept
    best = {}
    for d in kept:
        if d.cls_id not in best or d.conf > best[d.cls_id].conf:
            best[d.cls_id] = d
    return list(best.values())

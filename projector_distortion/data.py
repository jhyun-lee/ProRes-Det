"""
Sample discovery, ground truth loading, and the training dataset.

Filenames tie the views of one moment together (see data/README_data.md):

    pro    projected_<oriId>_<beamId>.jpg
    beam   output_video_<beamId>.jpg
    clean  Ori<oriId>.jpg
    label  Ori<oriId>.txt

`oriId` must not contain '_'; `beamId` may. Layouts 'flat' (pro/ beam/) and
'research' (ProjectorImage/ BeamImage/ OriginalImage/) are auto-detected.
"""

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils.image import IMAGE_EXT, read_bgr, resize

FLAT_DIRS = {"pro": "pro", "beam": "beam", "clean": "clean"}
RESEARCH_DIRS = {"pro": "ProjectorImage", "beam": "BeamImage",
                 "clean": "OriginalImage"}


def _images_in(directory) -> Dict[str, str]:
    if not directory or not os.path.isdir(directory):
        return {}
    return {f: os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(IMAGE_EXT)}


def ori_id(name: str) -> str:
    """oriId from a pro filename, or from a clean/label filename."""
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.startswith("projected_"):
        return stem[len("projected_"):].split("_")[0]
    return stem.replace("Ori", "")


def beam_id(name: str) -> str:
    """beamId from a pro filename, or from a beam filename."""
    stem = os.path.splitext(os.path.basename(name))[0]
    if stem.startswith("projected_"):
        return "_".join(stem[len("projected_"):].split("_")[1:])
    return stem.replace("output_video_", "")


def resolve_dirs(root) -> Tuple[str, Dict[str, str]]:
    """Return (layout_name, {role: absolute dir}) for whichever layout `root` uses."""
    root = str(root)
    for layout, mapping in (("flat", FLAT_DIRS), ("research", RESEARCH_DIRS)):
        if os.path.isdir(os.path.join(root, mapping["pro"])):
            return layout, {k: os.path.join(root, v) for k, v in mapping.items()}
    if _images_in(root):
        return "mixed", {"pro": root, "beam": root, "clean": root}
    raise FileNotFoundError(
        f"no recognised data layout under {root}\n"
        f"    expected '{FLAT_DIRS['pro']}/' or '{RESEARCH_DIRS['pro']}/', "
        f"or images directly in the folder"
    )


@dataclass(frozen=True)
class Sample:
    """One inference unit. `clean` and `label` are optional (needed only to score)."""

    name_id: str
    pro: str
    beam: str
    clean: Optional[str] = None
    label: Optional[str] = None

    @property
    def ori_id(self) -> str:
        return ori_id(self.pro)


def find_samples(input_root, gt_root=None, limit=0) -> List[Sample]:
    """
    Pair every `pro` image with its `beam`, attaching clean/label when available.

    Unmatched `pro` files are skipped with a warning rather than failing the run.
    """
    layout, dirs = resolve_dirs(input_root)
    pro_files = _images_in(dirs["pro"])
    beam_files = _images_in(dirs["beam"])

    if layout == "mixed":
        pro_files = {k: v for k, v in pro_files.items() if k.startswith("projected_")}
        beam_files = {k: v for k, v in beam_files.items()
                      if k.startswith("output_video_")}

    beam_by_id = {beam_id(k): v for k, v in beam_files.items()}

    clean_by_id, label_by_id = {}, {}
    if gt_root and os.path.isdir(str(gt_root)):
        gt_root = str(gt_root)
        clean_dir = next((os.path.join(gt_root, d) for d in ("clean", "OriginalImage")
                          if os.path.isdir(os.path.join(gt_root, d))), gt_root)
        clean_by_id = {ori_id(k): v for k, v in _images_in(clean_dir).items()}
        labels_dir = os.path.join(gt_root, "labels")
        if os.path.isdir(labels_dir):
            label_by_id = {ori_id(f): os.path.join(labels_dir, f)
                           for f in os.listdir(labels_dir) if f.endswith(".txt")}

    samples, skipped = [], []
    for fname, pro_path in pro_files.items():
        beam_path = beam_by_id.get(beam_id(fname))
        if beam_path is None:
            skipped.append(fname)
            continue
        oid = ori_id(fname)
        samples.append(Sample(
            name_id=os.path.splitext(fname)[0],
            pro=pro_path, beam=beam_path,
            clean=clean_by_id.get(oid), label=label_by_id.get(oid),
        ))

    if skipped:
        print(f"warning: {len(skipped)} pro image(s) had no matching beam, skipped "
              f"(e.g. {skipped[0]})")
    if not samples:
        raise RuntimeError(
            f"no pro/beam pairs found under {input_root} (layout: {layout}).\n"
            f"    check the naming: projected_<oriId>_<beamId>.jpg and "
            f"output_video_<beamId>.jpg"
        )
    samples.sort(key=lambda s: s.name_id)
    return samples[:limit] if limit else samples


def images_under(patterns) -> Dict[str, str]:
    """
    Every image under one or more directories; a glob may stand for several.

    Real captures arrive date-partitioned (`WarpData_0520_pro`, `..._0529_pro`, ...)
    and the beam frames often sit under a root of their own, so a training set is
    rarely one folder.
    """
    if not patterns:
        return {}
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]

    found = {}
    for pattern in patterns:
        matches = sorted(d for d in glob.glob(str(pattern)) if os.path.isdir(d))
        for directory in matches or ([str(pattern)] if os.path.isdir(str(pattern))
                                     else []):
            found.update(_images_in(directory))
    return found


def index_triplets(root=None, clean_root=None, sample=0, seed=42,
                   pro=None, beam=None, clean=None) -> List[List[str]]:
    """
    [[pro, clean, beam], ...] for training.

    Either pass `root` and let the layout be detected, or name the three directories
    outright - `pro` / `beam` / `clean` each accept a directory, a glob, or a list.
    `clean_root` overrides only the clean side of a detected layout.
    """
    if pro or beam or clean:
        source = f"pro={pro}"
        pro_files = images_under(pro)
        beam_files = images_under(beam)
        clean_files = images_under(clean)
    else:
        layout, dirs = resolve_dirs(root)
        source = f"{root} (layout: {layout})"
        pro_files = _images_in(dirs["pro"])
        beam_files = _images_in(dirs["beam"])
        clean_files = _images_in(clean_root or dirs["clean"])

    if not clean_files:
        raise FileNotFoundError(f"training needs clean targets; none found for {source}")
    beam_by_id = {beam_id(k): v for k, v in beam_files.items()}
    clean_by_id = {ori_id(k): v for k, v in clean_files.items()}

    triplets, no_beam, no_clean = [], 0, 0
    for fname, pro_path in pro_files.items():
        c = clean_by_id.get(ori_id(fname))
        b = beam_by_id.get(beam_id(fname))
        if c and b:
            triplets.append([pro_path, c, b])
        elif not b:
            no_beam += 1
        else:
            no_clean += 1
    if not triplets:
        raise RuntimeError(f"no complete triplets for {source}")

    triplets.sort()
    if sample and len(triplets) > sample:
        triplets = random.Random(seed).sample(triplets, sample)
    print(f"data: {len(triplets):,} triplets of {len(pro_files):,} pro image(s) "
          f"from {source}")
    if no_beam or no_clean:
        print(f"      skipped {no_beam:,} without a beam, {no_clean:,} without a clean")
    return triplets


def load_yolo_labels(path, img_w, img_h) -> List[Tuple[int, Tuple[int, int, int, int]]]:
    """YOLO txt (`cls cx cy w h`, normalised) -> [(cls_id, (x1,y1,x2,y2)), ...] pixels."""
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = int(round((cx - bw / 2) * img_w))
            y1 = int(round((cy - bh / 2) * img_h))
            x2 = int(round((cx + bw / 2) * img_w))
            y2 = int(round((cy + bh / 2) * img_h))
            out.append((cls_id, (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))))
    return out


def load_class_names(path=None) -> Optional[List[str]]:
    """Read `names:` from a YOLO dataset.yaml or the bundled detection config."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"class list not found: {path}")
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names") or (data.get("detector") or {}).get("class_names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        raise ValueError(f"no 'names' entry in {path}")
    return list(names)


class TripletPatchDataset:
    """
    Resize each triplet to `big_size`, then take one shared random crop of `small_size`.

    Yields (pro, beam, clean) as CHW float tensors in [-1, 1]. A map-style dataset
    that torch's DataLoader accepts directly; torch is imported per item so this
    module stays usable for pure discovery when torch is absent.
    """

    def __init__(self, triplets: Sequence[Sequence[str]],
                 small_size: Tuple[int, int] = (180, 320),
                 big_size: Tuple[int, int] = (360, 640)):
        self.triplets = list(triplets)
        self.small_h, self.small_w = small_size
        self.big_h, self.big_w = big_size
        if self.small_h > self.big_h or self.small_w > self.big_w:
            raise ValueError("small_size must fit inside big_size")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        import torch
        pro_path, clean_path, beam_path = self.triplets[idx]
        size = (self.big_w, self.big_h)
        pro = resize(read_bgr(pro_path), size)
        clean = resize(read_bgr(clean_path), size)
        beam = resize(read_bgr(beam_path), size)

        top = random.randint(0, self.big_h - self.small_h)
        left = random.randint(0, self.big_w - self.small_w)
        crop = (slice(top, top + self.small_h), slice(left, left + self.small_w))

        def to_chw(img):
            rgb = cv2.cvtColor(img[crop], cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

        return to_chw(pro), to_chw(beam), to_chw(clean)

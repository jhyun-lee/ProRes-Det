"""
Sample discovery, ground truth loading, and the training dataset.

Filenames tie the views of one moment together (see data/README_data.md):

    distorted  distorted_<surfaceId>_<lightId>.jpg
    light      light_<lightId>.jpg
    surface    surface_<surfaceId>.jpg
    label      surface_<surfaceId>.txt

`surfaceId` must not contain '_'; `lightId` may. Layouts 'flat' (distorted/ light/
surface/) and 'research' (ProjectorImage/ BeamImage/ OriginalImage/) are
auto-detected.

Sessions collected before the rename used pro/ beam/ clean/ and the prefixes
`projected_` / `output_video_` / `Ori`. Those are still recognised on read so an
existing dataset keeps working; nothing writes them any more.
"""

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils.image import IMAGE_EXT, read_bgr, resize

DISTORTED_PREFIX = "distorted_"
LIGHT_PREFIX = "light_"
SURFACE_PREFIX = "surface_"

LEGACY_DISTORTED_PREFIX = "projected_"
LEGACY_LIGHT_PREFIX = "output_video_"
LEGACY_SURFACE_PREFIX = "Ori"

# Every prefix a filename of that role may carry, newest first.
DISTORTED_PREFIXES = (DISTORTED_PREFIX, LEGACY_DISTORTED_PREFIX)
LIGHT_PREFIXES = (LIGHT_PREFIX, LEGACY_LIGHT_PREFIX)
SURFACE_PREFIXES = (SURFACE_PREFIX, LEGACY_SURFACE_PREFIX)

FLAT_DIRS = {"distorted": "distorted", "light": "light", "surface": "surface"}
LEGACY_FLAT_DIRS = {"distorted": "pro", "light": "beam", "surface": "clean"}
RESEARCH_DIRS = {"distorted": "ProjectorImage", "light": "BeamImage",
                 "surface": "OriginalImage"}

# Tried in order by resolve_dirs; the first whose `distorted` folder exists wins.
LAYOUTS = (("flat", FLAT_DIRS), ("legacy-flat", LEGACY_FLAT_DIRS),
           ("research", RESEARCH_DIRS))

# Where a ground-truth root may keep the surface images, newest first.
SURFACE_GT_DIRS = (FLAT_DIRS["surface"], RESEARCH_DIRS["surface"],
                   LEGACY_FLAT_DIRS["surface"])


def _images_in(directory) -> Dict[str, str]:
    if not directory or not os.path.isdir(directory):
        return {}
    return {f: os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(IMAGE_EXT)}


def _strip(stem: str, prefixes) -> Optional[str]:
    """The part after whichever of `prefixes` the stem starts with, else None."""
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return None


def surface_id(name: str) -> str:
    """surfaceId from a distorted filename, or from a surface/label filename."""
    stem = os.path.splitext(os.path.basename(name))[0]
    rest = _strip(stem, DISTORTED_PREFIXES)
    if rest is not None:
        return rest.split("_")[0]
    rest = _strip(stem, SURFACE_PREFIXES)
    return stem if rest is None else rest


def light_id(name: str) -> str:
    """lightId from a distorted filename, or from a light filename."""
    stem = os.path.splitext(os.path.basename(name))[0]
    rest = _strip(stem, DISTORTED_PREFIXES)
    if rest is not None:
        return "_".join(rest.split("_")[1:])
    rest = _strip(stem, LIGHT_PREFIXES)
    return stem if rest is None else rest


def resolve_dirs(root) -> Tuple[str, Dict[str, str]]:
    """Return (layout_name, {role: absolute dir}) for whichever layout `root` uses."""
    root = str(root)
    for layout, mapping in LAYOUTS:
        if os.path.isdir(os.path.join(root, mapping["distorted"])):
            return layout, {k: os.path.join(root, v) for k, v in mapping.items()}
    if _images_in(root):
        return "mixed", {"distorted": root, "light": root, "surface": root}
    raise FileNotFoundError(
        f"no recognised data layout under {root}\n"
        f"    expected '{FLAT_DIRS['distorted']}/' or "
        f"'{RESEARCH_DIRS['distorted']}/', or images directly in the folder"
    )


@dataclass(frozen=True)
class Sample:
    """One inference unit. `surface` and `label` are optional (needed only to score)."""

    name_id: str
    distorted: str
    light: str
    surface: Optional[str] = None
    label: Optional[str] = None

    @property
    def surface_id(self) -> str:
        return surface_id(self.distorted)


def find_samples(input_root, gt_root=None, limit=0) -> List[Sample]:
    """
    Pair every `distorted` image with its `light`, attaching surface/label when
    available.

    Unmatched `distorted` files are skipped with a warning rather than failing the run.
    """
    layout, dirs = resolve_dirs(input_root)
    distorted_files = _images_in(dirs["distorted"])
    light_files = _images_in(dirs["light"])

    if layout == "mixed":
        distorted_files = {k: v for k, v in distorted_files.items()
                           if k.startswith(DISTORTED_PREFIXES)}
        light_files = {k: v for k, v in light_files.items()
                       if k.startswith(LIGHT_PREFIXES)}

    light_by_id = {light_id(k): v for k, v in light_files.items()}

    surface_by_id, label_by_id = {}, {}
    if gt_root and os.path.isdir(str(gt_root)):
        gt_root = str(gt_root)
        surface_dir = next((os.path.join(gt_root, d) for d in SURFACE_GT_DIRS
                            if os.path.isdir(os.path.join(gt_root, d))), gt_root)
        surface_by_id = {surface_id(k): v
                         for k, v in _images_in(surface_dir).items()}
        labels_dir = os.path.join(gt_root, "labels")
        if os.path.isdir(labels_dir):
            label_by_id = {surface_id(f): os.path.join(labels_dir, f)
                           for f in os.listdir(labels_dir) if f.endswith(".txt")}

    samples, skipped = [], []
    for fname, distorted_path in distorted_files.items():
        light_path = light_by_id.get(light_id(fname))
        if light_path is None:
            skipped.append(fname)
            continue
        sid = surface_id(fname)
        samples.append(Sample(
            name_id=os.path.splitext(fname)[0],
            distorted=distorted_path, light=light_path,
            surface=surface_by_id.get(sid), label=label_by_id.get(sid),
        ))

    if skipped:
        print(f"warning: {len(skipped)} distorted image(s) had no matching light, "
              f"skipped (e.g. {skipped[0]})")
    if not samples:
        raise RuntimeError(
            f"no distorted/light pairs found under {input_root} (layout: {layout}).\n"
            f"    check the naming: {DISTORTED_PREFIX}<surfaceId>_<lightId>.jpg and "
            f"{LIGHT_PREFIX}<lightId>.jpg"
        )
    samples.sort(key=lambda s: s.name_id)
    return samples[:limit] if limit else samples


def images_under(patterns) -> Dict[str, str]:
    """
    Every image under one or more directories; a glob may stand for several.

    Real captures arrive date-partitioned (`WarpData_0520_distorted`, `..._0529_...`)
    and the light frames often sit under a root of their own, so a training set is
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


def _triplet_sources(root, surface_root, distorted, light, surface):
    """
    (distorted, light, surface) name -> path maps, plus a label for the messages.

    The two ways of naming a training set, kept apart from the pairing below: three
    directories given outright, or one root whose layout is detected.
    """
    if distorted or light or surface:
        return (images_under(distorted), images_under(light), images_under(surface),
                f"distorted={distorted}")
    layout, dirs = resolve_dirs(root)
    return (_images_in(dirs["distorted"]), _images_in(dirs["light"]),
            _images_in(surface_root or dirs["surface"]),
            f"{root} (layout: {layout})")


def index_triplets(root=None, surface_root=None, sample=0, seed=42,
                   distorted=None, light=None, surface=None) -> List[List[str]]:
    """
    [[distorted, surface, light], ...] for training.

    Either pass `root` and let the layout be detected, or name the three directories
    outright - `distorted` / `light` / `surface` each accept a directory, a glob, or
    a list. `surface_root` overrides only the surface side of a detected layout.
    """
    distorted_files, light_files, surface_files, source = _triplet_sources(
        root, surface_root, distorted, light, surface)

    if not surface_files:
        raise FileNotFoundError(
            f"training needs surface targets; none found for {source}")
    light_by_id = {light_id(k): v for k, v in light_files.items()}
    surface_by_id = {surface_id(k): v for k, v in surface_files.items()}

    triplets, no_light, no_surface = [], 0, 0
    for fname, distorted_path in distorted_files.items():
        s = surface_by_id.get(surface_id(fname))
        li = light_by_id.get(light_id(fname))
        if s and li:
            triplets.append([distorted_path, s, li])
        elif not li:
            no_light += 1
        else:
            no_surface += 1
    if not triplets:
        raise RuntimeError(f"no complete triplets for {source}")

    triplets.sort()
    if sample and len(triplets) > sample:
        triplets = random.Random(seed).sample(triplets, sample)
    print(f"data: {len(triplets):,} triplets of {len(distorted_files):,} distorted "
          f"image(s) from {source}")
    if no_light or no_surface:
        print(f"      skipped {no_light:,} without a light, {no_surface:,} without "
              f"a surface")
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


class TripletPatchDataset:
    """
    Resize each triplet to `big_size`, then take one shared random crop of `small_size`.

    Yields (distorted, light, surface) as CHW float tensors in [-1, 1]. A map-style
    dataset that torch's DataLoader accepts directly; torch is imported per item so
    this module stays usable for pure discovery when torch is absent.
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
        distorted_path, surface_path, light_path = self.triplets[idx]
        size = (self.big_w, self.big_h)
        distorted = resize(read_bgr(distorted_path), size)
        surface = resize(read_bgr(surface_path), size)
        light = resize(read_bgr(light_path), size)

        top = random.randint(0, self.big_h - self.small_h)
        left = random.randint(0, self.big_w - self.small_w)
        crop = (slice(top, top + self.small_h), slice(left, left + self.small_w))

        def to_chw(img):
            rgb = cv2.cvtColor(img[crop], cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32) / 127.5 - 1.0
            return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

        return to_chw(distorted), to_chw(light), to_chw(surface)

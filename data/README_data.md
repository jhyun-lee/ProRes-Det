# data/

A toy dataset that runs out of the box. Real data drops in with no code change as
long as it follows the filename convention below — no counts or paths are hard-coded.

Unlike `weights/`, this directory is tracked in git, so it is there right after a
clone.

```
data/
├── sample_input/           input for demo.py / evaluate.py / train.py
│   ├── pro/    projected_<oriId>_<beamId>.jpg   22 files, 640×360
│   └── beam/   output_video_<beamId>.jpg        22 files, mixed sizes
├── sample_gt/              ground truth (demo.py runs without it, just unscored)
│   ├── clean/  Ori<oriId>.jpg      10 files, 640×360  ← training target / PSNR·SSIM ref
│   └── labels/ Ori<oriId>.txt      10 files, 107 boxes total  ← mAP ref (YOLO format)
└── live/                   input for demo.py --live
    ├── BeamVideo.mp4       clip to play through the projector
    │                       5,858 frames @30fps = 3.3 min, 854×480, 55 MiB
    └── BaseBackGround.jpg  background shown during calibration, 1280×960
```

Image sizes need not match. The pipeline resizes everything to the model's
`input_size` (640×360 by default), which is why `beam/` mixes 854×480 and 1280×720
without any special handling.

## Filename convention

Three views of one moment are tied together by ids inside the filename.

```
projected_0409001429_0404023332_294_75.jpg
          └───┬────┘ └──────┬───────┘
            oriId         beamId

  → sample_gt/clean/Ori0409001429.jpg                     (ground truth screen)
  → sample_gt/labels/Ori0409001429.txt                    (detection ground truth)
  → sample_input/beam/output_video_0404023332_294_75.jpg  (frame the projector emitted)
```

| Role | Filename | In the model |
|---|---|---|
| `pro` | `projected_<oriId>_<beamId>.jpg` | input ch 0:3 — camera view of the projected screen |
| `beam` | `output_video_<beamId>.jpg` | input ch 3:6 — the frame the projector emitted |
| `clean` | `Ori<oriId>.jpg` | training target / PSNR·SSIM reference |
| `label` | `Ori<oriId>.txt` | detection mAP reference |

- No `_` in `oriId` — everything up to the first `_` is taken as the oriId.
  `beamId` may contain `_` (the example above is `0404023332_294_75`).
- One `clean` backing several `pro` captures is normal. Here 10 clean images back 22
  `pro` files, 1–3 each.
- `beam` is 1:1 with `pro`.
- Recognised extensions: `.jpg` `.jpeg` `.png` `.bmp`.
- A `pro` with no matching `beam` is skipped with a warning — the run does not fail.
- `clean` and `label` are optional. Missing `clean` only drops PSNR/SSIM; `evaluate.py`
  scores just the samples that have both.

## Layouts

Two are auto-detected by which folder exists.

| Name | Folders |
|---|---|
| `flat` (used here) | `pro/` `beam/` (+ `clean/`) |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/sample_input --gt data/sample_gt      # flat
python demo.py --input /path/to/WarpData_0520                     # research, auto-detected
```

If images sit loose in one folder it is handled as `mixed`, split by the `projected_`
and `output_video_` prefixes.

## Label format

Standard YOLO — one line per box, `<cls_id> <cx> <cy> <w> <h>`, all normalised to 0–1.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id` runs `0..16` and follows the `names` order in
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) — 11 fruits
(Apple … Watermelon) then 6 animals (Cat … Snake). All 17 classes appear in the bundled
labels.

## Swapping in real data

```bash
# 1) fill the same structure and just run
python demo.py
python evaluate.py

# 2) or point at the original dataset directly
python train.py --data-root /mnt/.../0_ImageData/1_WarpData_0520 --epochs 30
```

Training requires `clean` targets. `train.py` searches this order and reads whichever
exists — no symlink or copy needed:

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```

## Size

`data/` is 59 MiB, almost all of it `data/live/BeamVideo.mp4` (55 MiB). Delete that file
if you never use `--live`.

> `data/sample_input/clean` is a **broken symlink** pointing at an absolute path from
> another machine. The current code ignores it, but a plain `os.walk('data')` will
> break on it. Cleaning up is advised: `git rm data/sample_input/clean`

---

[한국어](README_data.ko.md) · [← README](../README.md)

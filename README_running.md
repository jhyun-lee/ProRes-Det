# Running ProRes-Det

Input, output and options for each entry point. For the one-liners that just work, see
[Quick start](README.md#3-running).

- [`demo.py` — restore → detect](#demopy--restore--detect)
- [`evaluate.py` — score before vs after](#evaluatepy--score-before-vs-after)
- [`train.py` — retrain the restoration model](#trainpy--retrain-the-restoration-model)
- [`demo.py --live` — webcam + projector](#demopy---live--webcam--projector)
- [`data/collect.py` — build a dataset with the rig](#datacollectpy--build-a-dataset-with-the-rig)
- [`data/record.py` — project and record, no models](#datarecordpy--project-and-record-no-models)
- [Output format](#output-format)

Flags shared by all three scripts: `--restorer-weights`, `--det-weights`, `--device`,
`--fp16`, `--classes`, `--restoration-config`, `--detection-config`. Full list:
`python <script>.py --help`.

---

## `demo.py` — restore → detect

Restoration and detection only. It never touches ground truth and never scores
anything - `evaluate.py` does that.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| Output | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detector yolo\|ssd\|none` | Detection backend (default yolo) |
| `--conf <float>` | Detector confidence floor (default 0.25) |
| `--limit N` | Cap how many pairs are processed (0 = all) |
| `--best-per-class` | Keep only the highest-confidence box per class |
| `--save-every N` | Image save interval. `0` keeps csv only |
| `--save-kinds a,b` | Which image kinds to save: `distorted`, `restored`, `panel` (default: all three) |
| `--max-saved-frames N` | Hard cap on how many frame sets land on disk |
| `--jpeg-quality N` | JPEG quality for saved images (default 92) |
| `--video` | Also write the 2×2 panels as `result.mp4` |

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds restored
python demo.py --input /path/to/pairs --name my_run
```

`--save-every 0` writes no images at all — useful when you only want the csv and the
summary, e.g. when sweeping detectors:

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

---

## `evaluate.py` — score before vs after

Only samples that have both `clean` and `label` are scored.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| GT | `data/sample_input/` (`clean/` + `labels/`) | `--gt <dir>` |
| Output | `output/Eval_<input dataset>/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--detectors yolo,ssd` | Compare several backends in one run (one row each) |
| `--iou <float>` | IoU threshold for a true positive (default 0.5) |
| `--limit N` | Cap how many pairs are scored |
| `--best-per-class` | Keep only the highest-confidence box per class |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

The report directory is named after the input dataset, so `data/sample_input` scores
into `output/Eval_sample_input/`. Re-running the same dataset replaces it, and stale
per-backend csvs from the previous run are cleared first.

A summary table is printed; per-class P / R / F1 / AP goes into the csv.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_input/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # per-class AP shift
```

`mAP` here is the single-IoU-threshold average precision over classes (VOC style, area
under the interpolated PR curve) — not COCO's IoU-averaged metric.

> `--det-weights` names one checkpoint, so it can only belong to one backend. With
> `--detectors a,b`, say which one owns it via `--detector`; otherwise it is warned
> about and ignored, and both backends fall back to `configs/detection.yaml`.

---

## `train.py` — retrain the restoration model

Only complete `pro` / `beam` / `clean` triplets are used.

| | Default path | Option to change it |
|---|---|---|
| Input | `data/sample_input/` (`pro/` + `beam/`) | `--data-root <dir>` |
| Target | `data/sample_input/clean/` — **required** | `--gt <dir>` |
| Output | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/`<br>`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png` | `--out <dir>` |

Clean targets are searched in this order, and whichever exists is read directly — no
symlink or copy needed (filename convention: [data/README_data.md](data/README_data.md)):

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```

The bundled set keeps them in `data/sample_input/clean/`, so the second entry hits and
`--gt` is only needed when the targets live outside `--data-root`.

| Option | Meaning |
|---|---|
| `--epochs N` `--batch-size N` `--lr F` `--accum-steps N` | Defaults come from `configs/restoration.yaml` |
| `--sample N` | Cap how many triplets are used |
| `--num-workers N` | DataLoader workers (default 4, or 8 for batch > 4) |
| `--resume <ckpt>` | Continue from a checkpoint (its architecture wins over `--no-*`) |
| `--seed N` | Default 42 |
| `--no-amp` | Disable mixed precision on CUDA |
| `--save-every N` | Epochs between `epoch_N.pt` saves |

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

Needs the training extra: `pip install -e ".[train]"`.


Where the training set lives is configured, not hard-coded. `train.data` in
[configs/restoration.yaml](projector_distortion/configs/restoration.yaml) names the
three directories; each takes a directory, a glob, or a list, because real captures
come date-partitioned and the beam frames usually sit under a root of their own:

```yaml
train:
  data:
    pro:   "D:/captures/WarpData_*_pro"
    clean: "D:/captures/WarpData_*_ori"
    beam:  "D:/captures/Learning_video_frames"
```

```bash
python train.py --epochs 30                      # uses train.data
python train.py --pro-dir ... --beam-dir ... --clean-dir ...   # override for one run
```

Clear all three (or pass `--data-root`) to fall back to `data/sample_input`. A `pro`
with no matching `clean` or `beam` is counted and skipped, not fatal - the run prints
how many went each way.

### Ablation

Ten structural pieces can be switched off individually. Whatever is off lands in `tag`,
which goes into both the run folder and the checkpoint filename (`NoCA`,
`NoCA-NoSkip3`, …).

```
--no-prenorm  --no-naf-norm  --no-simple-gate  --no-naf-scale  --no-ca
--no-skip1    --no-skip2     --no-skip3        --no-bottleneck  --no-tanh
```

Capacity can be overridden too: `--base-dim`, `--enc-depth`, `--dec-depth`,
`--bottleneck-depth`, `--dw-expand`, `--ffn-expand`, `--ca-reduction`.

```bash
python train.py --no-ca --epochs 30
```

Checkpoints embed their own architecture config, so the flags never need repeating:

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

---

## `demo.py --live` — webcam + projector

Needs real hardware: a webcam pointed at a screen a projector is throwing to.

| | Default | Option to change it |
|---|---|---|
| Projected clip | `data/live/BeamVideo.mp4` | `--clip <path>` |
| Calibration background | `data/live/BaseBackGround.jpg` | `--background <path>` |
| Camera | webcam 0, 1280×960 @30fps | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| Output | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <name>` |

| Option | Meaning |
|---|---|
| `--screen N` | Monitor index the projector is attached to (0 = primary) |
| `--offset N` | Projector→camera latency in frames (default 6) |
| `--manual-calib` | Click the 4 corners instead of auto-detecting them |
| `--debug-view` | Show the pre-warp camera feed with the quad, live |
| `--calib-settle F` | Seconds to wait after each calibration flash (default 0.8) |
| `--max-frames N` | 0 = until the clip ends |
| `--analyse-every N` | Restore+detect every Nth projected frame (0 = measure the machine and pick N) |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

If you do not know `--screen`, pass anything and run — the detected monitor table is
printed first. Press `q` in the `Combined_View` window to stop.



### Playback and analysis run at different rates

The projector plays the clip at the clip's own fps no matter how slow the model is.
Restore+detect happens on a worker thread over an evenly spaced subset of frames -
every Nth - so the recorded panel plays like the clip, only at a lower rate. Frames
between two analysed ones are still projected and captured; they are just not scored.

`--analyse-every 0` (the default) times the first dozen analysed frames, discards the
CUDA warmup outlier, and fixes N from the median. It is never re-tuned mid-run,
because changing N is itself what makes the analysed video uneven.

The summary reports both rates:

```
projector 29.1 fps (450 frames) | analysis 13.0 fps (every 2 frame(s): 201 analysed,
249 skipped, 0 dropped)
```

`skipped` is by design - those frames were never meant for the model. `dropped` means
the worker missed its deadline and is the number to watch.


`--analyse-every 0` is the default: the densest N whose spacing stays perfectly even.
Going denser than that means giving up some of the evenness, which is a call worth
making deliberately - on a 30 fps clip with a worker that takes ~41 ms per frame:

| `--analyse-every` | projector | analysis | frames analysed | spacing |
|---|---|---|---|---|
| `1` | 28.5 fps | 21.8 fps | 76% | 89% even |
| `2` (auto here) | 28.8 fps | 13.9 fps | 48% | 100% even |
| `3` | 28.8 fps | 9.4 fps | 32% | 100% even |

`1` analyses 1.6x more frames and barely touches playback; it just cannot hit every
slot, since a 30 fps budget is 33 ms and the worker needs 41. Pick it when coverage
matters more than a perfectly smooth panel.


### How calibration works

Black/white flash → the two camera shots are differenced (only the projection changes,
so ambient light cancels) → the largest 4-corner contour is the screen → one homography
is computed and reused for every frame.

**Calibration happens once, before the loop.** If the camera or projector moves mid-run,
the warp stays wrong for the rest of the session. Watch for that with `--debug-view`.
Auto-detection falls back to manual clicking on failure; `--manual-calib` starts there.

### When calibration looks wrong

`output/<run>/calib/` is written even when detection failed — that is when it matters.

| File | What to look for |
|---|---|
| `quad.jpg` | Do the 4 points sit exactly on the screen corners? |
| `mask.jpg` | Is the white region just the screen, or did lights/windows get caught? |
| `diff.jpg` | Is the black/white flash difference strong enough? If not, raise `--calib-settle` |
| `warped.jpg` | Is the rectified result actually square? |
| `frame_pre.jpg` | the first camera frame of the run, raw |
| `frame_post.jpg` | that frame rectified, at the model input size |
| `frame_compare.jpg` | both side by side — also shown once in the `Warp_FirstFrame` window |

The first four come from the black/white flashes taken *before* the loop; the three
`frame_*` files come from the run itself, so they show the warp the frames were
actually rectified with. If the projection drifted between calibration and the first
frame, that is where it shows.

### Long unattended recording

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

The csv covers every frame while images land sparsely, which keeps disk usage
predictable.

On non-Windows systems this needs `pip install -e ".[live]"` for `screeninfo`. Windows
drives monitor placement through the Win32 API instead.

---

## `data/collect.py` — build a dataset with the rig

Four stages. `check` and `capture` drive the projector and the webcam; `beam` and `warp`
are plain file work and run anywhere. Everything lands in one session folder
(`data/collected_<MMDD>/` by default, `--root` moves it).

| Stage | Command | In → out |
|---|---|---|
| check | `python data/collect.py check` | monitor table + webcam preview |
| beam | `python data/collect.py beam --src videos/` | video → `<session>/beam/` |
| capture | `python data/collect.py capture --screen 2` | projector + webcam → `<session>/raw/` |
| warp | `python data/collect.py warp` | `raw/` → aligned `<session>/clean/` + `pro/` |

```bash
python data/collect.py beam    --src data/live/BeamVideo.mp4 --step 30
python data/collect.py capture --screen 2 --rounds 3 --limit 200 --shuffle
python data/collect.py warp
python demo.py --input data/collected_0803 --gt data/collected_0803
```

`capture`:

| Option | Meaning |
|---|---|
| `--screen N` | Monitor the projector is attached to (`check` prints the table) |
| `--rounds N` | Scene setups; each one starts with a fresh clean shot, `s` takes it |
| `--limit N` · `--shuffle` | Beam frames per round, and whether to spread them over the clip |
| `--settle-ms N` | Wait after showing a frame before capturing it (default 150) |
| `--flush N` | Buffered camera frames to drop first (default 3) — this is what keeps `pro` matched to its `beam` |
| `--background <path>` | Image projected while the clean shot is taken |

`warp`:

| Option | Meaning |
|---|---|
| `--warp boundary` | Default: corner homography + the measured edge bow |
| `--warp homography` | Corners only, for a flat screen and a clean boundary |
| `--warp tps` | The legacy thin-plate spline; needs `opencv-python<5`, which still ships the shape module |
| `--inset N` | Pixels to pull the boundary in, off the bright projection rim (default 2) |
| `--final-size W H` | Saved resolution, default 640×360 — the model input |
| `--no-debug` | Skip the before/after overlays in `<session>/debug/` |

Check `<session>/debug/<oriId>_warp.jpg` before capturing a full session: it shows the
detected boundary, the sampled points and the rectified result together. Naming, the
`oriId`/`beamId` contract and what to do about labels are in
[data/README_data.md](data/README_data.md).

---

## `data/record.py` — project and record, no models

The projection half of `--live` with the restore/detect half removed: the clip goes out
fullscreen at its own fps, the camera feed goes into one mp4, nothing else happens. No
weights are loaded, so this runs on a machine that has none — useful for capturing raw
distorted footage before there is a checkpoint, or for proving a rig works end to end.

| | Default | Option to change it |
|---|---|---|
| Projected clip | `data/live/BeamVideo.mp4` | `--clip <path>` |
| Calibration background | `data/live/BaseBackGround.jpg` | `--background <path>` |
| Output | `data/recordings/rec_<MMDDHHMMSS>.mp4` + a `.json` beside it | `--out <path>` |

| Option | Meaning |
|---|---|
| `--screen N` | Monitor the projector is attached to (0 = primary) |
| `--loop` | Restart the clip instead of stopping at its end |
| `--seconds F` · `--max-frames N` | Stop conditions (0 = no limit) |
| `--start-delay F` | Hold the background this long before recording starts |
| `--warp` | Calibrate once and record the rectified screen, not the raw camera frame |
| `--rec-size W H` | Resize before encoding (`0 0` keeps the camera resolution) |
| `--fps F` | mp4 header fps (0 = measure the camera's real rate) |
| `--codec <fourcc>` | Default `mp4v`; `avc1`, `XVID`, … |
| `--no-preview` | Skip the small camera preview window |

```bash
python data/record.py --screen 2 --seconds 30
python data/record.py --clip data/sample_video/mIni_Video_1.mp4 --loop
python data/record.py --warp --rec-size 640 360
```

Recording starts as soon as the window is up — no keypress. `q` stops early, as do the
end of the clip, `--seconds` and `--max-frames`.

The header fps is **measured**, not requested: webcams routinely deliver something other
than what they were asked for and report a third number, and a wrong header is exactly
what makes a recording play back in fast or slow motion. The probe frames are buffered
until the rate is known, so measuring costs no footage. The `.json` alongside the mp4
records the clip, camera and monitor settings the run actually used, plus the measured
rate and how many frames the encoder dropped.

---


## Making restoration faster

Restoration is ~46% of a live frame, so it is the first thing to shrink. The network
is fully convolutional, which makes its working resolution a runtime knob - no
retraining involved. Measured on the bundled set, `--detectors yolo`:

| `--input-size` | restore | detection mAP | PSNR gain | SSIM gain |
|---|---|---|---|---|
| `320 180` | 9.7 ms | 0.9866 | +8.95 dB | +0.163 |
| `480 270` | 13.4 ms | **1.0000** | +11.32 dB | +0.183 |
| `640 360` (default) | 20.5 ms | **1.0000** | **+13.17 dB** | **+0.218** |
| `854 480` | 41.2 ms | 1.0000 | +9.89 dB | +0.145 |

```bash
python demo.py --live --screen 2 --input-size 480 270
```

`480 270` costs a third of the restoration time and detection does not notice; only
the PSNR/SSIM gain narrows. Below that, mAP starts to slip.

Going **above** 640x360 is worse on both counts: the checkpoint was trained on
180x320 crops resized from 360x640, and 854x480 is far enough outside that scale that
restoration quality drops while costing twice the time.

`--fp16` and `torch.compile` are not worth reaching for here - fp16 measured 2% slower
(the network is small enough to be memory-bound, and autocast adds more than it saves)
and compile needs a Triton build Windows does not ship.

A genuinely smaller network needs retraining. For reference, at 640x360:

| `train.py` flags | params | forward |
|---|---|---|
| shipped | 4,184,259 | 13.8 ms |
| `--base-dim 32` | 1,878,659 | 9.7 ms |
| `--enc-depth 1,1,1 --dec-depth 1,1,1 --bottleneck-depth 1` | 2,418,837 | 7.3 ms |
| `--no-ca` | 4,116,147 | 13.1 ms |

Shallower blocks buy more than narrower ones. Dropping channel attention buys 5% and
is not worth the retrain.

## Output format

`demo.py` writes one directory per run:

```
output/<run_name>/
├── run_meta.json      config · environment · calibration · summary (all in one file)
├── detections.csv     one row per box, `source` separates distorted / restored
├── captures/          the footage itself, no boxes drawn on it
│   ├── <id>_distorted.jpg      before restoration
│   └── <id>_restored.jpg      after restoration
├── frames_all/        the 2×2 figures, tiles captioned (a)…(d)
│   └── <id>_panel.jpg          beam · distorted+boxes · restored+boxes · residual
├── calib/             --live only: the quad, the flashes it came from, and the
│                       first frame before and after rectification
└── result.mp4         video of the 2×2 panels (--live or --video)
```

`captures/` holds the run un-annotated. That is what lets `evaluate.py` score this
restoration afterwards, or another detector re-run over identical pixels - boxes burnt
into a jpg cannot be undone.

Only three kinds are written. The annotated views, the residual heatmap, the beam and
the raw camera frame used to be saved as separate jpgs under `frames/` as well; they are
tiles of the panel already, so four extra encodes per frame bought nothing and the
directory is gone. Drop kinds further with `--save-kinds`:

```bash
python demo.py --save-kinds panel            # comparison figures only
python demo.py --save-kinds distorted,restored   # re-scorable pixels only
```

`--save-every 0` skips the image directories entirely; `detections.csv` still covers
every frame.

`evaluate.py` writes `report.json` + `per_class_*.csv` + `per_image_*.csv` instead, and
`train.py` writes `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png` under
`runs/`.

---

[한국어](README_running.ko.md) · [← README](README.md)

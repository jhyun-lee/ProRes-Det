# Working in this repository

## Commits

Do not add a `Co-Authored-By:` trailer, or any other attribution line, to commit
messages. Authorship of this repository is the human author's alone.

## Verifying a change

Before claiming anything works, run these from the repo root and check the output:

```bash
python -m pytest src/tests -q          # expect: 151 passed
python src/evaluate.py                 # expect the numbers below, to the digit
```

`evaluate.py` on the default split must reproduce the README's reference table:

```
ssd  distorted   P 0.6793  R 0.8756  F1 0.7650  mAP 0.8086
ssd  restored    P 0.9956  R 1.0000  F1 0.9978  mAP 1.0000
PSNR 13.760 -> 26.928 dB    SSIM 0.7051 -> 0.9227
```

Those numbers are the fastest way to catch a broken path: everything the pipeline
loads hangs off `PROJECT_ROOT`, so a path that is off by one directory changes them
without raising anything.

## Layout

Source lives under `src/` — the package, the four entry scripts and the tests. The
dataset and its collection scripts live under `datasets/` at the root; those scripts
deliberately run without torch, so they must not import the pipeline package at
module level.

# 실행 가이드

각 진입점의 입력 · 출력 · 옵션. 인자 없이 바로 도는 명령들은
[빠른 실행](../README.ko.md#3-실행) 참고.

- [`demo.py` — 복원 → 검출](#demopy--복원--검출)
- [`evaluate.py` — 복원 전후 점수 비교](#evaluatepy--복원-전후-점수-비교)
- [`train.py` — 복원 모델 재학습](#trainpy--복원-모델-재학습)
- [`demo.py --live` — 웹캠 + 프로젝터](#demopy---live--웹캠--프로젝터)
- [출력물 형식](#출력물-형식)

세 스크립트가 공통으로 받는 플래그: `--restorer-weights`, `--det-weights`, `--device`,
`--fp16`, `--classes`, `--restoration-config`, `--detection-config`.
전체 목록은 `python <script>.py --help`.

---

## `demo.py` — 복원 → 검출

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| 입력 | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| 정답(선택) | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| 출력 | `output/<타임스탬프>/` | `--output <dir>` · `--name <이름>` |

정답이 없으면 PSNR/SSIM 만 생략되고 실행은 계속된다.

| 옵션 | 설명 |
|---|---|
| `--detector yolo\|ssd\|none` | 검출 백엔드 (기본 yolo) |
| `--conf <float>` | 검출 신뢰도 하한 (기본 0.25) |
| `--limit N` | 처리할 쌍 개수 제한 (0 = 전부) |
| `--best-per-class` | 클래스당 최고 신뢰도 박스 하나만 남김 |
| `--save-every N` | 이미지 저장 간격. `0` 이면 csv 만 남김 |
| `--save-kinds a,b` | 저장할 이미지 종류만 선택 (`captured,restored,panel` 등) |
| `--max-saved-frames N` | 디스크에 남길 프레임 세트 개수 상한 |
| `--jpeg-quality N` | 저장 이미지 JPEG 품질 (기본 92) |
| `--video` | 2×2 패널을 `result.mp4` 로도 저장 |

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds captured,restored,beam
python demo.py --input /path/to/pairs --gt /path/to/gt --name my_run
```

`--save-every 0` 이면 이미지를 아예 안 쓴다. csv 와 요약만 필요할 때, 예를 들어
검출기를 훑을 때 쓴다.

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

---

## `evaluate.py` — 복원 전후 점수 비교

`clean` 과 `label` 이 둘 다 있는 샘플만 채점한다.

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| 입력 | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| 정답 | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| 출력 | `output/eval/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--detectors yolo,ssd` | 여러 백엔드를 한 번에 비교 (백엔드마다 한 행) |
| `--iou <float>` | TP 판정 IoU 임계값 (기본 0.5) |
| `--limit N` | 채점할 쌍 개수 제한 |
| `--best-per-class` | 클래스당 최고 신뢰도 박스 하나만 남김 |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

콘솔에 요약 표가 뜨고, 클래스별 P / R / F1 / AP 는 csv 에 들어간다.

```python
import pandas as pd
pc = pd.read_csv("output/eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # 클래스별 AP 변화
```

여기서 `mAP` 는 단일 IoU 임계값의 클래스 평균 AP (VOC 방식, 보간된 PR 곡선 아래 면적)
이고, COCO 의 IoU 평균 지표가 아니다.

> `--det-weights` 는 체크포인트 하나를 가리키므로 백엔드 하나에만 속할 수 있다.
> `--detectors a,b` 와 같이 쓸 때는 `--detector` 로 소유 백엔드를 지정한다. 안 하면
> 경고 후 무시되고, 두 백엔드 모두 `configs/detection.yaml` 값을 쓴다.

---

## `train.py` — 복원 모델 재학습

`pro` / `beam` / `clean` 3장이 모두 갖춰진 것만 쓴다.

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| 입력 | `data/sample_input/` (`pro/` + `beam/`) | `--data-root <dir>` |
| 정답 | `data/sample_gt/clean/` — **필수** | `--gt <dir>` |
| 출력 | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/`<br>`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png` | `--out <dir>` |

정답은 아래 순서로 찾아서 있는 것을 그대로 읽는다. 심볼릭 링크나 복사는 필요 없다.
파일명 규약은 [data/README_data.ko.md](../data/README_data.ko.md) 참고.

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```

| 옵션 | 설명 |
|---|---|
| `--epochs N` `--batch-size N` `--lr F` `--accum-steps N` | 기본값은 `configs/restoration.yaml` |
| `--sample N` | 학습에 쓸 triplet 개수 제한 |
| `--num-workers N` | DataLoader 워커 수 (기본 4, batch > 4 면 8) |
| `--resume <ckpt>` | 체크포인트에서 이어서 학습 (구조는 체크포인트 쪽이 우선) |
| `--seed N` | 기본 42 |
| `--no-amp` | CUDA 혼합 정밀도 비활성화 |
| `--save-every N` | `epoch_N.pt` 저장 간격 (epoch 단위) |

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

학습 extras 필요: `pip install -e ".[train]"`.

### Ablation

구조 10종을 개별로 끌 수 있다. 끈 항목이 `tag` 에 기록되고, 그 태그가 실행 폴더명과
체크포인트 파일명에 그대로 들어간다 (`NoCA`, `NoCA-NoSkip3` …).

```
--no-prenorm  --no-naf-norm  --no-simple-gate  --no-naf-scale  --no-ca
--no-skip1    --no-skip2     --no-skip3        --no-bottleneck  --no-tanh
```

용량도 덮어쓸 수 있다: `--base-dim`, `--enc-depth`, `--dec-depth`,
`--bottleneck-depth`, `--dw-expand`, `--ffn-expand`, `--ca-reduction`.

```bash
python train.py --no-ca --epochs 30
```

체크포인트에 구조 설정이 동봉되므로, 나중에 쓸 때 플래그를 다시 줄 필요가 없다.

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

---

## `demo.py --live` — 웹캠 + 프로젝터

실제 하드웨어가 필요하다. 프로젝터가 쏘고 있는 스크린을 웹캠이 바라보는 구성.

| | 기본값 | 바꾸는 옵션 |
|---|---|---|
| 투사 클립 | `data/live/BeamVideo.mp4` | `--clip <path>` |
| 캘리브 배경 | `data/live/BaseBackGround.jpg` | `--background <path>` |
| 카메라 | 웹캠 0번, 1280×960 @30fps | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| 출력 | `output/<타임스탬프>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--screen N` | 프로젝터가 붙은 모니터 인덱스 (0 = 주 모니터) |
| `--offset N` | 프로젝터→카메라 지연 보정 (프레임 수, 기본 6) |
| `--manual-calib` | 자동 검출 대신 4점 직접 클릭 |
| `--debug-view` | 워핑 전 카메라 + 4점을 실시간 표시 |
| `--calib-settle F` | 캘리브 플래시 후 대기 시간 (초, 기본 0.8) |
| `--max-frames N` | 0 = 클립 끝까지 |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

`--screen` 을 모르면 아무 값이나 주고 실행 → 감지된 모니터 표가 먼저 출력된다.
중단은 `Combined_View` 창에서 `q`.

### 캘리브레이션 동작

흑/백 플래시 → 두 장의 카메라 샷을 차영상으로 (투사 영역만 바뀌므로 주변광이 상쇄됨)
→ 가장 큰 4각형 윤곽이 스크린 → 호모그래피 1회 계산 후 매 프레임 재사용.

**캘리브레이션은 루프 전에 한 번만** 한다. 실행 중 카메라나 프로젝터가 움직이면 남은
세션 내내 워프가 틀어진 채로 간다. `--debug-view` 로 확인할 것.
자동 검출이 실패하면 수동 클릭으로 폴백하고, 처음부터 수동은 `--manual-calib`.

### 캘리브레이션이 이상할 때

`output/<run>/calib/` 은 검출이 실패해도 기록된다. 그때가 제일 필요한 순간이라서다.

| 파일 | 보는 법 |
|---|---|
| `raw_points.jpg` | 4점이 스크린 모서리에 정확히 붙었나 |
| `mask.jpg` | 흰 영역이 스크린 하나인가, 조명/창문까지 잡았나 |
| `diff.jpg` | 흑백 플래시 차이가 충분한가. 부족하면 `--calib-settle` ↑ |
| `warped.jpg` | 최종 정면화 결과가 반듯한가 |

### 장시간 무인 녹화

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

csv 는 전 프레임 기록되고 이미지만 드문드문 남아 용량이 예측 가능해진다.

Windows 가 아니면 `screeninfo` 가 필요하다 (`pip install -e ".[live]"`).
Windows 는 Win32 API 로 모니터 배치를 직접 제어한다.

---

## 출력물 형식

`demo.py` 는 실행마다 디렉토리 하나를 만든다.

```
output/<run_name>/
├── run_meta.json      설정 · 환경 · 캘리브레이션 · 요약 (전부 한 파일)
├── detections.csv     박스 1개 = 1행, source 로 captured/restored 구분
├── frames.csv         프레임 1개 = 1행 (PSNR/SSIM/지연시간 포함)
├── frames/            샘플 이미지, --save-every 간격
│   ├── <id>_captured.jpg      복원 전 (박스 없음)
│   ├── <id>_restored.jpg      복원 후 (박스 없음)  ← 메트릭 입력용
│   ├── <id>_*_det.jpg         박스 그려진 버전
│   ├── <id>_residual.jpg      제거된 빛의 히트맵
│   └── <id>_panel.jpg         2×2 비교 패널
├── calib/             캘리브레이션 근거 (--live 만)
└── result.mp4         2×2 패널 영상 (--live 또는 --video)
```

clean 이미지와 annotated 이미지를 따로 저장하는 게 핵심이다. 덕분에 실행이 끝난 뒤에도
PSNR/SSIM 재계산과 다른 검출기로의 재실행이 가능하다.

`--save-every 0` 이면 `frames/` 자체가 안 생긴다. csv 는 항상 전 프레임을 담는다.

`evaluate.py` 는 대신 `report.json` + `per_class_*.csv` + `per_image_*.csv` 를,
`train.py` 는 `runs/` 아래에 `restorer_<tag>_best.pt` + `loss_log.csv` +
`loss_plots.png` 를 쓴다.

---

[English](README_running.md) · [← README](../README.ko.md)

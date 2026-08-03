# 실행 가이드

각 진입점의 입력 · 출력 · 옵션. 인자 없이 바로 도는 명령들은
[빠른 실행](README.ko.md#3-실행) 참고.

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

복원과 검출만 한다. 정답을 읽지 않고 채점도 하지 않는다. 그건 `evaluate.py` 몫이다.

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| 입력 | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| 출력 | `output/<타임스탬프>/` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--detector yolo\|ssd\|none` | 검출 백엔드 (기본 yolo) |
| `--conf <float>` | 검출 신뢰도 하한 (기본 0.25) |
| `--limit N` | 처리할 쌍 개수 제한 (0 = 전부) |
| `--best-per-class` | 클래스당 최고 신뢰도 박스 하나만 남김 |
| `--save-every N` | 이미지 저장 간격. `0` 이면 csv 만 남김 |
| `--save-kinds a,b` | 저장할 이미지 종류 (기본: `beam` 을 뺀 전부) |
| `--max-saved-frames N` | 디스크에 남길 프레임 세트 개수 상한 |
| `--jpeg-quality N` | 저장 이미지 JPEG 품질 (기본 92) |
| `--video` | 2×2 패널을 `result.mp4` 로도 저장 |

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds restored,residual
python demo.py --input /path/to/pairs --name my_run
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
| 출력 | `output/Eval_<입력데이터셋>/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--detectors yolo,ssd` | 여러 백엔드를 한 번에 비교 (백엔드마다 한 행) |
| `--iou <float>` | TP 판정 IoU 임계값 (기본 0.5) |
| `--limit N` | 채점할 쌍 개수 제한 |
| `--best-per-class` | 클래스당 최고 신뢰도 박스 하나만 남김 |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

리포트 디렉토리는 입력 데이터셋 이름을 딴다. `data/sample_input` 이면
`output/Eval_sample_input/` 이다. 같은 데이터셋을 다시 돌리면 덮어쓰되, 이전 실행의
백엔드별 csv 잔재는 먼저 지운다.

콘솔에 요약 표가 뜨고, 클래스별 P / R / F1 / AP 는 csv 에 들어간다.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_input/per_class_yolo.csv")
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
파일명 규약은 [data/README_data.ko.md](data/README_data.ko.md) 참고.

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

### 첫 프레임의 워핑 전후

`output/<run>/warp/` 는 첫 카메라 프레임이 루프에 들어올 때 딱 한 번 기록되고,
비교 그림은 `Warp_FirstFrame` 창에 띄워둔 채 실행이 계속된다.

| 파일 | 내용 |
|---|---|
| `first_frame_pre_warp.jpg` | 카메라 원본 프레임 |
| `first_frame_pre_warp_quad.jpg` | 같은 프레임 + 캘리브 4점 |
| `first_frame_post_warp.jpg` | 정면화 결과, 모델 입력 해상도 |
| `first_frame_compare.jpg` | 둘을 나란히 놓은 그림 — 화면에 뜨는 것과 동일 |

`calib/` 은 루프 **전**의 흑백 플래시에서 나온 것이고, 이쪽은 실행 자체에서 나온 것이다.
즉 프레임들이 실제로 적용받은 워프다. 캘리브레이션과 첫 프레임 사이에 투사가
틀어졌다면 여기서 드러난다.

### 장시간 무인 녹화

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

csv 는 전 프레임 기록되고 이미지만 드문드문 남아 용량이 예측 가능해진다.

Windows 가 아니면 `screeninfo` 가 필요하다 (`pip install -e ".[live]"`).
Windows 는 Win32 API 로 모니터 배치를 직접 제어한다.

---

## `data/collect.py` — 리그로 데이터셋 만들기

4단계. `check` 와 `capture` 는 프로젝터+웹캠이 필요하고, `beam` 과 `warp` 는 파일 작업뿐이라
아무 데서나 돌아간다. 전부 세션 폴더 하나에 쌓인다 (기본 `data/collected_<MMDD>/`,
`--root` 로 변경).

| 단계 | 명령 | 입력 → 출력 |
|---|---|---|
| check | `python data/collect.py check` | 모니터 표 + 웹캠 프리뷰 |
| beam | `python data/collect.py beam --src videos/` | 영상 → `<세션>/beam/` |
| capture | `python data/collect.py capture --screen 2` | 프로젝터+웹캠 → `<세션>/raw/` |
| warp | `python data/collect.py warp` | `raw/` → 정합된 `<세션>/clean/` + `pro/` |

```bash
python data/collect.py beam    --src data/live/BeamVideo.mp4 --step 30
python data/collect.py capture --screen 2 --rounds 3 --limit 200 --shuffle
python data/collect.py warp
python demo.py --input data/collected_0803 --gt data/collected_0803
```

`capture`:

| 옵션 | 설명 |
|---|---|
| `--screen N` | 프로젝터가 붙은 모니터 (`check` 가 표를 출력한다) |
| `--rounds N` | 장면 세팅 횟수. 매 라운드는 `s` 키로 찍는 정답 샷으로 시작 |
| `--limit N` · `--shuffle` | 라운드당 beam 프레임 수, 클립 전체에 고르게 뿌릴지 여부 |
| `--settle-ms N` | 프레임을 띄우고 촬영까지의 대기 (기본 150) |
| `--flush N` | 촬영 전 버릴 카메라 버퍼 프레임 수 (기본 3). `pro` 와 `beam` 을 맞추는 핵심 |
| `--background <path>` | 정답 샷을 찍는 동안 투사할 이미지 |

`warp`:

| 옵션 | 설명 |
|---|---|
| `--warp boundary` | 기본값. 코너 호모그래피 + 실측한 경계 휘어짐 보정 |
| `--warp homography` | 코너 4점만. 평평한 스크린 + 깨끗한 경계일 때 |
| `--warp tps` | 기존 thin-plate spline. shape 모듈이 남아있는 `opencv-python<5` 필요 |
| `--inset N` | 경계를 안쪽으로 당길 픽셀 수. 투사 테두리를 피한다 (기본 2) |
| `--final-size W H` | 저장 해상도. 기본 640×360 = 모델 입력 |
| `--no-debug` | `<세션>/debug/` 의 전후 비교 이미지 생략 |

본 세션을 찍기 전에 `<세션>/debug/<oriId>_warp.jpg` 를 먼저 확인할 것. 검출된 경계와
샘플 점, 정면화 결과가 한 장에 들어있다. 파일명 규칙(`oriId`/`beamId`)과 라벨 처리는
[data/README_data.ko.md](data/README_data.ko.md) 참고.

---

## 출력물 형식

`demo.py` 는 실행마다 디렉토리 하나를 만든다.

```
output/<run_name>/
├── run_meta.json      설정 · 환경 · 캘리브레이션 · 요약 (전부 한 파일)
├── detections.csv     박스 1개 = 1행, source 로 distorted/restored 구분
├── captures/          박스를 그리지 않은 촬영 원본
│   ├── <id>_distorted.jpg      복원 전
│   └── <id>_restored.jpg      복원 후
├── frames/            박스 그려진 뷰, --save-every 간격
│   ├── <id>_distorted_det.jpg  복원 전 + 박스
│   ├── <id>_restored_det.jpg  복원 후 + 박스
│   └── <id>_residual.jpg      제거된 빛의 히트맵
├── frames_all/        2×2 비교 그림, 각 칸에 (a)…(d) 캡션
├── calib/             캘리브레이션 근거 (--live 만)
├── warp/              첫 프레임의 워핑 전후 (--live 만)
└── result.mp4         2×2 패널 영상 (--live 또는 --video)
```

`captures/` 에는 박스를 그리지 않은 원본이 들어간다. 이게 있어야 나중에
`evaluate.py` 로 이 복원 결과를 채점하거나 동일한 픽셀에 다른 검출기를 다시 돌릴 수 있다.
jpg 에 한번 그려진 박스는 되돌릴 수 없다.

기본값에서 빠지는 건 `beam` 하나뿐이다. 패널에 이미 나오기 때문이다. 필요하면
`--save-kinds` 로 추가한다. 임의의 조합을 받는다.

```bash
python demo.py --save-kinds distorted,restored,panel,beam
```

`--save-every 0` 이면 이미지 디렉토리가 아예 안 생긴다. `detections.csv` 는 항상
전 프레임을 담는다.

`evaluate.py` 는 대신 `report.json` + `per_class_*.csv` + `per_image_*.csv` 를,
`train.py` 는 `runs/` 아래에 `restorer_<tag>_best.pt` + `loss_log.csv` +
`loss_plots.png` 를 쓴다.

---

[English](README_running.md) · [← README](README.ko.md)

# 실행 가이드

루트 엔트리포인트 3개의 옵션·입력·출력. 설치와 인자 없이 바로 도는 명령은
[README.ko.md](README.ko.md) 참고.

`collect.py`와 `record.py`는 `data/` 아래에 있고,
[data/README_data.ko.md](data/README_data.ko.md)에서 다룬다.

- [공용 플래그](#공용-플래그)
- [`demo.py` — 복원 → 검출](#demopy--복원--검출)
- [`demo.py --live` — 웹캠 + 프로젝터](#demopy---live--웹캠--프로젝터)
- [`evaluate.py` — 복원 전후 점수 비교](#evaluatepy--복원-전후-점수-비교)
- [`train.py` — 복원기 재학습](#trainpy--복원기-재학습)
- [출력물 형식](#출력물-형식)
- [복원 속도 줄이기](#복원-속도-줄이기)

---

## 공용 플래그

세 스크립트 모두 받는다. 전체 목록: `python <script>.py --help`.

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--restorer <name>` | `naf_se_unet` | 복원 백엔드. `@register_restorer`로 등록한 아무 이름 |
| `--restorer-weights <path>` | `restoration.yaml` | 복원 체크포인트 |
| `--device cuda\|cpu` | 있으면 cuda | — |
| `--input-size W H` | `640 360` | 복원기가 돌아가는 해상도 |
| `--restoration-config <yaml>` | — | `configs/restoration.yaml` 위에 병합 |
| `--detection-config <yaml>` | — | `configs/detection.yaml` 위에 병합 |

검출 플래그는 `demo.py`와 `evaluate.py`만 받는다. `train.py`는 복원망만 학습하므로 없다.

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--detector <name>` | `yolo` | `yolo` · `ssd` · `none`, 또는 `@register_detector`로 등록한 아무 백엔드 |
| `--det-weights <path>` | 백엔드별, `detection.yaml` | 검출기 체크포인트 |
| `--conf <float>` | `0.25` | 검출 신뢰도 하한 |

클래스명은 `configs/detection.yaml`의 `names:`에서 온다. 바꾸려면 `--detection-config`로 작은
YAML을 넘기면 된다. YOLO 체크포인트가 자체 names를 가지고 있으면 그쪽이 이긴다.

`--no-*` ablation 플래그 10개는 `train.py`만 받는다. [Ablation](#ablation) 참고. 추론
스크립트 둘은 필요 없다 — 체크포인트가 학습 당시 아키텍처를 품고 있다.

---

## `demo.py` — 복원 → 검출

복원과 검출만 한다. GT를 읽지 않고 채점도 하지 않는다. 그건 `evaluate.py`의 일이다.

| | 기본값 | 변경 |
|---|---|---|
| 입력 | `data/sample_input/` (`distorted/` + `light/`) | `--input <dir>` |
| 출력 | `output/<timestamp>/` | `--output <dir>` · `--name <name>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--limit N` | `0` | 처리할 쌍 수 상한. `0` = 전부 |
| `--save-every N` | `1` | 이미지 저장 간격. `0`이면 csv만 |
| `--save-kinds a,b` | 셋 다 | `distorted`, `restored`, `panel` 중 일부 |
| `--video` | off | 2×2 패널을 `result.mp4`로도 기록 |

이미지는 JPEG 품질 92로 저장된다. 박스 크기 게이트는 `configs/detection.yaml`의
`detector.min_width` / `min_height` / `min_area`에서 온다. 게이트를 통과한 박스는 전부
유지된다 — 지표 계산이 중복 박스를 잃으면 안 되기 때문이다.

```bash
python demo.py --detector ssd --conf 0.4
python demo.py --detector none --save-kinds restored
python demo.py --input /path/to/pairs --name my_run
```

`--save-every 0`은 이미지를 전혀 쓰지 않는다. csv와 요약만 필요할 때 유용하다. 예를 들어
검출기를 훑을 때:

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

---

## `demo.py --live` — 웹캠 + 프로젝터

실제 하드웨어가 필요하다. 프로젝터가 쏘는 스크린을 향한 웹캠.

| | 기본값 | 변경 |
|---|---|---|
| 투사 클립 | `data/live/BeamVideo.mp4` | `--clip <path>` |
| 캘리브레이션 배경 | `data/live/BaseBackGround.jpg` | `--background <path>` |
| 카메라 | 0번, 1280×960 @30fps | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| 출력 | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <name>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--screen N` | `1` | 프로젝터가 붙은 모니터 인덱스. `0` = 주 디스플레이 |
| `--offset N` | `6` | 프로젝터→카메라 지연, 프레임 단위 |
| `--analyse-every N` | `0` | N프레임마다 복원+검출. `0` = 측정 후 자동 결정 |
| `--max-frames N` | `0` | `0` = 클립 끝까지 |
| `--manual-calib` | off | 자동 검출 대신 네 코너를 직접 클릭 |
| `--debug-view` | off | 워프 전 카메라 피드 + 사각형을 실시간 표시 |
| `--calib-settle F` | `0.8` | 캘리브레이션 플래시 후 대기 초 |
| `--cam-backend` | `auto` | `auto` · `any` · `dshow` · `msmf` · `v4l2` |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

`Combined_View` 창에서 `q`를 누르면 중지된다.

Windows가 아니면 `screeninfo`를 위해 `pip install -e ".[live]"`가 필요하다. Windows는
Win32 API로 모니터 배치를 처리한다.

### 재생과 분석은 다른 속도로 돈다

프로젝터는 모델이 얼마나 느리든 클립의 원래 fps로 재생한다. 복원+검출은 워커 스레드에서
균등 간격 부분집합 — N프레임마다 — 만 처리한다. 기록된 패널은 클립처럼 재생되지만 프레임
레이트만 낮다. 사이의 프레임들도 투사·촬영은 되고 채점만 되지 않는다.

`--analyse-every 0`은 처음 12개 분석 프레임의 시간을 재고, CUDA 워밍업 이상치를 버리고,
중앙값에서 N을 고정한다. 실행 중 재조정하지 않는다. N을 바꾸는 것 자체가 분석 영상을
불균등하게 만들기 때문이다.

요약에 두 속도가 모두 나온다:

```
projector 29.1 fps (450 frames) | analysis 13.0 fps (every 2 frame(s): 201 analysed,
249 skipped, 0 dropped)
```

`skipped`는 설계상 정상이다. 애초에 모델에 줄 프레임이 아니었다. `dropped`는 워커가 데드라인을
놓친 것이고, 봐야 할 숫자는 이쪽이다.

자동값보다 촘촘하게 가면 균등성을 잃는다. 30fps 클립, 워커가 프레임당 ~41ms일 때:

| `--analyse-every` | 프로젝터 | 분석 | 분석된 프레임 | 간격 균등도 |
|---|---|---|---|---|
| `1` | 28.5 fps | 21.8 fps | 76% | 89% |
| `2` (여기서 자동값) | 28.8 fps | 13.9 fps | 48% | 100% |
| `3` | 28.8 fps | 9.4 fps | 32% | 100% |

`1`은 1.6배 많은 프레임을 분석하고 재생에는 거의 영향이 없다. 다만 모든 슬롯을 맞출 수는
없다. 30fps 예산은 33ms인데 워커는 41ms가 필요하다. 패널의 매끄러움보다 커버리지가 중요할 때
고른다.

### 캘리브레이션

흑/백 플래시 → 두 카메라 샷을 차분하면 환경광은 상쇄되고 투사만 남는다 → 최대 4각 컨투어가
스크린 → homography 하나를 계산해 모든 프레임에 재사용.

**루프 시작 전 한 번만 수행된다.** 실행 중 카메라나 프로젝터가 움직이면 남은 세션 전체의
워프가 틀어진다. `--debug-view`로 감시할 것. 자동 검출이 실패하면 수동 클릭으로 넘어가고,
`--manual-calib`은 처음부터 수동이다.

`output/<run>/calib/`는 검출이 실패했을 때도 기록된다. 오히려 그때가 중요하다.

| 파일 | 볼 것 |
|---|---|
| `quad.jpg` | 네 점이 스크린 코너에 정확히 있는가? |
| `mask.jpg` | 흰 영역이 스크린만인가, 조명/창문이 섞였는가? |
| `diff.jpg` | 플래시 차분이 충분히 강한가? 아니면 `--calib-settle`을 올린다 |
| `warped.jpg` | 정류 결과가 실제로 직사각형인가? |
| `frame_pre.jpg` | 실행의 첫 카메라 프레임, 원본 |
| `frame_post.jpg` | 그 프레임을 모델 입력 크기로 정류한 것 |
| `frame_compare.jpg` | 둘을 나란히. `Warp_FirstFrame` 창에도 한 번 표시됨 |

앞 네 개는 루프 *전* 플래시에서 나온다. `frame_*` 세 개는 실행 자체에서 나오므로, 프레임이
실제로 정류된 워프를 보여준다. 그 사이에 투사가 흔들렸다면 여기서 드러난다.

### 장시간 무인 녹화

```bash
python demo.py --live --screen 2 --save-every 300 --save-kinds panel
```

csv는 모든 프레임을 덮고 이미지만 드물게 떨어져서, 디스크 사용량이 예측 가능해진다.
`--save-every 0`이면 이미지를 아예 쓰지 않는다.

---

## `evaluate.py` — 복원 전후 점수 비교

`surface`과 `label`을 **둘 다** 가진 샘플만 채점한다.

| | 기본값 | 변경 |
|---|---|---|
| 입력 | `data/sample_input/` (`distorted/` + `light/`) | `--input <dir>` |
| GT | `data/sample_input/` (`surface/` + `labels/`) | `--gt <dir>` |
| 출력 | `output/Eval_<입력 데이터셋>/` | `--output <dir>` · `--name <name>` |

`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv`를 쓴다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--detector yolo,ssd` | `yolo` | 여기서는 콤마 구분. 여러 백엔드를 한 번에 비교, 각각 한 행 |
| `--iou <float>` | `0.5` | TP 판정 IoU 임계값 |
| `--limit N` | `0` | 채점할 쌍 수 상한 |

```bash
python evaluate.py --detector yolo,ssd --iou 0.5
```

리포트 디렉터리는 입력 데이터셋 이름을 따른다. `data/sample_input`은
`output/Eval_sample_input/`으로 들어간다. 같은 데이터셋을 다시 돌리면 덮어쓰고, 이전 실행의
백엔드별 csv는 먼저 정리된다.

요약 표가 출력되고, 클래스별 P / R / F1 / AP는 csv에 들어간다.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_input/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # 클래스별 AP 변화
```

여기서 `mAP`은 단일 IoU 임계값에서 클래스 평균 AP다 — VOC 방식, 보간된 PR 곡선 아래 면적.
COCO의 IoU 평균 지표가 아니다.

> `--det-weights`는 체크포인트 하나를 가리키므로 백엔드 하나에만 속할 수 있다.
> `--detector a,b`처럼 여러 개를 비교할 때는 경고 후 무시되고, 모든 백엔드가
> `configs/detection.yaml`로 폴백한다.

---

## `train.py` — 복원기 재학습

학습 extra 필요: `pip install -e ".[train]"`. `distorted` / `light` / `surface` 삼중쌍이 완성된
것만 사용한다.

| | 기본값 | 변경 |
|---|---|---|
| 데이터 | `configs/restoration.yaml`의 `train.data` | `--data-root <dir>`, 또는 `train.data` 수정 |
| 출력 | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/` | `--out <dir>` |

`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png`를 쓴다.

### 학습 데이터가 오는 곳

세 디렉터리는 하드코딩이 아니라 설정값이다. 각 항목은 디렉터리, glob, 또는 리스트를 받는다.
실제 촬영본은 날짜로 분할되고 light 프레임은 보통 따로 있는 루트에 놓이기 때문이다.

```yaml
train:
  data:
    distorted: "D:/captures/WarpData_*_pro"
    surface:   "D:/captures/WarpData_*_ori"
    light:     "D:/captures/Learning_video_frames"
```

결정 순서:

```
--data-root   >   train.data   >   data/sample_input
```

`--data-root`는 세 역할을 모두 담은 폴더 하나를 가리키며 무조건 이긴다. `collect.py`로 만든
세션은 이것만 주면 된다. 주지 않으면 설정된 세 디렉터리를 쓰는데, 그 경로들은 그대로
glob되므로 상대 경로가 프로젝트 루트가 아니라 **작업 디렉터리** 기준으로 읽힌다. 저장소
루트에서 실행하거나 절대 경로를 쓸 것.

```bash
python train.py --epochs 30                        # train.data 사용
python train.py --data-root data/collected_0803    # 폴더 하나, 설정은 무시됨
```

`--data-root`는 레이아웃을 자동 감지하고, surface 타깃을 이 순서로 찾아 존재하는 쪽을
바로 읽는다. 심볼릭 링크나 복사는 필요 없다:

```
--data-root/OriginalImage/  →  --data-root/surface/  →  --gt/surface/
                            →  --data-root/clean/    →  --gt/clean/   (pre-rename)
```

짝이 없는 `distorted`는 치명적 오류가 아니라 집계 후 스킵된다. 실행 시 몇 개가 어느 쪽으로 갔는지
출력된다.

```
data: 10 triplets of 22 distorted image(s) from distorted=data/sample_input/distorted
      skipped 0 without a light, 12 without a surface
```

### 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--epochs N` | `30` | `configs/restoration.yaml`에서 |
| `--batch-size N` | `4` | 〃 |
| `--accum-steps N` | `4` | 그래디언트 누적 구간 |
| `--lr F` | `0.0002` | 〃 |
| `--sample N` | `0` | 사용할 삼중쌍 수 상한 |
| `--num-workers N` | `4`, batch > 4면 `8` | DataLoader 워커 |
| `--resume <ckpt>` | — | 체크포인트에서 이어서. 그 아키텍처가 `--no-*`보다 우선 |
| `--seed N` | `42` | — |
| `--no-amp` | off | CUDA에서 mixed precision 끄기 |
| `--save-every N` | `1` | `epoch_N.pt` 저장 간격 (에폭) |

```bash
python train.py --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

손실은 `L1 + perceptual + SSIM + wavelet`이고 가중치는 `train.loss`에서 온다. 각 삼중쌍은
360×640으로 리사이즈된 뒤 180×320으로 랜덤 크롭된다. 손실이 무엇을 측정하고 왜 그런지:
[weights/README_weights.ko.md](weights/README_weights.ko.md#residual-규약).

### Ablation

구조 요소 10개를 개별로 끌 수 있다. 꺼진 것이 `tag`에 들어가고, 그 tag가 실행 폴더명과
체크포인트 파일명에 모두 반영된다.

| 플래그 | 끄는 것 | Tag |
|---|---|---|
| `--no-prenorm` | NAFSEBlock의 pre-LayerNorm | `NoPre` |
| `--no-naf-norm` | NAFBlock 내부 LayerNorm2d | `NoNorm` |
| `--no-simple-gate` | SimpleGate (`x1*x2`) → GELU로 대체 | `NoGate` |
| `--no-naf-scale` | 학습 가능한 residual 스케일 beta / gamma | `NoScale` |
| `--no-ca` | 채널 어텐션. 블록이 순수 NAFBlock이 됨 | `NoCA` |
| `--no-skip1` | U-Net skip enc1 → dec1 (원해상도) | `NoSkip1` |
| `--no-skip2` | U-Net skip enc2 → dec2 (1/2) | `NoSkip2` |
| `--no-skip3` | U-Net skip enc3 → dec3 (1/4) | `NoSkip3` |
| `--no-bottleneck` | 1/8 해상도 bottleneck → Identity | `NoBott` |
| `--no-tanh` | 출력 tanh. residual이 무한 범위가 됨 | `NoTanh` |

여러 개는 결합된다: `NoCA-NoSkip3`. 하나도 끄지 않으면 `FULL`.

용량도 덮을 수 있다: `--base-dim` (48), `--enc-depth` (`2,2,3`), `--dec-depth` (`2,2,2`),
`--bottleneck-depth` (2), `--dw-expand` (2), `--ffn-expand` (2), `--ca-reduction` (16).

```bash
python train.py --no-ca --epochs 30
```

체크포인트가 자신의 아키텍처 config를 품고 있어서 플래그를 다시 쓸 필요가 없다. `demo.py`와
`evaluate.py`는 아예 받지도 않는다:

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

예외는 config 없이 순수 `state_dict`로 저장된 *레거시* 체크포인트다. 이 경우 기본(`FULL`)
아키텍처로 폴백하므로, ablation된 레거시 가중치는 `--restoration-config` YAML의 `ablation:`
블록으로 구조를 알려줘야 한다. `train.py`가 쓰는 체크포인트는 전부 config를 품으므로 외부에서
받은 가중치에만 해당한다.

---

## 출력물 형식

`demo.py`는 실행당 디렉터리 하나를 쓴다:

```
output/<run_name>/
├── run_meta.json      설정 · 환경 · 캘리브레이션 · 요약, 전부 한 파일
├── detections.csv     박스 1개당 1행. `source`가 distorted / restored 구분
├── captures/          박스를 그리지 않은 원본 픽셀
│   ├── <id>_distorted.jpg      복원 전
│   └── <id>_restored.jpg       복원 후
├── frames_all/        2×2 비교 figure
│   └── <id>_panel.jpg          light · distorted+box · restored+box · residual
├── calib/             --live 전용
└── result.mp4         2×2 패널 영상 (--live 또는 --video)
```

`captures/`가 박스 없는 이유: 나중에 `evaluate.py`로 이 복원 결과를 채점하거나, 동일 픽셀에
다른 검출기를 다시 돌릴 수 있어야 한다. jpg에 태운 박스는 되돌릴 수 없다.

저장되는 종류는 세 개뿐이다. 박스가 그려진 뷰, residual 히트맵, beam은 이미 패널의 타일이라
따로 다시 쓰는 것은 프레임당 인코딩 4번을 더 쓰면서 얻는 게 없었다. `--save-kinds`로 더
줄인다:

```bash
python demo.py --save-kinds panel                # 비교 figure만
python demo.py --save-kinds distorted,restored   # 재채점 가능한 픽셀만
```

`--save-every 0`은 이미지 디렉터리를 아예 만들지 않는다. `detections.csv`는 여전히 모든
프레임을 덮는다.

`evaluate.py`는 대신 `report.json` + `per_class_*.csv` + `per_image_*.csv`를 쓴다.
`train.py`는 `runs/` 아래에 `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png`를
쓴다.

---

## 복원 속도 줄이기

복원이 라이브 프레임의 약 46%다. 그래서 가장 먼저 줄일 대상이다. 네트워크가 fully
convolutional이라 작동 해상도가 런타임 노브가 된다 — 재학습 없이. 번들 데이터셋,
`--detector yolo` 기준 측정:

| `--input-size` | 복원 | 검출 mAP | PSNR 이득 | SSIM 이득 |
|---|---|---|---|---|
| `320 180` | 9.7 ms | 0.9866 | +8.95 dB | +0.163 |
| `480 270` | 13.4 ms | **1.0000** | +11.32 dB | +0.183 |
| `640 360` (기본) | 20.5 ms | **1.0000** | **+13.17 dB** | **+0.218** |
| `854 480` | 41.2 ms | 1.0000 | +9.89 dB | +0.145 |

```bash
python demo.py --live --screen 2 --input-size 480 270
```

`480 270`은 복원 시간의 1/3만 쓰고 검출은 차이를 느끼지 못한다. PSNR/SSIM 이득만 줄어든다.
그 아래로 가면 mAP가 흔들리기 시작한다.

640×360 **위로** 가면 양쪽 다 나빠진다. 체크포인트는 360×640에서 리사이즈한 180×320 크롭으로
학습됐고, 854×480은 그 스케일에서 충분히 벗어나 복원 품질이 떨어지면서 시간은 두 배가 된다.

mixed precision과 `torch.compile`은 손댈 가치가 없다. 그래서 둘 다 연결하지 않았다. fp16
autocast는 2% 더 *느리게* 측정됐다 — 네트워크가 memory-bound일 만큼 작아서 autocast가
절약분보다 더 든다. compile은 Windows가 제공하지 않는 Triton 빌드를 요구한다.

진짜 더 작은 네트워크는 재학습이 필요하다. 640×360 기준:

| `train.py` 플래그 | 파라미터 | forward |
|---|---|---|
| 기본 | 4,184,259 | 13.8 ms |
| `--base-dim 32` | 1,878,659 | 9.7 ms |
| `--enc-depth 1,1,1 --dec-depth 1,1,1 --bottleneck-depth 1` | 2,418,837 | 7.3 ms |
| `--no-ca` | 4,116,147 | 13.1 ms |

블록을 얕게 하는 쪽이 좁게 하는 쪽보다 이득이 크다. 채널 어텐션 제거는 5%를 벌지만 재학습할
가치는 없다.

---

[English](README_running.md) · [← README](README.ko.md)

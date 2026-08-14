# 실행 가이드

루트 엔트리포인트 3개의 옵션·입력·출력. 설치와 인자 없이 바로 도는 명령은
[README.ko.md](README.ko.md) 참고.

데이터셋 수집은 저장소 루트의 `Data.py`가 담당하며 `data/` 아래 스크립트로 분기한다.
[data/README_data.ko.md](data/README_data.ko.md)에서 다룬다.

- [공용 플래그](#공용-플래그)
- [`demo.py` — 복원 → 검출](#demopy--복원--검출)
- [`demo.py --live` — 웹캠 + 프로젝터](#demopy---live--웹캠--프로젝터)
- [`evaluate.py` — 복원 전후 점수 비교](#evaluatepy--복원-전후-점수-비교)
- [`train.py` — 복원기 재학습](#trainpy--복원기-재학습)
- [출력물 형식](#출력물-형식)
- [복원 속도 줄이기](#복원-속도-줄이기)

---

## 설정 파일이 가진 것

**모델을 지정하는 플래그는 없다.** 어떤 복원기가 도는지, 어떤 검출기가 도는지, 어떤
체크포인트를 어느 신뢰도로 쓰는지 — 전부 `projector_distortion/configs/*.yaml`에서
읽는다. 예전에는 항목마다 플래그가 있었지만 실행 사이에 실제로 바뀌는 값이 아니었고,
합치면 `--help`를 읽을 수 없게 만들었다.

| 설정 | 위치 |
|---|---|
| 라이브 리그: 모니터·웹캠·프로젝터→카메라 지연 | `live.yaml`의 `rig:` |
| 수집 리그, 세션 폴더, 워프 기하 | `collect.yaml` — `Data.py`가 읽고, 이 셋은 읽지 않는다 |
| 복원 백엔드 | `restoration.yaml`의 `model.backend` |
| 복원 체크포인트 | `restoration.yaml`의 `model.weights` |
| 복원기가 돌아가는 해상도 | `restoration.yaml`의 `model.input_size` |
| 아키텍처 / ablation / 용량 | `restoration.yaml`의 `ablation:` — [Ablation](#ablation) 참고 |
| 학습 에폭·배치·lr·누적·손실 가중치 | `restoration.yaml`의 `train:` |
| 검출 백엔드 | `detection.yaml`의 `detector.backend` — `yolo` · `ssd` · `none`, 또는 `@register_detector`로 등록한 이름. 리스트로 주면 `evaluate.py`가 백엔드당 한 행 |
| 검출 체크포인트 | `detection.yaml`의 `weights.<backend>` |
| 검출 신뢰도 하한 | `detection.yaml`의 `detector.conf` |
| 클래스명 | `detection.yaml`의 `names:` — YOLO 체크포인트가 자체 names를 가지면 그쪽이 이긴다 |
| 박스 크기 게이트 | `detection.yaml`의 `detector.min_width` / `min_height` / `min_area` |
| 검출기 추론 해상도 | `detection.yaml`의 `detector.imgsz` |

추론 스크립트는 애초에 아키텍처 설정이 필요 없다 — 체크포인트가 학습 당시 아키텍처를
품고 있다.

남은 플래그는 눈앞의 실행에 관한 것들이다 — 어떤 데이터, 출력 위치, 얼마나 처리할지,
라이브 리그라면 어떤 하드웨어. 전체 목록: `python <script>.py --help`.
`--device cuda|cpu`는 `demo.py`와 `train.py`에 있고, `evaluate.py`는 항상 자동
결정한다(있으면 cuda).

---

## `demo.py` — 복원 → 검출

복원과 검출만 한다. GT를 읽지 않고 채점도 하지 않는다. 그건 `evaluate.py`의 일이다.

| | 기본값 | 변경 |
|---|---|---|
| 입력 | `data/SampleData/sample_eval/` (`distorted/` + `light/`) | `--input <dir>` |
| 출력 | `output/<timestamp>/` | `--output <dir>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--limit N` | `0` | 처리할 쌍 수 상한. `0` = 전부 |
| `--save-every N` | `1` | 이미지 저장 간격. `0`이면 csv만 |
| `--video` | off | 2×2 패널을 `result.mp4`로도 기록 |
| `--device cuda\|cpu` | 있으면 cuda | — |

이미지는 JPEG 품질 92로 저장된다. 박스 크기 게이트는 `configs/detection.yaml`의
`detector.min_width` / `min_height` / `min_area`에서 온다. 게이트를 통과한 박스는 전부
유지된다 — 지표 계산이 중복 박스를 잃으면 안 되기 때문이다.

```bash
python demo.py --input /path/to/pairs
python demo.py --limit 5 --save-every 0
```

검출 없이 복원만 하려면 `configs/detection.yaml`에 `detector.backend: none`을 넣는다.
검출기 비교는 `evaluate.py`의 일이다 — 거기서 백엔드 리스트가 백엔드당 한 행이 된다.

`--save-every 0`은 이미지를 전혀 쓰지 않는다. csv와 요약만 필요할 때 그렇게 한다.
실행마다 `output/<timestamp>/`가 따로 생기므로 반복 실행이 서로 덮어쓰지 않는다.

---

## `demo.py --live` — 웹캠 + 프로젝터

실제 하드웨어가 필요하다. 프로젝터가 쏘는 스크린을 향한 웹캠.

### 리그는 플래그가 아니라 설정이다

어느 모니터인지, 어느 웹캠인지, 카메라가 얼마나 늦는지는 물리적 셋업의 속성이라
[configs/live.yaml](projector_distortion/configs/live.yaml)에 있다. 머신마다 한 번 설정:

```yaml
rig:
  screen: 1               # 프로젝터가 붙은 모니터; 0 = 주 디스플레이
  camera: 0               # 웹캠 인덱스
  cam_backend: auto       # auto | any | dshow | msmf | v4l2
  offset: 6               # 프로젝터 → 카메라 지연, 프레임 단위
```

측정할 가치가 있는 건 `offset`이다. light 프레임 N을 담은 카메라 프레임은 `offset`
프레임 뒤에 도착하므로, 촬영본을 그 원인이 된 프레임에 맞추는 값이 이것이다. 낮으면
모델에 이전 프레임의 light가, 높으면 미래 프레임이 들어간다. `--debug-view`로 한 번 재면
된다.

카메라에 요청하는 해상도와 프레임레이트는 1280×960 @30fps로 고정이다
(`projector_distortion/pipeline/live.py`의 `CAM_WIDTH` / `CAM_HEIGHT` / `CAM_FPS`).
드라이버가 요청을 무시하는 일이 흔해서, 시작 줄에 카메라가 실제로 열린 값을 찍는다.

`Data.py`와 `data/` 아래 스크립트는 같은 종류의 설정을
[configs/collect.yaml](projector_distortion/configs/collect.yaml)의 `session:` / `light:` /
`capture:` / `warp:` / `record:`에서 읽는다. 파일이 둘인 이유는 수집 스크립트가 torch 없이
돌고 파이프라인 패키지를 import하지 않기 때문이다.

### 플래그

| | 기본값 | 변경 |
|---|---|---|
| 투사 클립 | `data/live/test_light.mp4` — 홀드아웃 원본 | `--clip <path>` |
| 캘리브레이션 배경 | `data/live/BaseBackGround.jpg` | 고정 (`projector_distortion/cli.py`의 `DEFAULT_LIVE_BG`) |
| 출력 | `output/<timestamp>/` + `calib/` + `result.mp4` | `--output <dir>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--manual-calib` | off | 자동 검출 대신 네 코너를 직접 클릭 |
| `--debug-view` | off | 워프 전 카메라 피드 + 사각형을 실시간 표시 |

실행은 클립 끝까지 가고, 분석 간격은 항상 지정이 아니라 측정으로 정해진다 — 아래 참고.

```bash
python demo.py --live
python demo.py --live --save-every 30 --debug-view
```

`Combined_View` 창에서 `q`를 누르면 중지된다.

Windows가 아니면 `screeninfo`를 위해 `pip install -e ".[live]"`가 필요하다. Windows는
Win32 API로 모니터 배치를 처리한다.

### 재생과 분석은 다른 속도로 돈다

프로젝터는 모델이 얼마나 느리든 클립의 원래 fps로 재생한다. 복원+검출은 워커 스레드에서
균등 간격 부분집합 — N프레임마다 — 만 처리한다. 기록된 패널은 클립처럼 재생되지만 프레임
레이트만 낮다. 사이의 프레임들도 투사·촬영은 되고 채점만 되지 않는다.

N은 지정이 아니라 측정으로 정해진다. 처음 12개 분석 프레임의 시간을 재고, CUDA 워밍업
이상치를 버리고, 중앙값에서 N을 고정한다 — 완전히 균등한 간격을 유지하는 가장 촘촘한
값이다. 실행 중 재조정하지 않는다. N을 바꾸는 것 자체가 분석 영상을 불균등하게 만들기
때문이다.

요약에 두 속도가 모두 나온다:

```
projector 29.1 fps (450 frames) | analysis 13.0 fps (every 2 frame(s): 201 analysed,
249 skipped, 0 dropped)
```

`skipped`는 설계상 정상이다. 애초에 모델에 줄 프레임이 아니었다. `dropped`는 워커가 데드라인을
놓친 것이고, 봐야 할 숫자는 이쪽이다.

측정값보다 촘촘하게 가면 균등성을 잃는다. 노브로 열어두지 않은 이유다. 30fps 클립, 워커가
프레임당 ~41ms일 때:

| 간격 | 프로젝터 | 분석 | 분석된 프레임 | 간격 균등도 |
|---|---|---|---|---|
| `1` | 28.5 fps | 21.8 fps | 76% | 89% |
| `2` (여기서 측정값) | 28.8 fps | 13.9 fps | 48% | 100% |
| `3` | 28.8 fps | 9.4 fps | 32% | 100% |

`1`은 1.6배 많은 프레임을 분석하고 재생에는 거의 영향이 없지만 모든 슬롯을 맞출 수는 없다.
30fps 예산은 33ms인데 워커는 41ms가 필요하다. 실행은 `2`를 고른다 — 정확히 균등하게
유지되는 가장 촘촘한 간격. 불균등하게 샘플링된 패널이 더 성긴 패널보다 읽기 어렵기 때문이다.

### 캘리브레이션

흑/백 플래시 → 두 카메라 샷을 차분하면 환경광은 상쇄되고 투사만 남는다 → 최대 4각 컨투어가
스크린 → homography 하나를 계산해 모든 프레임에 재사용.

**루프 시작 전 한 번만 수행된다.** 실행 중 카메라나 프로젝터가 움직이면 남은 세션 전체의
워프가 틀어진다. `--debug-view`로 감시할 것.

#### 실행이 멈추고 워프를 보여준다

한 번 추정해서 계속 재사용하므로, 코너 하나가 틀어진 사각형은 모든 프레임을 조용히 망친다 —
패널은 그럴듯해 보이고 residual만 엉망이 된다. 그래서 정류된 첫 프레임을, 검출된 사각형을
그린 원본 카메라 뷰와 나란히 띄우고 대기한다:

```
    auto calibration -> warp target 968x545:
      TL (   241,    118)
      TR (  1183,    131)
      BR (  1176,    701)
      BL (   233,    688)
      covers 41% of the 1280x960 camera frame
      edges  top 942  right 570  bottom 943  left 570 px
    [calibration] enter/a accept | m click the corners | r re-detect | q quit
```

| 키 | 동작 |
|---|---|
| `enter` · `space` · `a` | 수락하고 실행 시작 |
| `m` | 라이브 피드에서 네 코너를 직접 클릭. 순서 무관 — 자동 정렬된다 |
| `r` | 흑/백 플래시를 다시 쏘고 재검출. 조명이나 반사를 없앤 뒤 시도할 만하다 |
| `q` | 실행 중단 |

코너 아래 두 숫자를 볼 것. **covers**는 스크린이 실제로 차지하는 비율과 대략 맞아야 하고,
마주보는 **edges**는 서로 비슷해야 한다. 창문이나 조명을 잡은 사각형은 그림을 보기도 전에
둘 중 하나가 크게 어긋나는 것으로 드러난다.

자동 검출이 아무것도 못 찾으면 프리뷰도 수락도 없다 — `m`, `r`, `q`만 동작한다.
`--manual-calib`은 바로 클릭으로 넘어가고, 그 뒤에도 리뷰는 그대로 뜬다.

무인 실행처럼 키를 누를 사람이 없으면
[configs/live.yaml](projector_distortion/configs/live.yaml)에
`rig.review_calibration: false`를 넣는다. 그러면 자동 검출 실패 시 예전처럼 수동 클릭으로
넘어간다.

`output/<run>/calib/`는 검출이 실패했을 때도 기록된다. 오히려 그때가 중요하다.

| 파일 | 볼 것 |
|---|---|
| `quad.jpg` | 네 점이 스크린 코너에 정확히 있는가? |
| `mask.jpg` | 흰 영역이 스크린만인가, 조명/창문이 섞였는가? |
| `diff.jpg` | 플래시 차분이 충분히 강한가? 아니면 `pipeline/live.py`의 `CALIB_SETTLE`을 올린다 |
| `warped.jpg` | 정류 결과가 실제로 직사각형인가? |
| `frame_pre.jpg` | 실행의 첫 카메라 프레임, 원본 |
| `frame_post.jpg` | 그 프레임을 모델 입력 크기로 정류한 것 |
| `frame_compare.jpg` | 둘을 나란히. `Warp_FirstFrame` 창에도 한 번 표시됨 |

앞 네 개는 루프 *전* 플래시에서 나온다. `frame_*` 세 개는 실행 자체에서 나오므로, 프레임이
실제로 정류된 워프를 보여준다. 그 사이에 투사가 흔들렸다면 여기서 드러난다.

### 장시간 무인 녹화

```bash
python demo.py --live --save-every 300
```

csv는 모든 프레임을 덮고 이미지만 드물게 떨어져서, 디스크 사용량이 예측 가능해진다.
`--save-every 0`이면 이미지를 아예 쓰지 않는다.

---

## `evaluate.py` — 복원 전후 점수 비교

`surface`과 `label`을 **둘 다** 가진 샘플만 채점한다.

| | 기본값 | 변경 |
|---|---|---|
| 입력 | `data/SampleData/sample_eval/` (`distorted/` + `light/`) | `--input <dir>` |
| GT | `data/SampleData/sample_eval/` (`surface/` + `labels/`) | `--gt <dir>` |
| 출력 | `output/Eval_<입력 데이터셋>/` | `--output <dir>` |

`labels/`에는 YOLO `.txt`나 LabelMe `.json`이 들어갈 수 있고 확장자로 리더가 정해지므로,
홀드아웃 split은 경로만 주면 된다. LabelMe는 클래스를 *이름*으로 저장하며, 그 이름은 탐지기
자신의 클래스 목록과 대조된다 — [data/README_data.ko.md](data/README_data.ko.md#라벨-포맷) 참조.

```bash
python evaluate.py --input data/SampleData/sample_test --gt data/SampleData/sample_test
```

`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv`를 쓴다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--iou <float>` | `0.5` | TP 판정 IoU 임계값 |
| `--limit N` | `0` | 채점할 쌍 수 상한 |

`evaluate.py`에는 모델 플래그가 하나도 없다 — `--device`조차 없다. 어떤 백엔드를 채점할지는
`configs/detection.yaml`의 `detector.backend`에서 오고, 거기에 리스트를 주면 같은 리포트에
백엔드당 한 행이 들어간다:

```yaml
detector:
  backend: [yolo, ssd]
```

```bash
python evaluate.py --iou 0.5
```

리포트 디렉터리는 입력 데이터셋 이름을 따른다. `data/SampleData/sample_eval`은
`output/Eval_sample_eval/`로, 테스트 split은 `output/Eval_sample_test/`로 들어간다. 같은
데이터셋을 다시 돌리면 덮어쓰고, 이전 실행의 백엔드별 csv는 먼저 정리된다.

요약 표가 출력되고, 클래스별 P / R / F1 / AP는 csv에 들어간다.

```python
import pandas as pd
pc = pd.read_csv("output/Eval_sample_eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # 클래스별 AP 변화
```

여기서 `mAP`은 단일 IoU 임계값에서 클래스 평균 AP다 — VOC 방식, 보간된 PR 곡선 아래 면적.
COCO의 IoU 평균 지표가 아니다.

> 리스트의 각 백엔드는 `weights.<backend>`에서 자기 체크포인트를 읽는다. 그래서
> `[yolo, ssd]` 비교에서 한쪽 가중치가 다른 쪽으로 들어가는 일이 없다.

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
--data-root   >   train.data   >   data/SampleData/sample_train
```

`train.data`는 라벨이 없는 유일한 split인 `sample_train`을 가리킨 채로 배포된다 — 학습에는
라벨이 필요 없기 때문이다. `sample_eval`과 `sample_test`는 `evaluate.py`용 홀드아웃이므로,
`train.data`를 그쪽으로 돌리면 학습한 데이터로 채점하게 된다.

`--data-root`는 세 역할을 모두 담은 폴더 하나를 가리키며 무조건 이긴다. `Data.py`로 만든
세션은 `warp_<MMDD>/` 폴더만 주면 된다. 주지 않으면 설정된 세 디렉터리를 쓰는데, 그 경로들은
그대로 glob되므로 상대 경로가 프로젝트 루트가 아니라 **작업 디렉터리** 기준으로 읽힌다.
저장소 루트에서 실행하거나 절대 경로를 쓸 것.

```bash
python train.py --epochs 30                              # train.data 사용
python train.py --data-root data/Create_Data/warp_0813   # 폴더 하나, 설정은 무시됨
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
data: 994 triplets of 1,000 distorted image(s) from distorted=data/SampleData/sample_train/distorted
      skipped 6 without a light, 0 without a surface
```

### 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--epochs N` | `30` | `configs/restoration.yaml`의 `train.epochs` |
| `--batch-size N` | `4` | `train.batch_size` |
| `--lr F` | `0.0002` | `train.lr` |
| `--sample N` | `0` | 사용할 삼중쌍 수 상한 |
| `--resume <ckpt>` | — | 체크포인트에서 이어서. 그 아키텍처가 `ablation:`보다 우선 |
| `--no-amp` | off | CUDA에서 mixed precision 끄기 |
| `--device cuda\|cpu` | 있으면 cuda | — |

플래그가 아니라 고정된 값: 그래디언트 누적 구간은 YAML의 `train.accum_steps`, 시드는 `42`
(`train.py`의 `SEED`), DataLoader 워커는 `4` (batch > 4면 `8`), `epoch_N.pt`는 매 에폭 저장.

```bash
python train.py --epochs 30
python train.py --resume runs/0730_1948_30ep_FULL/restorer_FULL_best.pt --epochs 10
```

손실은 `L1 + perceptual + SSIM + wavelet`이고 가중치는 `train.loss`에서 온다. 각 삼중쌍은
360×640으로 리사이즈된 뒤 180×320으로 랜덤 크롭된다. 손실이 무엇을 측정하고 왜 그런지:
[weights/README_weights.ko.md](weights/README_weights.ko.md#residual-규약).

### Ablation

구조 요소 10개를 `projector_distortion/configs/restoration.yaml`의 `ablation:` 블록에서
개별로 끌 수 있다. 꺼진 것이 `tag`에 들어가고, 그 tag가 실행 폴더명과 체크포인트 파일명에
모두 반영된다.

| `false`로 두면 | 끄는 것 | Tag |
|---|---|---|
| `use_prenorm` | NAFSEBlock의 pre-LayerNorm | `NoPre` |
| `use_naf_norm` | NAFBlock 내부 LayerNorm2d | `NoNorm` |
| `use_simple_gate` | SimpleGate (`x1*x2`) → GELU로 대체 | `NoGate` |
| `use_naf_scale` | 학습 가능한 residual 스케일 beta / gamma | `NoScale` |
| `use_ca` | 채널 어텐션. 블록이 순수 NAFBlock이 됨 | `NoCA` |
| `use_skip1` | U-Net skip enc1 → dec1 (원해상도) | `NoSkip1` |
| `use_skip2` | U-Net skip enc2 → dec2 (1/2) | `NoSkip2` |
| `use_skip3` | U-Net skip enc3 → dec3 (1/4) | `NoSkip3` |
| `use_bottleneck` | 1/8 해상도 bottleneck → Identity | `NoBott` |
| `use_tanh` | 출력 tanh. residual이 무한 범위가 됨 | `NoTanh` |

여러 개는 결합된다: `NoCA-NoSkip3`. 하나도 끄지 않으면 `FULL`.

용량도 같은 블록에 있다: `base_dim` (48), `enc_depth` (`[2, 2, 3]`), `dec_depth`
(`[2, 2, 2]`), `bottleneck_depth` (2), `dw_expand` (2), `ffn_expand` (2), `ca_reduction`
(16).

```yaml
# projector_distortion/configs/restoration.yaml
ablation:
  use_ca: false
```

```bash
python train.py --epochs 30      # runs/<날짜>_30ep_NoCA/ 생성
```

체크포인트가 자신의 아키텍처 config를 품고 있어서 추론 시점에는 아무것도 다시 지정할 필요가
없다:

```bash
# projector_distortion/configs/restoration.yaml
model:
  weights: runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

예외는 config 없이 순수 `state_dict`로 저장된 *레거시* 체크포인트다. 이 경우 현재 `ablation:`
블록의 값으로 폴백하므로, ablation된 레거시 가중치는 그 블록을 해당 구조에 맞춰 놔야 제대로
로드된다. `train.py`가 쓰는 체크포인트는 전부 config를 품으므로 외부에서 받은 가중치에만
해당한다.

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
따로 다시 쓰는 것은 프레임당 인코딩 4번을 더 쓰면서 얻는 게 없었다.

`--save-every 0`은 이미지 디렉터리를 아예 만들지 않는다. `detections.csv`는 여전히 모든
프레임을 덮는다.

`evaluate.py`는 대신 `report.json` + `per_class_*.csv` + `per_image_*.csv`를 쓴다.
`train.py`는 `runs/` 아래에 `restorer_<tag>_best.pt` + `loss_log.csv` + `loss_plots.png`를
쓴다.

---

## 복원 속도 줄이기

복원이 라이브 프레임의 약 46%다. 그래서 가장 먼저 줄일 대상이다. 네트워크가 fully
convolutional이라 작동 해상도가 런타임 노브가 된다 — 재학습 없이.
`projector_distortion/configs/restoration.yaml`의 `model.input_size`에서 지정한다. 번들
데이터셋, `yolo` 검출기 기준 측정:

| `model.input_size` | 복원 | 검출 mAP | PSNR 이득 | SSIM 이득 |
|---|---|---|---|---|
| `[320, 180]` | 9.7 ms | 0.9866 | +8.95 dB | +0.163 |
| `[480, 270]` | 13.4 ms | **1.0000** | +11.32 dB | +0.183 |
| `[640, 360]` (기본) | 20.5 ms | **1.0000** | **+13.17 dB** | **+0.218** |
| `[854, 480]` | 41.2 ms | 1.0000 | +9.89 dB | +0.145 |

```yaml
# projector_distortion/configs/restoration.yaml
model:
  input_size: [480, 270]
```

`[480, 270]`은 복원 시간의 1/3만 쓰고 검출은 차이를 느끼지 못한다. PSNR/SSIM 이득만 줄어든다.
그 아래로 가면 mAP가 흔들리기 시작한다.

640×360 **위로** 가면 양쪽 다 나빠진다. 체크포인트는 360×640에서 리사이즈한 180×320 크롭으로
학습됐고, 854×480은 그 스케일에서 충분히 벗어나 복원 품질이 떨어지면서 시간은 두 배가 된다.

mixed precision과 `torch.compile`은 손댈 가치가 없다. 그래서 둘 다 연결하지 않았다. fp16
autocast는 2% 더 *느리게* 측정됐다 — 네트워크가 memory-bound일 만큼 작아서 autocast가
절약분보다 더 든다. compile은 Windows가 제공하지 않는 Triton 빌드를 요구한다.

진짜 더 작은 네트워크는 재학습이 필요하다. 640×360 기준:

| `ablation:` 변경 | 파라미터 | forward |
|---|---|---|
| 기본 | 4,184,259 | 13.8 ms |
| `base_dim: 32` | 1,878,659 | 9.7 ms |
| `enc_depth: [1,1,1]` + `dec_depth: [1,1,1]` + `bottleneck_depth: 1` | 2,418,837 | 7.3 ms |
| `use_ca: false` | 4,116,147 | 13.1 ms |

블록을 얕게 하는 쪽이 좁게 하는 쪽보다 이득이 크다. 채널 어텐션 제거는 5%를 벌지만 재학습할
가치는 없다.

---

[English](README_running.md) · [← README](README.ko.md)

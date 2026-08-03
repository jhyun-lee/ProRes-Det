# ProRes-Det

**Pro**jector **Res**toration + **Det**ection · [English](README.md)

프로젝터가 쏘고 있는 화면을 카메라로 찍으면 투사광 때문에 원본이 망가진다.
이 프레임워크는 그 **투사광을 제거(restore)** 하고, **복원 전/후의 객체 검출 성능을 측정**한다.
복원 모델과 검출 모델 모두 두 개의 작은 인터페이스 뒤에서 **교체 가능**하다.

```
        camera capture (pro)  ─┐
                               ├─▶ restorer ─▶ residual ─▶ restored = pro − residual
   projected source (beam)  ──┘                              │
                                                             │
              detector(pro) ◀── 복원 전            복원 후 ──▶ detector(restored)
                     └──────────── 비교: mAP / PSNR / SSIM ────────────┘
```

---

## 1. 환경 구축

Python ≥ 3.9 필요.

```bash
git clone <repo> && cd ProRes-Det

pip install -e "."              # 기본: ssd + none 백엔드
pip install -e ".[yolo]"        # + ultralytics       → --detector yolo
pip install -e ".[train]"       # + pytorch-msssim 등 → train.py
pip install -e ".[live]"        # + screeninfo        → --live (Windows 는 불필요)
pip install -e ".[test]"        # + pytest
pip install -e ".[all]"         # 전부
```

> 대괄호는 셸에 따라 glob 문자로 해석된다(zsh 등). **따옴표를 씌우면 bash / zsh /
> PowerShell / cmd 어디서나 동작**한다.

기본 의존성은 `torch torchvision opencv-python numpy PyYAML tqdm`.

선택 의존성은 모듈 최상단이 아니라 **실제로 쓰는 함수 안에서 import** 한다.
`ultralytics` 를 설치하지 않아도 `--detector ssd` 는 그대로 동작하고, `--live` 를 안 쓰면
`screeninfo` 나 Win32 플러밍도 로드되지 않는다.

---

## 2. 디렉토리 구성

```
ProRes-Det/
├── demo.py                   복원 → 검출 전체 흐름 (오프라인 / --live)
├── evaluate.py               복원 전후 검출 성능 + 복원 품질 측정
├── train.py                  복원 모델 미세조정
├── setup.py  requirements.txt  LICENSE
│
├── projector_distortion/     ── 라이브러리 본체
│   ├── __init__.py           공개 API
│   ├── cli.py                공통 인자, 설정 우선순위, 콘솔 진입점
│   ├── config.py             YAML 로딩 + 경로 해석
│   ├── data.py               샘플 탐색, 라벨 로딩, 학습 데이터셋
│   ├── configs/              기본 설정 YAML
│   ├── models/               복원 + 검출 모델
│   ├── pipeline/             오프라인 / 라이브 실행 루프
│   └── utils/                이미지·시각화·기록 헬퍼
│
├── tests/                    pytest 테스트
├── weights/                  가중치 3개 → weights/README.md
├── data/                     샘플 데이터셋 → data/README.md
└── output/                   실행 결과
```

### 각 모듈의 역할

| 파일 | 역할 |
|---|---|
| [demo.py](demo.py) | 진입점. 인자 파싱 → 모델 생성 → 오프라인/라이브 파이프라인 실행 → 요약 출력 |
| [evaluate.py](evaluate.py) | 진입점. GT 가 있는 샘플만 골라 P/R/F1/mAP 와 PSNR/SSIM 계산 후 리포트 저장 |
| [train.py](train.py) | 진입점. 복원 모델 학습 루프 (L1 + perceptual + SSIM + wavelet 손실) |
| [cli.py](projector_distortion/cli.py) | 세 진입점이 공유하는 인자·설정 우선순위·모델 생성 로직 |
| [config.py](projector_distortion/config.py) | YAML 로드/병합, 프로젝트 루트 기준 경로 해석 |
| [data.py](projector_distortion/data.py) | 파일명으로 pro/beam/clean/label 을 짝짓고, 학습용 패치 데이터셋 제공 |
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · 검출기 레지스트리 — **확장 지점** |
| [models/restoration.py](projector_distortion/models/restoration.py) | 복원 네트워크(3-level U-Net), 구조 설정, 체크포인트 I/O, 파이프라인용 래퍼 |
| [models/detection.py](projector_distortion/models/detection.py) | yolo(ultralytics) · ssd(torchvision) 래퍼, 박스 크기 필터 |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | 디스크의 이미지 쌍을 배치 처리. 하드웨어 불필요 |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | 웹캠 + 프로젝터 리그. 모니터 배치, 캘리브레이션, 워핑, 워커 스레드 |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ 텐서 변환, 리사이즈, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | 박스 그리기, 2×2 비교 패널, 캘리브레이션 오버레이 |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — 출력 디렉토리에 쓰는 모든 것을 전담 |

### 번들 데이터셋

인자 없이 바로 돌 수 있도록 샘플 데이터가 저장소에 들어있다.

| 경로 | 내용 | 쓰는 곳 |
|---|---|---|
| [data/sample_input/pro/](data/sample_input/pro) | 투사된 화면을 찍은 이미지 22장 | 모델 입력 ch 0:3 |
| [data/sample_input/beam/](data/sample_input/beam) | 프로젝터가 쏜 원본 프레임 22장 | 모델 입력 ch 3:6 |
| [data/sample_gt/clean/](data/sample_gt/clean) | 투사광 없는 정답 화면 10장 | 학습 타겟 / PSNR·SSIM 기준 |
| [data/sample_gt/labels/](data/sample_gt/labels) | YOLO 포맷 검출 라벨 10개 | mAP 기준 |
| [data/live/BeamVideo.mp4](data/live/BeamVideo.mp4) | 프로젝터로 재생할 클립 (3.3분) | `--live` 입력 |
| [data/live/BaseBackGround.jpg](data/live/BaseBackGround.jpg) | 캘리브레이션 중 띄울 배경 | `--live` 입력 |

파일명 규약(`pro` ↔ `beam` ↔ `clean` 을 id 로 짝짓는 방식)과 실데이터 교체 방법은
[data/README.md](data/README.md), 가중치 정보는 [weights/README.md](weights/README.md).

---

## 3. 실행

### 3.1 빠른 실행

```bash
python demo.py        # 번들 샘플 22쌍 복원 + 검출 → output/<타임스탬프>/
python evaluate.py    # 복원 전/후 mAP + PSNR/SSIM 표 → output/eval/
```

인자 없이 바로 돈다. 아래 셋은 모델만 바꾼 것으로, 입력·출력 경로는 동일하다.

```bash
python demo.py --detector ssd     # 검출기 교체 (ultralytics 불필요)
python demo.py --detector none    # 복원만, 검출 생략
python demo.py --limit 5          # 앞 5쌍만
```

### 3.2 `demo.py` — 복원 → 검출

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| **입력** | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| **정답**(선택) | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| **출력** | `output/<타임스탬프>/` | `--output <dir>` · `--name <이름>` |

정답이 없으면 PSNR/SSIM 만 생략되고 실행은 계속된다.

| 옵션 | 설명 |
|---|---|
| `--detector yolo\|ssd\|none` | 검출 백엔드 (기본 yolo) |
| `--conf <float>` | 검출 신뢰도 하한 (기본 0.25) |
| `--limit N` | 처리할 쌍 개수 제한 (0 = 전부) |
| `--save-every N` | 이미지 저장 간격. `0` 이면 csv 만 남김 |
| `--save-kinds a,b` | 저장할 이미지 종류만 선택 (`captured,restored,panel` 등) |
| `--video` | 2×2 패널을 `result.mp4` 로도 저장 |

### 3.3 `evaluate.py` — 복원 전후 점수 비교

`clean` 과 `label` 이 **둘 다 있는** 샘플만 채점한다.

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| **입력** | `data/sample_input/` (`pro/` + `beam/`) | `--input <dir>` |
| **정답** | `data/sample_gt/` (`clean/` + `labels/`) | `--gt <dir>` |
| **출력** | `output/eval/`<br>`report.json`, `per_class_<backend>.csv`, `per_image_<backend>.csv` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--detectors yolo,ssd` | 여러 백엔드를 한 번에 비교 (백엔드마다 한 행) |
| `--iou <float>` | TP 판정 IoU 임계값 (기본 0.5) |
| `--limit N` | 채점할 쌍 개수 제한 |

```bash
python evaluate.py --detectors yolo,ssd --iou 0.5
```

콘솔에 요약 표가 뜨고, 클래스별 P / R / F1 / AP 는 csv 에 들어간다.

```python
import pandas as pd
pc = pd.read_csv("output/eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # 클래스별 AP 변화
```

### 3.4 `train.py` — 복원 모델 재학습

`pro` / `beam` / `clean` 3장이 모두 갖춰진 것만 학습에 쓴다.

| | 기본 경로 | 바꾸는 옵션 |
|---|---|---|
| **입력** | `data/sample_input/` (`pro/` + `beam/`) | `--data-root <dir>` |
| **정답** | `data/sample_gt/clean/` — **필수** | `--gt <dir>` |
| **출력** | `runs/<MMDD_HHMM>_<epochs>ep_<tag>/`<br>`restorer_<tag>_best.pt`, `epoch_N.pt`, `loss_log.csv`, `loss_plots.png` | `--out <dir>` |

| 옵션 | 설명 |
|---|---|
| `--epochs N` `--batch-size N` `--lr F` | 기본값은 `configs/restoration.yaml` |
| `--sample N` | 학습에 쓸 triplet 개수 제한 |
| `--resume <ckpt>` | 체크포인트에서 이어서 학습 |
| `--no-ca` 등 10종 | ablation. 끈 구조가 `tag` 에 기록됨 (`NoCA`, `NoCA-NoSkip3` …) |

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --no-ca --epochs 30
```

체크포인트에 구조 설정이 동봉되므로, 나중에 쓸 때 플래그를 다시 줄 필요가 없다.

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

### 3.5 `demo.py --live` — 웹캠 + 프로젝터

| | 기본값 | 바꾸는 옵션 |
|---|---|---|
| **투사 클립** | `data/live/BeamVideo.mp4` | `--clip <path>` |
| **캘리브 배경** | `data/live/BaseBackGround.jpg` | `--background <path>` |
| **카메라** | 웹캠 0번 | `--camera N` · `--cam-width/height/fps` · `--cam-backend` |
| **출력** | `output/<타임스탬프>/` + `calib/` + `result.mp4` | `--output <dir>` · `--name <이름>` |

| 옵션 | 설명 |
|---|---|
| `--screen N` | 프로젝터가 붙은 모니터 인덱스 (0 = 주 모니터) |
| `--offset N` | 프로젝터→카메라 지연 보정 (프레임 수, 기본 6) |
| `--manual-calib` | 자동 검출 대신 4점 직접 클릭 |
| `--debug-view` | 워핑 전 카메라 + 4점을 실시간 표시 |
| `--max-frames N` | 0 = 클립 끝까지 |

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

`--screen` 을 모르면 아무 값이나 주고 실행 → 감지된 모니터 표가 먼저 출력된다.

동작 순서: 흑/백 플래시 → 차영상에서 4점 자동 검출 → 호모그래피 1회 계산 → 매 프레임 워핑.
자동 검출이 실패하면 수동 클릭으로 폴백한다. 결과가 이상하면 `--debug-view` 와
`output/*/calib/` 의 중간 이미지(`mask.jpg`, `diff.jpg`, `warped.jpg`)를 본다.

### 3.6 출력물 형식

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

**clean 이미지와 annotated 이미지를 따로 저장**하는 게 핵심이다. 덕분에 실행이 끝난 뒤에도
PSNR/SSIM 재계산과 다른 검출기로의 재실행이 가능하다.

`evaluate.py` 는 위 대신 `report.json` + `per_class_*.csv` + `per_image_*.csv` 를 쓴다.

---

## 4. YAML 설정

```
configs/*.yaml  <  --restoration-config / --detection-config 로 넘긴 YAML  <  CLI 플래그
```

| 파일 | 담는 것 |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | 복원 가중치 경로, 입력 크기, 구조 토글, 학습 하이퍼파라미터·손실 가중치 |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | 검출 백엔드, 백엔드별 가중치 경로, conf 임계값, 박스 크기 필터, 17개 클래스 목록 |

YAML 안의 상대 경로는 **작업 디렉토리가 아니라 프로젝트 루트 기준**으로 해석된다.
어디서 `python demo.py` 를 실행해도 동작한다.

일부만 덮어쓰려면 바꿀 키만 담은 YAML 을 넘기면 된다 (재귀 병합).

```yaml
# my_det.yaml
detector:
  backend: ssd
  conf: 0.4
```

```bash
python demo.py --detection-config my_det.yaml
```

---

## 5. 모듈 교체

### 검출기 추가

```python
from projector_distortion.models.base import BaseDetector, Detection, register_detector

@register_detector("mydet")
class MyDetector(BaseDetector):
    name = "mydet"

    def __init__(self, weights, class_names=None, conf=0.25, device="cpu", **_):
        super().__init__(class_names or [], conf, device)
        self.net = load_my_model(weights)

    def detect(self, bgr):
        return [Detection(cls_id, self.label_of(cls_id), score, (x1, y1, x2, y2))
                for cls_id, score, (x1, y1, x2, y2) in self.net(bgr)]
```

이걸로 끝이다. `--detector mydet` 이 바로 동작하고, 파이프라인·기록·평가 코드는
아무것도 바뀌지 않는다.

### 복원기 교체

`BaseRestorer` 를 상속해 `restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)`
하나만 구현하면 파이프라인에 꽂힌다.

```
입력  (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
출력  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

#### 왜 clean 이 아니라 residual 을 예측하나

네트워크가 복원 이미지를 **직접 그리지 않고**, `pro` 에서 뺄 빛만 내놓는다.

**1) 원본 보존이 기본값이 된다.**
투사광이 닿지 않은 영역은 residual ≈ 0 이면 되고, 그러면 `pro` 픽셀이 그대로 통과한다.
"아무것도 하지 않음"이 항등 함수라, 네트워크는 **바뀌어야 하는 곳만** 학습하면 된다.
clean 을 직접 회귀하면 멀쩡한 배경까지 전부 재생성해야 해서 손대지 말아야 할 곳이 뭉개진다.

**2) 오브젝트를 그려 넣는 과적합을 막는다.**
clean 을 직접 출력하게 두면, 입력을 거의 무시하고 **학습셋에서 본 화면을 통째로 외워
그려내는** 지름길이 손실을 가장 빠르게 낮춘다. 특히 이 데이터는 clean 10장에 pro 22장이
붙어 clean 이 반복 등장하므로, `oriId` 별 정답 화면을 암기하는 게 유리해진다.
그렇게 학습된 모델은 **검출기가 원래 없는 물체를 보게 만들어** 평가 자체를 무의미하게 만든다.
`restored = pro − residual` 구조는 출력이 **항상 실제 카메라 픽셀에서 파생**되도록 강제해서
그 경로를 막는다.

**3) 값이 폭주하지 않는다.**
출력단 `tanh` 로 residual ∈ [-1, 1], 뺄셈 뒤 `clamp(-1, 1)`. 두 겹으로 범위가 잡힌다.
(`--no-tanh` 로 끄면 무제한 residual 이 되고, 그게 ablation 항목 중 하나다.)

#### 어떻게 강제하나 — 손실은 residual 이 아니라 restored 에 건다

핵심은 **뺄셈이 그래프 안에 있다**는 것이다. residual 에 직접 정답을 주지 않고,
뺀 결과만 clean 과 비교한다. 그래서 "무엇을 빼야 하는가"는 네트워크가 스스로 찾는다.

```python
residual = net(torch.cat([pro, beam], dim=1))     # 네트워크 출력
restored = (pro - residual).clamp(-1, 1)          # 뺄셈이 그래프 안
loss = (0.93 * L1(restored, clean)
      + 2.04 * Perceptual(restored, clean)
      + 0.53 * (1 - SSIM(restored, clean))
      + 0.90 * WaveletHF(restored, clean))        # 4항 전부 restored 기준
```

| 손실 항 | 무엇을 재나 | 무엇을 벌하나 |
|---|---|---|
| `L1` | 픽셀 절대 오차 | 전역 색·밝기 어긋남 |
| `Perceptual` (VGG19 relu3_3) | 특징맵 거리 | 픽셀은 비슷한데 구조가 깨진 결과 |
| `1 − SSIM` | 국소 휘도·대비·구조 상관 | 평균만 맞춘 밋밋한 출력 |
| `WaveletHF` (Haar LH/HL/HH, **LL 제외**) | 엣지·텍스처만 | 흐릿하게 뭉개서 손실 낮추기 |

`WaveletHF` 가 저주파(LL) 를 빼고 보는 게 요점이다. 전체를 블러 처리해도 L1 은 잘 내려가지만
고주파 항은 안 내려간다. 이게 residual 이 실제 투사광 패턴의 경계를 따라가게 만든다.

가중치는 [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) 의
`train.loss` 에 있고, 이 데이터셋에 대한 Optuna 스윕 결과다. 구현은
[train.py](train.py) 참고.

> 인터페이스 자체는 clean 을 직접 뱉는 복원기도 받아준다. 다만 그 경우 `residual` 시각화와
> `residual_mean` 지표가 의미를 잃고, 위 두 가지 이점도 사라진다.

---

## 6. 라이선스

MIT. [LICENSE](LICENSE) 참고.

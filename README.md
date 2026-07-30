# projector-distortion-framework

프로젝터가 쏘고 있는 화면을 카메라로 찍으면 투사광 때문에 원래 화면이 망가진다.
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

## 번들 샘플 실측 결과

`python evaluate.py --detectors yolo,ssd` — 22쌍, IoU 0.5

| backend | source | P | R | F1 | mAP | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| yolo | captured | 0.859 | 0.893 | 0.876 | 0.884 | 201 | 33 | 24 |
| yolo | **restored** | **1.000** | **1.000** | **1.000** | **1.000** | 225 | 0 | 0 |
| ssd | captured | 0.677 | 0.876 | 0.764 | 0.809 | 197 | 94 | 28 |
| ssd | **restored** | **0.996** | **1.000** | **0.998** | **1.000** | 225 | 1 | 0 |

복원 품질 (clean 정답 대비): **PSNR 13.76 → 26.93 dB (+13.17)**, **SSIM 0.705 → 0.923 (+0.218)**

> **이 수치를 그대로 인용하지 말 것.** 번들 데이터는 clean 이미지 10장 / 22쌍의 토이셋이고,
> 검출 라벨을 그 clean 이미지 위에 그렸다. 복원 결과가 clean 에 가까워지면(SSIM 0.92)
> 검출기는 사실상 clean 화면을 보게 되므로 mAP 1.0 은 **구조적으로 상한에 가깝다.**
> 실데이터에서는 더 낮게 나온다. 파이프라인이 동작한다는 증거로만 보면 된다.

## 설치

```bash
git clone <repo> && cd projector-distortion-framework
pip install -e .              # 기본: ssd + none 백엔드까지 동작
pip install -e .[yolo]        # + ultralytics (yolo 백엔드)
pip install -e .[train]       # + pytorch-msssim, pandas, matplotlib (train.py)
pip install -e .[all]         # 전부 + pytest
```

Python ≥ 3.9. 기본 의존성은 `torch torchvision opencv-python numpy PyYAML tqdm`.
안 쓰는 백엔드의 패키지는 import 조차 하지 않는다.

## 빠른 실행

```bash
python demo.py                        # 번들 샘플 22쌍, 결과는 output/<타임스탬프>/
python evaluate.py                    # 복원 전/후 mAP + PSNR/SSIM 표
python demo.py --detector ssd         # 검출기 교체 (ultralytics 불필요)
python demo.py --detector none        # 복원만
```

인자 없이 바로 돈다 — 가중치와 샘플 데이터가 이미 들어있다.

## 디렉토리

```
projector-distortion-framework/
├── demo.py                   왜곡 복원 → 객체 인식 전체 흐름 (오프라인 / --live)
├── evaluate.py               복원 전후 검출 성능 + 복원 품질 측정
├── train.py                  복원 모델 미세조정
├── setup.py  requirements.txt  LICENSE
│
├── projector_distortion/
│   ├── __init__.py           공개 API
│   ├── cli.py                공통 인자 처리, 콘솔 스크립트 진입점
│   ├── config.py             YAML 로딩 + 우선순위 규칙
│   ├── data.py               샘플 탐색, 정답 로딩, 학습 데이터셋
│   ├── configs/
│   │   ├── restoration.yaml  복원 모델 + 학습 설정
│   │   └── detection.yaml    검출 백엔드 + 클래스 목록
│   ├── models/
│   │   ├── base.py           BaseRestorer / BaseDetector / Detection / 레지스트리
│   │   ├── restoration.py    config + 블록 + U-Net + 체크포인트 I/O + 래퍼
│   │   └── detection.py      yolo + ssd 래퍼
│   ├── pipeline/
│   │   ├── offline.py        이미지/폴더 배치 처리
│   │   └── live.py           웹캠 + 프로젝터 리그 (캘리브레이션·워핑·모니터·워커)
│   └── utils/
│       ├── image.py          텐서 변환, 리사이즈, PSNR/SSIM/IoU
│       ├── visualize.py      박스 그리기, 비교 레이아웃, 캘리브레이션 오버레이
│       └── recording.py      RunRecorder — 출력 디렉토리 전담
│
├── tests/                    77개 테스트 (pytest)
├── weights/                  가중치 3개 52MB → weights/README.md
├── data/                     토이 데이터셋 → data/README.md
└── output/                   실행 결과
```

## 사용 시나리오

### S1. 복원이 검출을 개선하는지 측정

```bash
python evaluate.py --detectors yolo,ssd
```

`output/eval/report.json` + `per_class_*.csv` + `per_image_*.csv`. 클래스별 P/R/F1/AP 까지 나온다.

```python
import pandas as pd
pc = pd.read_csv("output/eval/per_class_yolo.csv")
pc.pivot_table(index="name", columns="source", values="ap")   # 클래스별 AP 변화
```

### S2. 검출기만 바꿔서 비교

```bash
for D in yolo ssd; do python demo.py --detector $D --save-every 0 --name run_$D; done
```

`--save-every 0` 이면 이미지를 안 쓰고 csv + 요약만 남긴다. 각 실행의 `run_meta.json` 에
백엔드·가중치·conf 가 기록되니 나중에 폴더만 모아 비교하면 된다.

### S3. 복원 품질만 정량 평가

```bash
python demo.py --detector none --save-kinds captured,restored,beam
```

`frames/*_restored.jpg` 는 **박스가 없는 clean** 이미지라 그대로 메트릭 입력이 된다.
`_det` 접미사가 붙은 것만 박스가 그려져 있다.

### S4. 라이브 리그 (웹캠 + 프로젝터)

```bash
python demo.py --live --screen 2
python demo.py --live --screen 2 --save-every 30 --debug-view
```

`--screen` 을 모르면 아무 값이나 주고 실행 → 감지된 모니터 표가 먼저 출력된다.

### S5. 캘리브레이션이 이상할 때

```bash
python demo.py --live --screen 2 --debug-view
```

실행 중 `PreWarp_Debug` 창에 **워핑 전** 카메라 + 4점이 계속 뜬다. 끝난 뒤 `output/*/calib/`:

| 파일 | 보는 법 |
|---|---|
| `raw_points.jpg` | 4점이 스크린 모서리에 정확히 붙었나 |
| `mask.jpg` | 흰 영역이 스크린 하나인가, 조명/창문까지 잡았나 |
| `diff.jpg` | 흑백 플래시 차이가 충분한가 (부족하면 `--calib-settle` ↑) |
| `warped.jpg` | 최종 정면화 결과가 반듯한가 |

자동 캘리브레이션이 실패하면 수동 클릭으로 자동 폴백한다. 처음부터 수동은 `--manual-calib`.

### S6. 실데이터로 재학습

```bash
python train.py --epochs 30
python train.py --data-root /path/to/dataset --epochs 30
python train.py --no-ca --epochs 30            # ablation
```

체크포인트에 config 가 동봉되므로 나중에 플래그 없이 그대로 쓸 수 있다:

```bash
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
```

### S7. 라이브러리로 쓰기

```python
from projector_distortion import build_restorer, build_detector
from projector_distortion.data import find_samples
from projector_distortion.pipeline import process_sample

restorer = build_restorer("weights/restorer_restormerlike.pt")
detector = build_detector("ssd", "weights/detector_ssdlite.pth")

for i, s in enumerate(find_samples("data/sample_input", "data/sample_gt")):
    r = process_sample(s, restorer, detector, frame_id=i)
    print(s.name_id, len(r.det_captured), "->", len(r.det_restored), r.metrics())
```

### S8. 장시간 무인 녹화

```bash
python demo.py --live --screen 2 --save-every 300 --max-saved-frames 50 --jpeg-quality 85
```

csv 는 전 프레임 기록되고 이미지만 드문드문 남아 용량이 예측 가능해진다.

## 모듈 교체하기

### 검출기 추가

```python
from projector_distortion.models.base import BaseDetector, Detection, register_detector

@register_detector("mydet")
class MyDetector(BaseDetector):
    name = "mydet"
    def __init__(self, weights, **kw):
        super().__init__(kw.get("class_names") or [], kw.get("conf", 0.25),
                         kw.get("device", "cpu"))
        ...
    def detect(self, bgr):
        return [Detection(cls_id, self.label_of(cls_id), conf, (x1, y1, x2, y2)), ...]
```

`--detector mydet` 이 바로 동작한다. 파이프라인·기록·평가 코드는 손댈 필요 없다.

### 복원기 교체

`BaseRestorer` 를 상속해 `restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)`
하나만 구현한다. 스칼라 residual 을 싸게 낼 수 있으면 `restore_full()` 도 오버라이드한다.

## 복원 모델

`RestorationNet` — RestormerLikeBlock 의 3-level U-Net. 4,184,259 params (base_dim 48).

```
입력 (B,6,H,W) = cat([pro, beam]), [-1,1]
  stem 3x3 ──▶ enc1(48) ─▶ enc2(96) ─▶ enc3(192) ─▶ bottleneck(384)
                 │            │           │              │
                 └─ skip1 ────┼─ skip2 ───┼─ skip3 ──────┘
                              ▼           ▼
              dec1(48) ◀── dec2(96) ◀── dec3(192)
  out 3x3 ─▶ tanh ─▶ residual (B,3,H,W)      restored = (pro − residual).clamp(−1,1)
```

완전 convolution 이라 학습 패치 180×320 과 추론 640×360 에서 동작이 동일하다.

> **이름 주의.** `RestormerLikeBlock` 은 `LayerNorm2d → NAFBlock → CALayer` 로,
> Restormer 의 **MDTA 어텐션이 없다.** 원 구현이 "lightweight proxies / baselines for
> measurement only" 라고 명시한 대리 구현이며 기능적으로는 NAFNet + squeeze-excite 에 가깝다.

### Ablation

모든 구조를 개별로 끌 수 있다. 기여도 측정용이고, 설정이 체크포인트에 동봉된다.

| 플래그 | 끄는 구조 |
|---|---|
| `--no-prenorm` | RestormerLikeBlock 의 pre-LayerNorm |
| `--no-naf-norm` | NAFBlock norm1 / norm2 |
| `--no-simple-gate` | SimpleGate(x1·x2) → GELU (채널 반감이 사라져 conv 가 넓어짐) |
| `--no-naf-scale` | learnable residual scale beta / gamma |
| `--no-ca` | 채널 어텐션 (`--no-prenorm` 과 같이 끄면 순수 NAFBlock) |
| `--no-skip1/2/3` | U-Net skip (join conv 가 2C→C 에서 C→C 로) |
| `--no-bottleneck` | 1/8 해상도 bottleneck |
| `--no-tanh` | 출력 tanh (residual 무제한) |

용량 조절은 `train.py` 에만 있다:
`--base-dim --enc-depth --dec-depth --bottleneck-depth --dw-expand --ffn-expand --ca-reduction`

## 검출 백엔드

| `--detector` | 구조 | 기본 가중치 | 프레임워크 |
|---|---|---|---|
| `yolo` (기본) | YOLO11s | `weights/detector_yolo11s.pt` | `ultralytics` |
| `ssd` | SSDLite320-MobileNetV3-L | `weights/detector_ssdlite.pth` | torchvision 만 |
| `none` | — | — | 복원만 |

클래스 17종. 라벨 규약이 백엔드마다 달라 내부에서 0-based 로 통일한다 —
ultralytics 는 이미 0-based, torchvision 은 COCO id(1..N, 0=background) 를 낸다.

## 출력

```
output/<run>/
├── run_meta.json     설정·환경·캘리브레이션·요약 전부
├── detections.csv    박스 1개 = 1행  (frame_id, name_id, source, cls_id, name, conf, x1..y2)
├── frames.csv        프레임 1개 = 1행 (검출 수, residual, 타이밍, psnr, ssim, saved)
├── calib/            캘리브레이션 근거 (--live 만)
└── frames/           --save-every 간격
    └── <name>_{beam,captured,captured_det,restored,restored_det,residual,panel,raw}.jpg
```

`source` 는 `captured`(복원 전) / `restored`(복원 후). csv 는 **저장 안 한 프레임까지 전부** 기록된다.

`captured.jpg` / `restored.jpg` = **박스 없는 clean**, `_det` = 박스 그린 것.
clean 을 남기는 게 이 구조의 핵심이다 — 그래야 PSNR/SSIM 과 다른 검출기 재평가가 가능하다.

### 용량 제어

| 옵션 | 효과 |
|---|---|
| `--save-every N` | N프레임마다만 이미지 저장 (`0` = 안 씀). csv 는 항상 전체 |
| `--save-kinds` | 저장 종류 선택. 기본은 8종 전부 |
| `--max-saved-frames N` | 이미지 저장 총량 하드 캡 |
| `--jpeg-quality` (92) | |
| `--video` / `--no-video` | 2×2 패널 mp4 |

프레임당 8종 = 약 680 KB. 실행 직전에 예상 용량을 출력한다.

## 설정 우선순위

낮음 → 높음: `configs/*.yaml` → `--restoration-config` / `--detection-config` → 개별 CLI 플래그

설정 안의 상대 경로는 **프로젝트 루트 기준**으로 해석되므로 어느 디렉토리에서 실행해도 동작한다.

## 라이브 모드 참고

- 캘리브레이션은 **시작할 때 1회**만 한다. 이후 호모그래피를 재사용하며 자동 재보정하지 않는다.
  실행 중 카메라나 프로젝터가 움직이면 어긋난 채로 끝까지 간다 (`--debug-view` 로 감시).
- Windows 에서 `cv2.setWindowProperty(WND_PROP_FULLSCREEN)` 은 창을 primary 로 되돌려버려
  `--screen` 이 무시된다. 그래서 Win32 API 로 `WS_POPUP` + `SetWindowPos` 로 직접 고정하고,
  DPI 스케일링에서도 물리 픽셀이 나오도록 `SetProcessDpiAwareness` 를 먼저 호출한다.
  프로젝터는 Windows 표시 설정에서 **"복제"가 아니라 "확장"** 이어야 별도 모니터로 잡힌다.
- Windows 기본 MSMF 백엔드는 웹캠 오픈에 20초 넘게 걸리는 경우가 있어 `auto` 는 DirectShow 를
  먼저 시도한다 (실측 20.8s → 0.9s). `--cam-backend` 로 강제 지정 가능.
- 4점 클릭 순서는 자유다 — 내부에서 TL/TR/BR/BL 로 정렬한다.

## 테스트

```bash
pip install -e .[test]
pytest tests -q
```

77개. 가중치나 옵션 패키지가 없으면 해당 테스트만 skip 한다.

| 파일 | 범위 |
|---|---|
| `test_restoration.py` | 네트워크 shape, ablation 10종 backward, 체크포인트 왕복, 래퍼 계약 |
| `test_detection.py` | 레지스트리, 두 래퍼, 라벨 규약, 박스 필터 |
| `test_pipeline.py` | 파일명 규약, 샘플 탐색, 메트릭, RunRecorder, end-to-end, CLI |

## 검증 현황

| 항목 | 상태 |
|---|---|
| `pytest tests` | 77/77 통과 |
| `demo.py` 오프라인 | 실행 확인 (22쌍) |
| `evaluate.py` | 실행 확인, yolo + ssd 양쪽 |
| `train.py` | 1 epoch 실행 확인, 체크포인트가 cfg 동봉으로 재로드됨 |
| `pip install -e .` | 확인 |
| **`demo.py --live`** | **미검증** — 웹캠·프로젝터 필요. 자동 캘리브레이션은 실제 조명·프로젝터 지연에서 `--calib-settle` 조정이 필요할 수 있다 |

## 라이선스

MIT — [LICENSE](LICENSE)

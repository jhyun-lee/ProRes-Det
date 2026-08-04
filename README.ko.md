# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

프로젝터가 쏘고 있는 화면을 카메라로 찍으면 투사광 때문에 원본이 망가진다.
이 프레임워크는 그 투사광을 제거(restore) 하고, 복원 전/후의 객체 검출 성능을 측정한다.
복원 모델과 검출 모델 모두 두 개의 작은 인터페이스 뒤에서 교체 가능하다.

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
pip install -e ".[all]" -r requirements-cuda.txt    # 전부 + CUDA torch (아래 참고)
```

> 대괄호는 셸에 따라 glob 문자로 해석된다(zsh 등). 따옴표를 씌우면 bash / zsh /
> PowerShell / cmd 어디서나 동작한다.

기본 의존성은 `torch torchvision opencv-python numpy PyYAML tqdm`.

### GPU

**`.[all]` 만 실행하면 Windows 에서는 CPU 전용 torch 가 깔린다.** PyPI 기본 `torch` 휠에
Windows용 CUDA 가 없기 때문이다. 전부 정상 동작하고 증상은 하나뿐이다 — 복원이 프레임당
13 ms 대신 380 ms 걸린다. 같은 명령에 `requirements-cuda.txt` 를 붙이면 CUDA 빌드까지 한 번에:

```bash
pip install -e ".[all]" -r requirements-cuda.txt
```

extra(`.[cuda]`) 로는 안 된다 — 패키지가 "어느 인덱스에서 받아라" 를 지정할 방법이 없다.
그래서 인덱스 지정과 버전 핀을 저 파일에 넣었다. `cu128` 은 RTX 50 시리즈(Blackwell)와
12.8 드라이버가 지원하는 구형 카드까지 커버한다. 더 오래된 드라이버면 파일 안의 `cu128` 을
`cu121` / `cu118` 로 바꾸면 된다. 실제로 뭐가 깔렸는지 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.8.0+cu128 True      <- 정상
# 2.8.0+cpu   False     <- CPU 전용 빌드. -r requirements-cuda.txt 붙여서 재설치
```

모든 진입점이 첫 줄에 해석된 device 를 출력하므로, 조용히 CPU 로 떨어진 실행은 바로 티가 난다.

```
device: cuda - NVIDIA GeForce RTX 5090, torch 2.8.0+cu128
device: cpu - torch 2.8.0+cpu is a CPU-only build, so restoration runs ~25x slower.
```

`--fp16` 은 CUDA 에서만 적용된다. CPU 면 무시된다.

선택 의존성은 모듈 최상단이 아니라 실제로 쓰는 함수 안에서 import 한다.
`ultralytics` 를 설치하지 않아도 `--detector ssd` 는 그대로 동작하고, `--live` 를 안 쓰면
`screeninfo` 나 Win32 플러밍도 로드되지 않는다.

---

## 2. 디렉토리 구성

```
ProRes-Det/
├── demo.py                   복원 → 검출 전체 흐름 (오프라인 / --live)
├── evaluate.py               복원 전후 검출 성능 + 복원 품질 측정
├── train.py                  복원 모델 미세조정
├── README_running.ko.md      스크립트별 옵션·입력·출력
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
├── weights/                  가중치 3개 (51 MiB) → README_weights.ko.md
├── data/                     샘플 데이터셋 + collect.py / record.py → README_data.ko.md
└── output/                   실행 결과
```

### 각 모듈의 역할

| 파일 | 역할 |
|---|---|
| [demo.py](demo.py) | 진입점. 인자 파싱 → 모델 생성 → 오프라인/라이브 파이프라인 실행 → 요약 출력 |
| [evaluate.py](evaluate.py) | 진입점. GT 가 있는 샘플만 골라 P/R/F1/mAP 와 PSNR/SSIM 계산 후 리포트 저장 |
| [train.py](train.py) | 진입점. 복원 모델 학습 루프 (L1 + perceptual + SSIM + wavelet 손실) |
| [data/collect.py](data/collect.py) | 진입점. 리그로 데이터셋 수집: 영상 → beam 프레임, 프로젝터+웹캠 → 촬영본, 정면화 → 정합된 쌍 |
| [data/record.py](data/record.py) | 진입점. 클립을 투사하면서 카메라 화면을 mp4 하나로 녹화. 가중치 불필요 |
| [cli.py](projector_distortion/cli.py) | 세 진입점이 공유하는 인자·설정 우선순위·모델 생성 로직 |
| [config.py](projector_distortion/config.py) | YAML 로드/병합, 프로젝트 루트 기준 경로 해석 |
| [data.py](projector_distortion/data.py) | 파일명으로 pro/beam/clean/label 을 짝짓고, 학습용 패치 데이터셋 제공 |
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · 검출기 레지스트리 — 확장 지점 |
| [models/restoration.py](projector_distortion/models/restoration.py) | 복원 네트워크(3-level U-Net), 구조 설정, 체크포인트 I/O, 파이프라인용 래퍼 |
| [models/detection.py](projector_distortion/models/detection.py) | yolo(ultralytics) · ssd(torchvision) 래퍼, 박스 크기 필터 |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | 디스크의 이미지 쌍을 배치 처리. 하드웨어 불필요 |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | 웹캠 + 프로젝터 리그. 모니터 배치, 캘리브레이션, 워핑, 워커 스레드 |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ 텐서 변환, 리사이즈, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | 박스 그리기, 2×2 비교 패널, 캘리브레이션 오버레이 |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — 출력 디렉토리에 쓰는 모든 것을 전담 |

### 번들 데이터셋

아래 샘플 데이터와 `weights/` 의 체크포인트 3개 모두 git 에 포함되어 있다.
clone 직후 별도 다운로드 없이 바로 실행된다.

| 경로 | 내용 | 쓰는 곳 |
|---|---|---|
| [data/sample_input/pro/](data/sample_input/pro) | 투사된 화면을 찍은 이미지 22장 | 모델 입력 ch 0:3 |
| [data/sample_input/beam/](data/sample_input/beam) | 프로젝터가 쏜 원본 프레임 22장 | 모델 입력 ch 3:6 |
| [data/sample_input/clean/](data/sample_input/clean) | 투사광 없는 정답 화면 10장 | 학습 타겟 / PSNR·SSIM 기준 |
| [data/sample_input/labels/](data/sample_input/labels) | YOLO 포맷 검출 라벨 10개 | mAP 기준 |
| [data/live/BeamVideo.mp4](data/live/BeamVideo.mp4) | 프로젝터로 재생할 클립 (3.3분) | `--live` 입력 |
| [data/live/BaseBackGround.jpg](data/live/BaseBackGround.jpg) | 캘리브레이션 중 띄울 배경 | `--live` 입력 |
| [data/sample_video/](data/sample_video) | 짧은 클립 2개. BeamVideo 대신 쓸 수 있다 | `data/record.py --clip` |

입력과 정답이 같은 폴더에 있으므로 `--input` 과 `--gt` 둘 다 `data/sample_input` 이
기본값이다. 파일명 규약(`pro` ↔ `beam` ↔ `clean` 을 id 로 짝짓는 방식)과 실데이터 교체
방법은 [data/README_data.ko.md](data/README_data.ko.md), 가중치 정보는
[weights/README_weights.ko.md](weights/README_weights.ko.md).

---

## 3. 실행

```bash
python demo.py                    # 번들 샘플 22쌍 복원 + 검출 → output/<타임스탬프>/
python evaluate.py                # 복원 전/후 mAP + PSNR/SSIM → output/Eval_<데이터셋>/
python train.py --epochs 30       # 복원 모델 재학습            → runs/<tag>/
python demo.py --live --screen 2  # 웹캠 + 프로젝터 리그         → output/<타임스탬프>/
python data/collect.py capture    # 직접 데이터 수집 (4단계)     → data/collected_<MMDD>/
python data/record.py --screen 2  # 모델 없이 투사 + 녹화        → data/recordings/
```

앞의 두 개는 `weights/` 에 체크포인트만 있으면 인자 없이 바로 돈다.

`--live` 는 실제 하드웨어가 필요하다. 프로젝터가 쏘는 스크린을 웹캠이 바라보는 구성.
`--screen N` 은 프로젝터가 연결된 모니터의 인덱스다. 그 모니터에 테두리 없는 전체화면
창을 띄워 클립을 재생하고, 그 화면을 웹캠이 찍는다. `0` 은 항상 주 모니터이므로
프로젝터를 보조 디스플레이로 붙였다면 보통 `1` 이나 `2` 다. 모르면 아무 값이나 주면 된다.
감지된 모니터 표가 제일 먼저 출력된다.

```
2 monitor(s) detected:
      --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

모델만 바꾸는 경우 입력·출력 경로는 그대로다.

```bash
python demo.py --detector ssd     # 검출기 교체 (ultralytics 불필요)
python demo.py --detector none    # 복원만, 검출 생략
python demo.py --limit 5          # 앞 5쌍만
```

각 스크립트의 입력·출력·전체 옵션 → [README_running.ko.md](README_running.ko.md).

---

## 4. YAML 설정

```
configs/*.yaml  <  --restoration-config / --detection-config 로 넘긴 YAML  <  CLI 플래그
```

| 파일 | 담는 것 |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | 복원 가중치 경로, 입력 크기, 구조 토글, 학습 하이퍼파라미터·손실 가중치 |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | 검출 백엔드, 백엔드별 가중치 경로, conf 임계값, 박스 크기 필터, 17개 클래스 목록 |

YAML 안의 상대 경로는 작업 디렉토리가 아니라 프로젝트 루트 기준으로 해석된다.
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

기본 제공 네트워크는 clean 을 직접 예측하지 않고 뺄 값(residual) 을 예측한다.

```
입력  (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
출력  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

이 규약이 원본 픽셀을 기본적으로 보존하고, 학습 중 외운 오브젝트를 그려 넣는 것을 막는다.
왜 그런지와 손실이 이를 어떻게 강제하는지 →
[weights/README_weights.ko.md](weights/README_weights.ko.md#residual-규약).

---

## 6. 라이선스

MIT. [LICENSE](LICENSE) 참고.

---

[English](README.md)

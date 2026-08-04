# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

스크린을 향한 카메라는 프로젝터가 쏘는 빛까지 함께 담는다. 이 프레임워크는 그 빛을 제거하고,
제거 전과 후의 객체 검출 성능을 측정한다.

복원 모델과 검출기 모두 작은 인터페이스 뒤에 있어, 파이프라인을 건드리지 않고 교체할 수 있다.

```
        카메라 촬영 (pro)  ─┐
                            ├─▶ 복원기 ─▶ residual ─▶ restored = pro − residual
   투사된 원본 (beam)  ────┘                            │
                                                        │
           detector(pro) ◀── 복원 전            복원 후 ──▶ detector(restored)
                  └────────── 비교: mAP / PSNR / SSIM ──────────┘
```

| 문서 | 담당 |
|---|---|
| [README_running.ko.md](README_running.ko.md) | `demo.py` · `evaluate.py` · `train.py` — 모든 옵션, 입력, 출력 |
| [data/README_data.ko.md](data/README_data.ko.md) | 데이터셋 구조, 파일명 규칙, 라벨, `collect.py`, `record.py` |
| [weights/README_weights.ko.md](weights/README_weights.ko.md) | 체크포인트 3개, 체크포인트 포맷, residual 규약 |

---

## 1. 설치

Python 3.9 이상.

```bash
git clone <repo> && cd ProRes-Det

conda create -n prores-det python=3.10 -y
conda activate prores-det

pip install -e "."              # 기본: ssd + none 백엔드
pip install -e ".[yolo]"        # + ultralytics       → --detector yolo
pip install -e ".[train]"       # + pytorch-msssim    → train.py
pip install -e ".[live]"        # + screeninfo        → --live (Windows는 불필요)
pip install -e ".[test]"        # + pytest
pip install -e ".[all]"         # 전부
```

기본 의존성: `torch torchvision opencv-python numpy PyYAML tqdm`.

대괄호는 따옴표로 감쌀 것. zsh에서는 glob 문자다.

선택 의존성은 필요한 함수 안에서 import한다. `--detector ssd`는 `ultralytics` 없이 돌고,
`--live`를 쓰지 않으면 `screeninfo`는 로드되지 않는다.

### GPU

**Windows에서 위 명령은 CPU 전용 torch를 설치한다.** 실패하지는 않는다. 복원이 프레임당
13ms 대신 약 380ms 걸릴 뿐이다. CUDA 빌드는 `requirements-cuda.txt`를 같이 넘긴다:

```bash
pip install -e ".[all]" -r requirements-cuda.txt
```

extra만으로는 불가능하다. 패키지는 자신이 어느 index에서 받아질지 지정할 수 없다. 그래서
index 줄과 버전 핀이 그 파일에 있다. `cu128`은 RTX 50 시리즈와 12.8 드라이버가 지원하는
이전 세대 전부를 덮는다. 구형 드라이버라면 파일 안의 `cu128`을 `cu121` 또는 `cu118`로 바꾼다.

받은 것 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.0+cu128 True      <- 정상
# 2.9.0+cpu   False     <- CPU 전용. -r requirements-cuda.txt 로 재설치
```

모든 엔트리포인트가 첫 줄에 결정된 디바이스를 출력하므로, 조용히 CPU로 떨어진 실행은 바로
드러난다:

```
device: cuda - NVIDIA GeForce RTX 5090, torch 2.9.0+cu128
device: cpu - torch 2.9.0+cpu is a CPU-only build, so restoration runs ~25x slower.
```

`--fp16`은 CUDA에서만 적용된다. CPU에서는 무시된다.

---

## 2. 실행

```bash
python demo.py                    # 번들 22쌍 복원 + 검출 → output/<timestamp>/
python evaluate.py                # 전후 mAP + PSNR/SSIM  → output/Eval_<dataset>/
python train.py --epochs 30       # 복원기 재학습         → runs/<tag>/
python demo.py --live --screen 2  # 웹캠 + 프로젝터 리그   → output/<timestamp>/
python data/collect.py capture    # 직접 수집 (4단계)     → data/collected_<MMDD>/
python data/record.py --screen 2  # 투사 + 녹화, 모델 없음 → data/recordings/
```

앞 두 개는 인자가 필요 없다. 샘플 데이터와 체크포인트 3개가 git에 추적되므로 클론 직후 바로
돌아간다.

**저장소 루트에서 실행할 것.** config 경로는 프로젝트 루트 기준으로 해석되지만, `train.data`
glob은 작업 디렉터리 기준으로 읽힌다.

모델을 바꿔도 입출력 경로는 그대로다:

```bash
python demo.py --detector ssd     # ultralytics 불필요
python demo.py --detector none    # 복원만
python demo.py --limit 5          # 앞 5쌍만
```

`--live`는 실제 하드웨어가 필요하다. 프로젝터가 쏘는 스크린을 향한 웹캠. `--screen N`은
프로젝터가 붙은 모니터 인덱스다. `0`은 항상 주 디스플레이라, 두 번째 디스플레이로 붙은
프로젝터는 보통 `1` 또는 `2`다. 모르면 아무 값이나 넣으면 된다. 모니터 표가 가장 먼저
출력된다.

```
2 monitor(s) detected:
      --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

모든 옵션과 입출력: [README_running.ko.md](README_running.ko.md).

---

## 3. 구조

```
ProRes-Det/
├── demo.py                   복원 → 검출, 엔드투엔드 (오프라인 / --live)
├── evaluate.py               검출 전후 비교 + 복원 품질
├── train.py                  복원 모델 파인튜닝
├── setup.py  requirements.txt  requirements-cuda.txt  LICENSE
│
├── projector_distortion/     ── 라이브러리
│   ├── __init__.py           공개 API
│   ├── cli.py                공용 인자, config 우선순위, 콘솔 엔트리포인트
│   ├── config.py             YAML 로딩 + 경로 해석
│   ├── data.py               샘플 탐색, 라벨 로딩, 학습 데이터셋
│   ├── configs/              기본 설정 (YAML)
│   ├── models/               복원 + 검출 모델
│   ├── pipeline/             오프라인 / 라이브 실행 루프
│   └── utils/                이미지, 시각화, 기록 헬퍼
│
├── tests/                    pytest 스위트, 102개
├── weights/                  체크포인트 3개, 51 MiB, git 추적
├── data/                     샘플 데이터셋 + collect.py / record.py, 추적 81 MiB
└── output/                   실행 산출물 (git 무시)
```

| 파일 | 역할 |
|---|---|
| [demo.py](demo.py) | 엔트리포인트. 인자 → 모델 → 오프라인/라이브 파이프라인 → 요약 |
| [evaluate.py](evaluate.py) | 엔트리포인트. GT 있는 샘플만 채점: P/R/F1/mAP + PSNR/SSIM |
| [train.py](train.py) | 엔트리포인트. 학습 루프 (L1 + perceptual + SSIM + wavelet) |
| [data/collect.py](data/collect.py) | 엔트리포인트. 리그로 데이터셋 수집, 4단계 |
| [data/record.py](data/record.py) | 엔트리포인트. 클립 투사 + 카메라 녹화. 가중치 로드 없음 |
| [cli.py](projector_distortion/cli.py) | 인자, config 우선순위, 모델 생성 — 세 스크립트 공용 |
| [config.py](projector_distortion/config.py) | YAML 로드/병합, 프로젝트 루트 기준 경로 해석 |
| [data.py](projector_distortion/data.py) | 파일명으로 pro/beam/clean/label 짝짓기, 학습 패치 데이터셋 |
| [models/base.py](projector_distortion/models/base.py) | `BaseRestorer` · `BaseDetector` · `Detection` · 검출기 레지스트리 |
| [models/restoration.py](projector_distortion/models/restoration.py) | 복원 네트워크, 구조 config, 체크포인트 I/O |
| [models/detection.py](projector_distortion/models/detection.py) | yolo · ssd 래퍼, 박스 크기 필터 |
| [pipeline/offline.py](projector_distortion/pipeline/offline.py) | 디스크의 이미지 쌍 일괄 처리. 하드웨어 불필요 |
| [pipeline/live.py](projector_distortion/pipeline/live.py) | 웹캠 + 프로젝터: 모니터 배치, 캘리브레이션, 워프, 워커 스레드 |
| [utils/image.py](projector_distortion/utils/image.py) | BGR ↔ 텐서, 리사이즈, PSNR / SSIM / IoU |
| [utils/visualize.py](projector_distortion/utils/visualize.py) | 박스 그리기, 2×2 패널, 캘리브레이션 오버레이 |
| [utils/recording.py](projector_distortion/utils/recording.py) | `RunRecorder` — 출력 디렉터리에 쓰이는 모든 것을 담당 |

`setup.py`는 `pdf-demo`, `pdf-evaluate`, `pdf-train`도 설치한다. 각각 루트 스크립트에
위임하므로 체크아웃의 editable 설치가 필요하다.

---

## 4. 설정

```
configs/*.yaml  <  --restoration-config / --detection-config 로 넘긴 YAML  <  CLI 플래그
```

| 파일 | 내용 |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | 복원 가중치, 입력 크기, 구조 토글, 학습 하이퍼파라미터, 손실 가중치, 학습 데이터 경로 |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | 백엔드, 백엔드별 가중치, conf 임계값, 박스 크기 필터, 17개 클래스명 |

일부만 덮으려면 바꿀 키만 담은 YAML을 넘긴다. 재귀 병합된다.

```yaml
# my_det.yaml
detector:
  backend: ssd
  conf: 0.4
```

```bash
python demo.py --detection-config my_det.yaml
```

config 안의 상대 경로는 프로젝트 루트 기준으로 해석된다. 단 `train.data`의 세 항목은
예외다. 그대로 glob되므로 작업 디렉터리 기준으로 읽힌다.

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

이게 변경 사항 전부다. `--detector mydet`이 즉시 동작하고, 파이프라인·기록·평가 코드는
그대로다.

### 복원기 교체

`BaseRestorer`를 상속하고 `restore(pro_bgr, beam_bgr) -> (restored_bgr, residual_bgr)`만
구현한다. 인터페이스는 이것뿐이다.

기본 네트워크는 clean 이미지가 아니라 **빼낼 residual**을 예측한다:

```
input   (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

이 규약이 왜 중요하고 손실 함수가 어떻게 강제하는지:
[weights/README_weights.ko.md](weights/README_weights.ko.md#residual-규약).

---

## 6. 테스트

```bash
python -m pytest -q
```

---

## 7. 라이선스

MIT. [LICENSE](LICENSE) 참고.

---

[English](README.md)

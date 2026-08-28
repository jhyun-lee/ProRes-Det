# ProRes-Det

**Pro**jector **Res**toration + **Det**ection

스크린을 향한 카메라는 프로젝터가 쏘는 빛까지 함께 담는다. 이 프레임워크는 그 빛을 제거하고,
제거 전과 후의 객체 검출 성능을 측정한다.

복원 모델과 검출기 모두 작은 인터페이스 뒤에 있어, 파이프라인을 건드리지 않고 교체할 수 있다.

![프레임워크: 라이브와 오프라인 입력이 하나의 프레임 쌍으로 모여 복원 → 검출 → 실행 기록기로 흐른다. 두 모델 단계 모두 YAML과 CLI 플래그로 설정하는 교체 가능한 백엔드다.](Image/Framework.png)

두 모드 모두 같은 `(distorted, light)` 프레임 쌍으로 수렴하므로 그 이후는 전부 공유된다.
`evaluate.py`는 복원 단계의 양쪽에서 검출기를 돌려 그 차이를 점수화한다 — mAP, PSNR, SSIM.

| 문서 | 담당 |
|---|---|
| [COMMANDS.ko.md](COMMANDS.ko.md) | 모든 명령어와 플래그를 담은 조회표, 대화형 단축키 포함 |
| [README_running.ko.md](README_running.ko.md) | `demo.py` · `evaluate.py` · `train.py` — 모든 옵션, 입력, 출력 |
| [projector_distortion/README_code.ko.md](projector_distortion/README_code.ko.md) | 모듈별 책임, 호출 그래프, 공개 API |
| [data/README_data.ko.md](data/README_data.ko.md) | 세 개의 split, 구조, 파일명 규칙, 라벨 포맷 두 가지, `Data.py` 전 스테이지 |
| [weights/README_weights.ko.md](weights/README_weights.ko.md) | 체크포인트 3개, 체크포인트 포맷, residual 규약 |

---

## 1. 설치

Python 3.9 이상.

```bash
git clone <repo> && cd ProRes-Det
conda create -n prores-det python=3.10 -y && conda activate prores-det

pip install -e ".[all]" -r requirements-cuda.txt    # GPU
pip install -e ".[all]"                             # CPU만
```

대괄호는 따옴표로 감쌀 것. zsh에서는 glob 문자다.

`[all]`은 모든 extra를 포함한다. 좁은 것도 있다: `yolo` 검출기용 `[yolo]`, `train.py`용
`[train]`, 非Windows `--live`용 `[live]`, pytest용 `[test]`.

GPU 적용 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.0+cu128 True
```

`requirements-cuda.txt`가 필요한 이유는 Windows에서 기본 PyPI `torch`에 CUDA가 없기
때문이다. 모든 엔트리포인트가 첫 줄에 결정된 디바이스를 출력하므로, CPU로 떨어진 실행은 바로
드러난다.

---

## 2. 실행

```bash
python demo.py                    # 번들 22쌍 복원 + 검출 → output/<timestamp>/
python evaluate.py                # 전후 mAP + PSNR/SSIM  → output/Eval_<dataset>/
python train.py --epochs 30       # 복원기 재학습         → runs/<tag>/
python demo.py --live             # 웹캠 + 프로젝터 리그   → output/<timestamp>/
python Data.py capture_warp       # 직접 수집 (아래 참고)  → data/Create_Data/warp_<MMDD>/
python Data.py record --screen 2  # 투사 + 녹화, 모델 없음 → data/recordings/
python demo.py --video-pair data/recordings/rec_<t>.mp4   # 그 녹화본 재생 → result.mp4
```

앞 세 개는 인자가 필요 없다. 샘플 데이터와 체크포인트 3개가 git에 추적되므로 클론 직후 바로
돌아간다. editable 설치를 하면 `pdf-demo` · `pdf-evaluate` · `pdf-train`도 PATH에 올라간다 —
같은 세 스크립트를 감싼 얇은 래퍼다.

번들 데이터셋은 `data/SampleData/` 아래 세 갈래로 분할되어 있고, 각 진입점이 자기 split을
기본값으로 쓴다:

| Split | 내용 | 기본값인 곳 |
|---|---|---|
| `sample_train` | 1,000쌍, 20장면, 라벨 없음 | `train.py` |
| `sample_eval` | 22쌍, 10장면, 박스 107개 | `demo.py`, `evaluate.py` |
| `sample_test` | 200쌍, 5장면, 박스 53개 | 홀드아웃 — `--input`/`--gt`로 지정해야 도달 |

장면이 겹치지 않으므로 `sample_train`으로 학습한 모델은 본 적 없는 프레임에서 채점된다.
자세한 내용과 라벨 포맷 두 가지: [data/README_data.ko.md](data/README_data.ko.md).

**저장소 루트에서 실행할 것.** config 경로는 프로젝트 루트 기준으로 해석되지만, `train.data`
glob은 작업 디렉터리 기준으로 읽힌다.

모델을 바꿔도 입출력 경로는 그대로다. 플래그가 아니라
[configs/detection.yaml](projector_distortion/configs/detection.yaml) 한 줄 수정이다:

```yaml
detector:
  backend: ssd      # ultralytics 불필요
  # backend: none   # 복원만
```

```bash
python demo.py --limit 5          # 앞 5쌍만
```

`--live`는 웹캠과 프로젝터가 필요하다. 어느 모니터에 투사할지, 어느 웹캠이 볼지,
프로젝터→카메라 지연이 얼마인지는
[configs/live.yaml](projector_distortion/configs/live.yaml)에서 온다. 실행마다가 아니라
머신마다 한 번 설정한다. 모니터 표는 시작할 때 출력되므로 `rig.screen`이 틀려도 실행 한 번
버리면 고칠 수 있다:

```
2 monitor(s) detected:
      screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
      screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

```yaml
# projector_distortion/configs/live.yaml
rig:
  screen: 1
  camera: 0
  offset: 6      # 프로젝터 → 카메라 지연, 프레임 단위
```

앞의 둘은 `--screen`·`--camera`로 실행 한 번만 덮을 수 있다. 잘못 짚었을 때 파일을 고치는
대신 이걸로 끝난다:

```bash
python demo.py --live --screen 2
```

리그와 모델이 같은 기계에 있을 필요도, 같은 주에 있을 필요도 없다. `Data.py record`는 모델을
하나도 올리지 않고 클립을 투사하며 카메라 시점을 raw mp4 하나로 녹화한다. `demo.py
--video-pair`는 그 파일을 다시 실행으로 되돌린다 — 정류하고, 카메라 프레임마다 그것을 만든
클립 프레임을 짝지어, 복원된 영상을 쓴다:

```bash
python Data.py record --screen 2 --seconds 30              # 리그에서
python demo.py --video-pair data/recordings/rec_<t>.mp4    # 나중에, 아무 데서나
```

인자는 하나면 된다. `record`가 mp4 옆에 쓰는 `.json`이 투사한 클립과 카메라가 실제로 낸
프레임률을 이미 담고 있어, 둘 다 따로 알려줄 필요가 없다. `--video-out restored panel`을 주면
한 번의 복원으로 두 가지 뷰를 모두 쓴다.

모든 옵션과 입출력: [README_running.ko.md](README_running.ko.md).

![촬영 리그: 위쪽의 단초점 프로젝터, 같은 암에 달린 카메라, 투사 패턴 아래 테이블에 놓인 물체들 — `(distorted, light)` 쌍은 정확히 이 구성에서 나온다.](Image/BeamPro.jpg)

직접 데이터를 수집하는 것은 `Data.py`이고, `data/` 아래 스크립트로 분기한다:

```bash
python Data.py                          # 스테이지 목록
python Data.py check                    # 모니터 + 웹캠. 제일 먼저
python Data.py make_light               # 영상 -> 투사할 light 프레임
python Data.py capture_warp --screen 2  # 투사·촬영·정류: 10장면 × 50 = 500쌍
```

`capture`와 `warp`를 따로 돌릴 수도 있다. 그래야 리그 없이 기하를 다시 잡을 수 있다. 전부
`data/Create_Data/` 아래로 떨어지고 `--input`·`--data-root`에 바로 쓸 수 있다. 스테이지·플래그·
세션 구조: [data/README_data.ko.md](data/README_data.ko.md).

![파이프라인: 1단계는 surface 촬영과 투사 프레임을 모아 ROI를 정합해 distorted 이미지를 만들고, 2단계는 투사 이미지와 distorted 이미지를 3레벨 U-Net에 넣어 빼낼 residual을 예측하며, 3단계는 distorted와 restored 양쪽에 검출기를 돌려 비교한다.](Image/Pipeline.png)

1단계는 `Data.py`, 2단계는 `train.py`와 복원기, 3단계는 `demo.py`와 `evaluate.py`다.
동봉된 샘플 데이터가 1단계 산출물을 이미 담고 있어, 클론하면 2단계부터 시작한다.

---

## 3. 기준 결과

이 레포가 배포 상태 그대로 내는 수치다. 믿고 넘어가는 대신 자기 실행과 대조하라고
적어둔다. 각 캡션의 명령으로 재현되며, split 지정 외에 플래그는 없고 명시된 곳
말고는 수정도 없다.

### 복원 전후 검출

`sample_test`가 대표 수치다. held-out이고, `sample_eval`의 9배 크기이며, 여기 있는
어떤 체크포인트도 이것으로 학습되지 않았다.

**`sample_test`** — 200쌍, 5장면, GT 박스 인스턴스 2,131개

| 백엔드 | 대상 | P | R | F1 | mAP@0.5 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| yolo | distorted | 0.8143 | 0.7611 | 0.7868 | 0.7804 | 1622 | 370 | 509 |
| yolo | **restored** | 0.9977 | 0.9986 | 0.9981 | **0.9952** | 2128 | 5 | 3 |
| | | | | | **+0.2148** | | | |
| ssd | distorted | 0.5428 | 0.7236 | 0.6203 | 0.7091 | 1542 | 1299 | 589 |
| ssd | **restored** | 0.9861 | 0.9977 | 0.9918 | **0.9968** | 2126 | 30 | 5 |
| | | | | | **+0.2877** | | | |

**`sample_eval`** — 22쌍, 10장면, GT 박스 인스턴스 225개

| 백엔드 | 대상 | P | R | F1 | mAP@0.5 | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| yolo | distorted | 0.8627 | 0.8933 | 0.8777 | 0.8843 | 201 | 32 | 24 |
| yolo | **restored** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 225 | 0 | 0 |
| | | | | | **+0.1157** | | | |
| ssd | distorted | 0.6793 | 0.8756 | 0.7650 | 0.8086 | 197 | 93 | 28 |
| ssd | **restored** | 0.9956 | 1.0000 | 0.9978 | **1.0000** | 225 | 1 | 0 |
| | | | | | **+0.1914** | | | |

건네받는 이미지 말고는 아무것도 공유하지 않는 두 아키텍처에서 모두 개선이 나온다.
검출기 하나의 특성이 아니라 복원의 특성이라는 뜻이다.

### 복원 품질

| Split | PSNR distorted | PSNR restored | 개선 | SSIM distorted | SSIM restored | 개선 |
|---|---|---|---|---|---|---|
| `sample_test` | 16.812 dB | 22.741 dB | +5.929 | 0.7189 | 0.8924 | +0.1735 |
| `sample_eval` | 13.760 dB | 26.928 dB | +13.168 | 0.7051 | 0.9227 | +0.2176 |

PSNR·SSIM은 검출기와 무관하므로 split당 한 행이 두 백엔드를 모두 덮는다.

두 split에서 방향이 엇갈리는 것을 보라. `sample_eval`은 PSNR 개선이 훨씬 크고,
`sample_test`는 mAP 개선이 훨씬 크다. 픽셀 충실도와 검출 가능성은 같은 축이 아니고,
이 프로젝트가 둘 다 채점하는 이유가 그것이다.

### 재현

```bash
python evaluate.py                                       # sample_eval, ssd
python evaluate.py --input data/SampleData/sample_test                    --gt    data/SampleData/sample_test   # sample_test, ssd
```

`ssd`가 기본 검출기이므로 위 두 명령이 `ssd` 행을 그대로 재현한다. `yolo` 행은 수정
한 번이 필요하다. 검출기는 플래그가 아니라 설정이기 때문이다 —
[configs/detection.yaml](projector_distortion/configs/detection.yaml)에서
`detector.backend: [yolo, ssd]`로 두면 항목마다 같은 리포트의 한 행이 된다. 이때
AGPL-3.0에 동의하는 것이 된다. [THIRD_PARTY.md](THIRD_PARTY.md) 참고.

수치는 이 레포에 추적된 체크포인트로, 배포 기본값인 `detector.conf: 0.25`와
`--iou 0.5`에서 나온 것이다. 복원은 결정적이라 같은 가중치로 다시 돌리면 표기된
자릿수까지 재현되고, GPU가 다르면 마지막 소수점이 움직일 수 있다.

---

## 4. 구조

```
ProRes-Det/
├── demo.py                   복원 → 검출, 엔드투엔드 (오프라인 / --live / --video-pair)
├── evaluate.py               검출 전후 비교 + 복원 품질
├── train.py                  복원 모델 파인튜닝
├── Data.py                   데이터셋 수집: data/ 전 스테이지의 단일 진입점
├── setup.py  requirements.txt  requirements-cuda.txt  LICENSE
│
├── projector_distortion/     라이브러리 → README_code.ko.md
│   ├── configs/              기본 설정 (YAML)
│   ├── models/               복원 + 검출 모델
│   ├── pipeline/             오프라인 / 라이브 / 재생 실행 루프
│   └── utils/                이미지, 시각화, 기록, 디스플레이 헬퍼
│
├── tests/                    pytest 스위트, 151개
├── weights/                  체크포인트 3개, 51 MiB, git 추적
├── data/                     SampleData/ (train · eval · test), live/ 클립, 그리고
│                             Data.py가 구동하는 수집 스크립트들. 추적 349 MiB
├── Image/                    위 그림들
└── output/                   실행 산출물 (git 무시)
```

모듈별 책임과 서로를 호출하는 방식:
[projector_distortion/README_code.ko.md](projector_distortion/README_code.ko.md).

---

## 5. 설정

```
configs/*.yaml  <  CLI 플래그
```

| 파일 | 내용 |
|---|---|
| [configs/restoration.yaml](projector_distortion/configs/restoration.yaml) | 복원 백엔드와 가중치, 입력 크기, 구조 토글, 학습 하이퍼파라미터, 손실 가중치, 학습 데이터 경로 |
| [configs/detection.yaml](projector_distortion/configs/detection.yaml) | 백엔드, 백엔드별 가중치, conf 임계값, 박스 크기 필터, 17개 클래스명 — LabelMe 라벨의 클래스 *이름*도 이 목록과 대조된다 |
| [configs/live.yaml](projector_distortion/configs/live.yaml) | `--live` 리그: 모니터, 웹캠, 카메라 백엔드, 프로젝터→카메라 지연, 캘리브레이션 검토 여부 |
| [configs/collect.yaml](projector_distortion/configs/collect.yaml) | `Data.py`가 읽는 전부: 세션 폴더, light 추출, 촬영 리그, 워프 기하, 녹화 |

모델 설정은 이 파일에서 바꾼다. 여기에 대응하는 CLI 플래그는 없다. 남은 플래그는 *이번
실행*에 관한 것들이다 — 어떤 데이터, 출력 위치, 얼마나 처리할지 — 여기에 `demo.py`의
`--device`와 `train.py`의 학습 하이퍼파라미터 3개가 더해진다. 모델에 관한 건 전부 파일을
직접 고친다:

```yaml
# projector_distortion/configs/detection.yaml
detector:
  backend: ssd
  conf: 0.4
```

config 안의 상대 경로는 프로젝트 루트 기준으로 해석된다. 단 `train.data`의 세 항목은
예외다. 그대로 glob되므로 작업 디렉터리 기준으로 읽힌다.

---

## 6. 모듈 교체

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

이게 변경 사항 전부다. `configs/detection.yaml`의 `detector.backend: mydet`이 즉시 동작하고, 파이프라인·기록·평가 코드는
그대로다.

### 복원기 추가

같은 모양에 레지스트리 하나 더:

```python
from projector_distortion.models import BaseRestorer, register_restorer

@register_restorer("myrest")
class MyRestorer(BaseRestorer):
    name = "myrest"

    def __init__(self, weights, device="cpu", input_size=(640, 360), **_):
        self.input_size = tuple(input_size)
        self.net = load_my_model(weights)

    def restore(self, distorted_bgr, light_bgr):
        restored = self.net(distorted_bgr)            # `light`은 제공될 뿐, 필수 아님
        return restored, residual_or_zeros
```

config가 가리키게 하면 바로 돈다:

```yaml
# projector_distortion/configs/restoration.yaml
model:
  backend: myrest
  weights: path/to.pt
```

백엔드 이름과 체크포인트 둘 다 config에서 온다. 어느 쪽도 플래그가 없다.

`light`(그 순간 프로젝터가 쏜 프레임)을 모든 복원기에 넘기는 이유는, 그것이 ProCam 리그에는
있고 일반 복원 환경에는 없는 유일한 신호이기 때문이다. 단일 이미지 백엔드는 그냥 무시하면
된다.

두 번째 반환값은 residual 뷰다. surface 이미지를 직접 예측하는 백엔드는 여기에 0을 넣으면
된다. 패널의 residual 타일과 `residual_mean`만 의미를 잃고, 채점은 그대로 된다.

기본 네트워크는 surface 이미지가 아니라 **빼낼 residual**을 예측한다:

```
input   (B, 6, H, W) = cat([distorted, light])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (distorted - residual).clamp(-1, 1)
```

이 규약이 왜 중요하고 손실 함수가 어떻게 강제하는지:
[weights/README_weights.ko.md](weights/README_weights.ko.md#residual-규약).

---

## 7. 테스트

```bash
python -m pytest -q
```

---

## 8. 라이선스

이 프로젝트의 소스·데이터·복원 가중치는 MIT다 — [LICENSE](LICENSE) 참고.

레포는 제3자 산출물도 각자의 라이선스로 함께 배포한다. 그중
`weights/detector_yolo11s.pt`는 Ultralytics YOLO11 체크포인트의 파인튜닝이라
**AGPL-3.0**이다. 기본 설정은 이 파일을 쓰지 않는다. 배포 기본값
`detector.backend`가 `ssd`(torchvision의 BSD 라이선스 SSDLite)이므로, 손대지 않은
상태의 실행은 MIT/BSD/Apache 코드만 거친다. `yolo`를 고르면 AGPL-3.0에 동의하는 것이
된다.

전체 내역·업스트림 링크·재구현한 아키텍처 출처:
[THIRD_PARTY.md](THIRD_PARTY.md).

---

## 9. 인용

연구에 도움이 됐다면 인용해달라. 릴리스마다 Zenodo가 DOI를 발급하므로, 실제로 돌린
버전에 해당하는 것을 쓰면 된다.

```bibtex
@software{lee_prores_det,
  author  = {Lee, JeongHyun},
  title   = {{ProRes-Det: Projector Restoration and Detection}},
  year    = {2026},
  version = {1.0.0},
  doi     = {TODO},
  url     = {https://github.com/jhyun-lee/ProRes-Det}
}
```

---

## 10. 연락

질문·버그 신고·출처 표기 정정:
[ljhyun0821@gnu.ac.kr](mailto:ljhyun0821@gnu.ac.kr), 또는 레포에 이슈를 남겨달라.

---

[English](README.md)

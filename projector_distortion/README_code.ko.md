# projector_distortion/

모듈별 책임, 실행이 모듈을 타고 흐르는 경로, 공개 API.

스크립트 옵션은 [README_running.ko.md](../README_running.ko.md)에 있다. 새 검출기·복원기를
붙이는 방법은 [README.ko.md](../README.ko.md#5-모듈-교체)에 있다.

- [호출 그래프](#호출-그래프)
- [파일](#파일)
- [계약](#계약)
- [공개 API](#공개-api)
- [테스트](#테스트)

---

## 호출 그래프

루트 스크립트 3개는 얇다. 인자를 파싱하고 넘긴다.

```
demo.py / evaluate.py / train.py
   │
   ├─ cli.add_common_args        공용 플래그
   ├─ cli.build_models           YAML + 플래그 → (restorer, detector, info)
   │     ├─ config.load_config   configs/*.yaml, 병합
   │     ├─ models.build_restorer
   │     └─ models.build_detector
   │
   ├─ data.find_samples          파일명으로 pro ↔ beam ↔ clean ↔ label 짝짓기
   │  data.index_triplets        학습 전용: 완성된 삼중쌍
   │
   ├─ pipeline.run_offline       ─┐
   │  pipeline.run_live          ─┤ 루프
   │     └─ pipeline.process_sample
   │            ├─ restorer.restore_full
   │            ├─ detector(distorted) · detector(restored)
   │            ├─ models.filter_detections
   │            └─ utils.visualize.draw_detections / grid_2x2
   │
   └─ utils.recording.RunRecorder   output/<run>/ 에 쓰이는 모든 것
```

`evaluate.py`는 `run_offline`을 건너뛰고 `process_sample`을 직접 돌린다. 프레임을 기록하는
대신 샘플별로 채점하기 때문이다.

`pipeline/__init__.py`는 `run_live`를 지연 프록시로 노출한다. `live`를 import하면 오프라인
실행에는 전혀 필요 없는 ctypes와 Win32 배관이 함께 딸려온다.

---

## 파일

### 최상위

| 파일 | 담당 |
|---|---|
| [`__init__.py`](__init__.py) | 공개 API 재노출, `__version__` |
| [`cli.py`](cli.py) | 공용 argparse 그룹, config 우선순위, `build_models`, 디바이스 결정, `run_dir`, `pdf-*` 콘솔 엔트리포인트 |
| [`config.py`](config.py) | `load_config` (YAML + 재귀 병합), `resolve_path` (`PROJECT_ROOT` 기준), `pick` (CLI가 YAML보다 우선) |
| [`data.py`](data.py) | 파일명 → id 파싱, 레이아웃 감지, `find_samples`, `index_triplets`, YOLO 라벨 로딩, `TripletPatchDataset` |

### `configs/`

| 파일 | 내용 |
|---|---|
| [`configs/restoration.yaml`](configs/restoration.yaml) | `model` (weights, input_size, fp16) · `ablation` (구조 + 용량) · `train` (데이터 경로, 하이퍼파라미터, 손실 가중치) |
| [`configs/detection.yaml`](configs/detection.yaml) | `detector` (backend, conf, imgsz, 박스 게이트) · 백엔드별 `weights` · `names` (17클래스) · `evaluate.iou_threshold` |

### `models/`

| 파일 | 담당 |
|---|---|
| [`models/base.py`](models/base.py) | `BaseRestorer`, `BaseDetector`, `Detection`, `NullDetector`, `@register_detector` 레지스트리 |
| [`models/restoration.py`](models/restoration.py) | `RestorationConfig` + 토글 10개(`TOGGLES`), 네트워크(`LayerNorm2d` → `SimpleGate` → `CALayer` → `NAFBlock` → `RestormerLikeBlock` → `RestorationNet`), 체크포인트 저장/로드, `RestormerLikeRestorer` |
| [`models/detection.py`](models/detection.py) | `YoloDetector` (ultralytics), `SsdDetector` (torchvision), `build_detector`, `filter_detections`, 기본 `CLASS_NAMES` |

네트워크는 3-level U-Net이다. 이름과 달리 MDTA 어텐션이 없어서 Restormer가 아니라 NAFNet +
squeeze-excite다. fully convolutional이므로 180×320 학습 패치와 640×360 추론 크기가 동일하게
동작한다.

`ssd`는 라벨을 정규화한다. torchvision 헤드는 COCO id(1..N, 0 = 배경)를 내므로 `detect()`에서
1을 뺀다. ultralytics는 이미 0-based다.

### `pipeline/`

| 파일 | 담당 |
|---|---|
| [`pipeline/offline.py`](pipeline/offline.py) | `FrameResult`, `process_sample` (한 쌍 엔드투엔드), `build_panel`, `run_offline`. 하드웨어 불필요 |
| [`pipeline/live.py`](pipeline/live.py) | Win32로 모니터 열거·테두리 없는 배치, 흑/백 플래시 캘리브레이션, homography + 워프, 복원/검출 워커 스레드, 라이터 스레드, stride 자동 결정 |

`live.py`는 스레드 3개로 돈다. 메인 루프가 투사·촬영하고, 워커가 복원·검출하고, 라이터가
인코딩한다. 큐 깊이는 `MAX_IN_FLIGHT = 3`, `MAX_PENDING_WRITES = 8`. 분석 stride는
`STRIDE_WARMUP = 12` 프레임에서 앞 `STRIDE_DISCARD = 2`개를 버린 뒤 한 번만 고정된다. CUDA
autotuning 때문에 1번 프레임이 정상 상태의 약 50배가 걸리기 때문이다.

창 제목: `Projector_Display`, `Combined_View`, `PreWarp_Debug`, `Warp_FirstFrame`.

### `utils/`

| 파일 | 담당 |
|---|---|
| [`utils/image.py`](utils/image.py) | `read_bgr`, `resize`, `bgr_to_tensor` / `tensor_to_bgr` / `residual_to_bgr`, `psnr`, `ssim`, `iou`, `IMAGE_EXT` |
| [`utils/visualize.py`](utils/visualize.py) | `draw_detections`, `draw_ground_truth`, `caption`, `side_by_side`, `grid_2x2`, `panel_size`, `draw_quad`, `warp_before_after` |
| [`utils/recording.py`](utils/recording.py) | `RunRecorder`, `FRAME_KINDS`, `parse_kinds`, `estimate_footprint_mb` |

`ssim`을 여기서 직접 구현한다(가우시안 윈도우 11×11, sigma 1.5). 지표 계산이 scikit-image나
pytorch-msssim을 끌어오지 않도록 하기 위해서다. `residual_to_bgr`은 mean |residual|을 JET
히트맵으로 컬러맵한다. 그래서 이미지에서 스칼라를 복원할 수 없고, `restore_full()`이 그 값을
따로 반환한다.

`RunRecorder`가 무엇을 어디에 쓸지 결정한다. `FRAME_KINDS`는
`("distorted", "restored", "panel")`이고, `distorted`·`restored`는 `captures/`,
`panel`은 `frames_all/`, 캘리브레이션 이미지는 `calib/`로 간다.

---

## 계약

작은 타입 4개가 모듈 사이의 모든 것을 운반한다.

```python
# models/base.py
class BaseRestorer:
    input_size: tuple[int, int] = (640, 360)
    def restore(self, pro_bgr, beam_bgr) -> tuple[restored_bgr, residual_bgr]: ...
    def restore_full(self, pro_bgr, beam_bgr) -> tuple[restored, residual, mean_abs]: ...

class BaseDetector:
    def detect(self, bgr) -> list[Detection]: ...

@dataclass(frozen=True)
class Detection:
    cls_id: int; name: str; conf: float; box: Sequence[int]   # (x1, y1, x2, y2), 픽셀
```

```python
# data.py
@dataclass(frozen=True)
class Sample:
    name_id: str; pro: str; beam: str
    clean: str | None = None      # 선택: 채점에만 필요
    label: str | None = None
```

```python
# pipeline/offline.py
@dataclass
class FrameResult:
    frame_id, name_id
    beam, distorted, distorted_det, restored, restored_det, residual   # BGR 배열
    residual_mean, det_distorted, det_restored, t_restore, t_detect
    clean, gt_boxes                                                    # GT 없으면 None / []
    def metrics(self) -> dict   # distorted·restored의 psnr/ssim, clean 없으면 None
```

`restore_full`이 있는 이유는 호출자가 forward를 두 번 돌리지 않고 한 번에 mean |residual|을
받게 하기 위해서다. 기본 구현은 틀린 값 대신 `0.0`을 반환하고, 값을 아는 서브클래스가
오버라이드한다.

`FrameResult`의 필드명은 `RunRecorder`가 읽는 이름이다. 하나를 바꾸면 기록이 깨진다.

---

## 공개 API

```python
from projector_distortion import build_restorer, build_detector
from projector_distortion.data import find_samples
from projector_distortion.pipeline import process_sample

restorer = build_restorer("weights/restorer_restormerlike.pt")
detector = build_detector("ssd", "weights/detector_ssdlite.pth")
root = "data/sample_input"               # pro/ beam/ + 채점용 clean/ labels/
for i, s in enumerate(find_samples(root, root)):
    r = process_sample(s, restorer, detector, frame_id=i)
    print(s.name_id, len(r.det_distorted), "->", len(r.det_restored))
```

`__init__.py`가 재노출하는 것: `BaseRestorer`, `BaseDetector`, `Detection`,
`RestorationConfig`, `build_restorer`, `build_detector`, `detector_names`,
`filter_detections`, `CLASS_NAMES`, `load_config`, `resolve_path`, `PROJECT_ROOT`.

서브모듈 `__init__.py`는 더 많이 노출한다. `models`는 네트워크 내부 구성과 체크포인트 헬퍼를,
`utils`는 위에 나열된 헬퍼 전부를 노출한다.

---

## 테스트

102개, 하드웨어 불필요.

| 파일 | 커버 |
|---|---|
| [`../tests/conftest.py`](../tests/conftest.py) | 픽스처 `root` / `bgr_image` / `pro_beam`, 체크포인트·선택 모듈 부재 시 skip |
| [`../tests/test_pipeline.py`](../tests/test_pipeline.py) | 파일명 id, 샘플 탐색, PSNR/SSIM/IoU, `RunRecorder` 동작, 삼중쌍 인덱싱, `average_precision`, argparse 기본값, `device_note`, `requirements-cuda.txt` 핀 |
| [`../tests/test_restoration.py`](../tests/test_restoration.py) | `RestorationConfig`와 tag, 모든 토글의 빌드 + 1스텝 학습, forward 크기, 체크포인트 왕복, 기본 가중치 |
| [`../tests/test_detection.py`](../tests/test_detection.py) | 레지스트리, 라벨 정규화, 박스 크기 게이트, `best_per_class`, 실제 체크포인트로 두 백엔드 검증 |
| [`../tests/test_collect.py`](../tests/test_collect.py) | 리그 없이 되는 `collect.py`: 코너 정렬, 경계 리샘플링, `boundary` vs `homography` 동등성, `warp`·`beam` 단계 출력 |

```bash
python -m pytest -q
```

---

[English](README_code.md) · [← README](../README.ko.md)

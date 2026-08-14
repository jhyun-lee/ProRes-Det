# data/

데이터셋에 관한 모든 것: 무엇이 들어 있는지, 파일명이 어떻게 세 시점을 묶는지, 라벨 포맷,
그리고 데이터를 더 만드는 도구들.

`demo.py` / `evaluate.py` / `train.py`의 스크립트 옵션은
[README_running.ko.md](../README_running.ko.md)에 있다.

- [무엇이 들어 있나](#무엇이-들어-있나)
- [세 개의 split](#세-개의-split)
- [파일명 규약](#파일명-규약)
- [레이아웃](#레이아웃)
- [라벨 포맷](#라벨-포맷)
- [`Data.py` — 리그로 데이터셋 만들기](#datapy--리그로-데이터셋-만들기)
- [`Data.py record` — 모델 없이 투사 + 녹화](#datapy-record--모델-없이-투사--녹화)
- [실데이터로 교체](#실데이터로-교체)

---

## 무엇이 들어 있나

바로 실행되는 데이터셋. `SampleData/` 아래에 세 갈래로 분할되어 있다. git에 추적되므로
클론 직후 바로 있다.

```
data/
├── check.py                디스플레이 목록 출력 + 웹캠 프리뷰
├── make_light.py           영상 -> 프로젝터가 쏠 light 프레임
├── capture.py              프로젝터 + 웹캠 -> raw 촬영본
├── warp.py                 raw 촬영본 -> 정류된 정렬 쌍
├── record.py               클립 투사 + 카메라 화면 녹화 (모델 불필요)
├── common.py               collect.yaml 접근, 세션 폴더, 공용 헬퍼
├── SampleData/             학습 / 검증 / 테스트로 나뉜 데이터셋
│   ├── sample_train/       train.py가 학습하는 대상            180 MiB
│   │   ├── distorted/  1,000장, 640×360
│   │   ├── light/        990장, 1280×720 · 854×480 · 856×480
│   │   └── surface/       20장, 640×360     — labels/ 없음, 학습에 필요 없다
│   ├── sample_eval/        demo.py와 evaluate.py의 기본값       4.2 MiB
│   │   ├── distorted/     22장, 640×360
│   │   ├── light/         22장, 1280×720 및 854×480
│   │   ├── surface/       10장, 640×360
│   │   └── labels/        10개, YOLO txt, 박스 107개
│   ├── sample_test/        홀드아웃. evaluate.py --input 으로만 도달   44 MiB
│   │   ├── distorted/    200장, 640×360
│   │   ├── light/        198장, 854×480
│   │   ├── surface/        5장, 640×360
│   │   └── labels/         7개, LabelMe json, 그 5개 장면에 박스 53개
├── live/                   투사 원본. 역할별로 분리                  121 MiB
│   ├── train_light_1.mp4   `Data.py make_light`가 학습용 light 프레임으로 만드는 클립
│   │                       4,262프레임 @30fps = 2.4분, 854×480, 15 MiB
│   ├── train_light_2.mp4   5,381프레임 @30fps = 3.0분, 856×480, 7 MiB
│   ├── train_light_3.mp4  24,272프레임 @30fps = 13.5분, 854×480, 44 MiB
│   ├── test_light.mp4      홀드아웃 — demo.py --live 와 `Data.py record`의 기본값
│   │                       5,858프레임 @30fps = 3.3분, 854×480, 55 MiB
│   └── BaseBackGround.jpg  캘리브레이션 중 표시할 배경, 1280×960
├── Create_Data/            수집 세션이 떨어지는 곳 (git 무시)
└── recordings/             `Data.py record` 출력 위치 (git 무시)
```

저장소 루트의 `Data.py`가 위 다섯 스테이지 스크립트의 진입점이다 (`common.py`는 스테이지가
아니라 공용 배관). 이들은 패키지가 아니라 독립 프로그램이라 각각 단독 실행도 된다.

| 경로 | 용도 |
|---|---|
| `<split>/distorted/` | 모델 입력 ch 0:3 — 투사된 스크린의 카메라 시점 |
| `<split>/light/` | 모델 입력 ch 3:6 — 프로젝터가 내보낸 프레임 |
| `<split>/surface/` | 학습 타깃, PSNR/SSIM 기준 |
| `<split>/labels/` | mAP 기준 — YOLO txt 또는 LabelMe json |

각 split은 자기완결적이다. `surface/`과 `labels/`가 `distorted/` 옆에 있으므로 `--input`과
`--gt`가 같은 경로를 받는다. 둘 다 선택 사항이다. 없어도 실행은 되고, 채점만 못 한다.

이미지 크기는 맞출 필요가 없다. 전부 모델의 `input_size`(기본 640×360)로 리사이즈된다.
그래서 `light/`에 1280×720, 854×480, 856×480이 섞여 있어도 별도 처리가 없다.

### 투사 원본도 split이 나뉜다

`live/`에는 프로젝터가 재생할 클립이 들어 있고, split 구분이 여기까지 이어진다.

| 클립 | 공급 대상 | 사용처 |
|---|---|---|
| `train_light_1.mp4` · `_2` · `_3` | 학습 split | `Data.py make_light`가 3개 전부를 기본값으로 |
| `test_light.mp4` | 홀드아웃 split | `demo.py --live`와 `Data.py record`의 기본값 |

이 분리가 홀드아웃을 홀드아웃답게 만든다. `Data.py`로 모은 셋은 `test_light.mp4`와
투사 프레임을 하나도 공유하지 않으므로, 그 셋으로 학습한 체크포인트는 채점에 쓰이는 광원을
본 적이 없다. `--src`와 `--clip`으로 양쪽 다 덮어쓸 수 있다.

`data/`는 추적 기준 349 MiB고 그중 121 MiB가 `live/`다. 클립은 **수집**하거나 `--live`를
돌릴 때만 필요하다 — 클립에서 뽑은 프레임은 이미 `SampleData/*/light/`에 들어 있어서 학습과
평가는 클립 없이 돌아간다. 둘 다 안 쓰면 지워도 된다.

---

## 세 개의 split

장면이 겹치지 않는다. `sample_train`의 어떤 `surfaceId`도 `sample_eval`이나 `sample_test`에
나타나지 않는다. 즉 학습한 체크포인트는 본 적 없는 장면에서 채점된다.

| Split | 쌍 | 장면 | 라벨 | 읽는 쪽 |
|---|---|---|---|---|
| `sample_train` | 1,000 (994개가 light 짝을 가짐) | 20 | 없음 | `train.py` — [restoration.yaml](../projector_distortion/configs/restoration.yaml)의 `train.data` 경유 |
| `sample_eval` | 22 | 10 | YOLO txt, 박스 107개 | `demo.py`·`evaluate.py` — 둘 다의 기본값 |
| `sample_test` | 200 | 5 | LabelMe json, 박스 53개 | 기본으로는 아무것도 안 읽는다. `--input`/`--gt`로 지정 |

```bash
python train.py                                          # sample_train
python evaluate.py                                       # sample_eval
python evaluate.py --input data/SampleData/sample_test \
                   --gt    data/SampleData/sample_test   # sample_test
```

개수에 대한 두 가지 주의. `sample_train`의 캡처 6장은 이 split에 없는 light 프레임을
가리키므로 경고와 함께 스킵되고 triplet 994개가 남는다. `sample_test/labels/`에는 이 split에
`surface`/`distorted`가 없는 장면의 주석 2개(`Ori0428095917`, `Ori0531135441`)가 더 들어 있다.
짝이 없는 라벨은 애초에 로드되지 않으므로 박스 수가 80이 아니라 53이다.

`sample_train`과 `sample_test`는 파일명 리네임 이전에 수집되어 `projected_` /
`output_video_` / `Ori` 표기를 쓰고, `sample_eval`은 현재 표기(`distorted_` / `light_` /
`surface_`)를 쓴다. 읽는 방식은 동일하다 — 아래 참조.

---

## 파일명 규약

한 시점의 세 뷰가 파일명 안의 id로 묶인다. 개수나 경로가 어디에도 하드코딩되어 있지 않으므로,
이 규약만 지키면 실데이터가 그대로 들어온다.

```
distorted_0409001429_0404023332_294_75.jpg
          └───┬────┘ └──────┬───────┘
            surfaceId         lightId

  → sample_eval/surface/surface_0409001429.jpg                  (정답 스크린)
  → sample_eval/labels/surface_0409001429.txt                 (검출 정답)
  → sample_eval/light/light_0404023332_294_75.jpg  (프로젝터가 내보낸 프레임)
```

| 역할 | 파일명 | 리네임 이전 표기 | 모델에서 |
|---|---|---|---|
| `distorted` | `distorted_<surfaceId>_<lightId>.jpg` | `projected_…` | 입력 ch 0:3 |
| `light` | `light_<lightId>.jpg` | `output_video_…` | 입력 ch 3:6 |
| `surface` | `surface_<surfaceId>.jpg` | `Ori<surfaceId>.jpg` | 학습 타깃 / PSNR·SSIM 기준 |
| `label` | `surface_<surfaceId>.txt` 또는 `.json` | `Ori<surfaceId>.…` | 검출 mAP 기준 |

- `surfaceId`에 `_`를 쓰지 말 것. 첫 `_` 앞까지가 surfaceId로 잡힌다.
- `lightId`에는 `_`가 들어가도 된다. 위 예시는 `0404023332_294_75`다.
- `surface` 하나가 여러 `distorted`를 받치는 것이 정상이다. `sample_eval`은 surface 10장이
  distorted 22장을(하나당 1~3장), `sample_test`는 5장이 200장을 받친다.
- `light`은 `distorted`와 1:1이다.
- 인식 확장자: `.jpg` `.jpeg` `.png` `.bmp`.
- 짝 `light`이 없는 `distorted`는 경고와 함께 스킵된다. 실행이 실패하지는 않는다.
- `surface`과 `label`은 선택이다. `surface`이 없으면 PSNR/SSIM만 빠지고, `evaluate.py`는 둘 다
  있는 샘플만 채점한다.
- 리네임 이전 표기는 읽기만 하고 쓰지는 않는다. `sample_train`·`sample_test`가 그 표기이며,
  이미 수집해 둔 세션이 계속 동작하는 이유이기도 하다.

---

## 레이아웃

어느 폴더가 존재하는지로 두 가지가 자동 감지된다.

| 이름 | 폴더 |
|---|---|
| `flat` (여기서 사용) | `distorted/` `light/` (+ `surface/`) |
| `legacy-flat` | `pro/` `beam/` (+ `clean/`) — 리네임 이전에 수집한 세션 |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/SampleData/sample_eval                # flat
python demo.py --input /path/to/WarpData_0520                     # research
```

한 폴더에 이미지가 그냥 흩어져 있으면 `mixed`로 처리되고, `distorted_`와 `light_`
접두사로 나뉜다.

---

## 라벨 포맷

두 가지를 읽으며 확장자로 구분한다. 둘 다 같은 픽셀 박스로 귀결되므로, split이 어느 쪽을
쓰든 `--gt`에 따로 알려줄 필요가 없다.

### YOLO `.txt` — `sample_eval`

박스당 한 줄, `<cls_id> <cx> <cy> <w> <h>`, 모두 0–1로 정규화.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id`는 `0..16`이고
[configs/detection.yaml](../projector_distortion/configs/detection.yaml)의 `names` 순서를
따른다 — 과일 11개(Apple … Watermelon), 그다음 동물 6개(Cat … Snake). `sample_eval` 라벨에
17개 클래스가 모두 등장한다.

### LabelMe `.json` — `sample_test`

LabelMe 주석 도구가 쓴 그대로다. 박스는 `rectangle` shape이고, 두 코너 점이 **절대 픽셀**
좌표이며, 클래스는 id가 아니라 *이름*이다.

```json
{
  "imageWidth": 640, "imageHeight": 360,
  "shapes": [
    {"label": "BlueBerry", "shape_type": "rectangle",
     "points": [[63.64, 27.97], [144.56, 105.43]]}
  ]
}
```

- 점은 그 파일 자신의 `imageWidth`/`imageHeight` 기준으로 읽혀 실행 시 `input_size`로
  재스케일된다. 480×270으로 다시 돌려도 재주석이 필요 없다.
- 읽을 때 코너를 정렬하므로, 우하단→좌상단으로 그린 박스도 문제없다.
- `label`은 탐지기 자신의 클래스 이름과 대조된다 — YOLO 체크포인트의 목록, SSD면
  `detection.yaml`의 `names`. 그 목록에 없는 이름은 경고와 함께 스킵한다. 탐지기가 낼 수도
  없는 클래스를 채점하면 영원히 false negative로만 잡히기 때문이다.
- `rectangle` shape만 쓴다. polygon·point는 무시된다.

한 장면에 `surface_<id>.txt`와 `surface_<id>.json`이 둘 다 있으면 `.txt`가 이긴다.

---

## `Data.py` — 리그로 데이터셋 만들기

위 샘플 데이터셋도 이 방식으로 만들었다. 저장소 루트의 `Data.py`가 `data/` 아래
스크립트로 분기하고, 각 스테이지는 자기 `--help`를 갖는다.

```bash
python Data.py                     # 스테이지 목록
python Data.py capture --help      # 해당 스테이지 옵션
```

| 스테이지 | 스크립트 | 하는 일 | 리그 필요 |
|---|---|---|---|
| `check` | `check.py` | 디스플레이 목록 + 웹캠 프리뷰 | 필요 |
| `make_light` | `make_light.py` | 영상 → light 프레임 | 불필요 |
| `capture` | `capture.py` | 투사하며 촬영 → raw 촬영본 | 필요 |
| `capture_warp` | `capture.py --warp` | 위와 같되 촬영하며 정류까지 | 필요 |
| `warp` | `warp.py` | raw 촬영본 → 정렬된 쌍 | 불필요 |
| `record` | `record.py` | 클립 투사 + 녹화, 모델 불필요 | 필요 |

`make_light`와 `warp`는 순수 파일 작업이라, 리그 기기에서 촬영하고 정류는 다른 데서 해도
된다. 스크립트 직접 실행도 동일하다 — `python data/warp.py --review`와
`python Data.py warp --review`는 같은 프로그램이다.

```bash
python Data.py check                       # 모니터 + 웹캠
python Data.py make_light                  # 영상 -> light 프레임
python Data.py capture_warp --screen 2     # 10장면 촬영, 촬영하며 정류
```

또는 두 단계를 나눈다. 이렇게 해야 나중에 기하를 다시 잡을 수 있다:

```bash
python Data.py capture --screen 2
python Data.py warp
```

`make_light`는 기본으로 `train_light_*.mp4` 3개를 읽는다. `test_light.mp4`는 의도적으로
빠져 있어서, 여기서 모은 셋은 홀드아웃 split과 투사 프레임을 하나도 공유하지 않는다.

light 프레임 추출이 별도 단계인 것은 의도된 설계다 — `capture`는 프레임을 직접 자르지 않고
이미 `projected/`에 있는 것에서 뽑아 쓴다. **캐시도 "이미 처리됨" 판단도 없다.** 실행할 때마다
새 타임스탬프가 파일명에 박히므로, 같은 클립에 `make_light`를 두 번 돌리면 건너뛰는 게 아니라
모든 프레임이 한 벌 더 저장된다.

### 어디에 떨어지나

전부 [configs/collect.yaml](../projector_distortion/configs/collect.yaml)의 `session.dir`
아래로 간다. 기본값은 `data/Create_Data`이고 git은 무시한다:

```
data/Create_Data/
├── projected/       light_<lightId>.jpg                 [make_light] 프로젝터가 내보내는 것
├── raw_<MMDD>/      surface/surface_<surfaceId>.jpg     [capture]    카메라 프레임, 정류 전
│                    distorted/distorted_<surfaceId>_<lightId>.jpg
│                    collect_meta.json                                설정·개수·코너
└── warp_<MMDD>/     surface/surface_<surfaceId>.jpg     [warp]       정류됨, 640×360
                     distorted/distorted_<surfaceId>_<lightId>.jpg
                     debug/<surfaceId>_warp.jpg                       전/후 확인용
                     collect_meta.json
```

`projected/`에 날짜가 없는 것은 의도다. 한 클립이 내놓는 프레임은 어느 날이든 같으므로
날짜별 사본은 디스크만 쓰고 얻는 게 없다. 촬영본에 날짜가 붙는 것은 반대 이유다 — 날이
바뀌면 리그가 움직이고 장면도 다시 꾸며진다.

warp 폴더는 warp를 돌린 날이 아니라 **그것이 나온 raw 폴더**를 따라 이름이 붙는다. 그래서
간격이 얼마든 `raw_0813`과 `warp_0813`은 같은 촬영본을 가리킨다. `capture`는 오늘자 raw를
채우고, `warp`는 `--raw`로 지정하지 않으면 가장 최신 raw를 잡는다.

`warp_<MMDD>/`는 현재 표기를 쓰는 `flat` 레이아웃이라 변환할 게 없다:

```bash
python demo.py  --input data/Create_Data/warp_0813
python train.py --data-root data/Create_Data/warp_0813
```

### 플래그가 아닌 설정

실행마다 실제로 바뀌는 것만 커맨드라인에 남았다. 해상도, 카메라 백엔드, 색상 회전 대역, 워프
기하, 코덱은 리그나 명명 규약의 속성이라
[configs/collect.yaml](../projector_distortion/configs/collect.yaml)에서 한 번 설정한다:

| 블록 | 담당 |
|---|---|
| `session:` | `dir`, `projected` / `raw_` / `warp_` 이름, `jpeg_quality` (95) |
| `light:` | `src`, `step`, `augment`, `size` (1280×720), `hue_bands`, `first_video_index`, `seed` |
| `capture:` | `screen`, `camera`, `cam_backend`, `background`, `rounds`, `limit`, `settle_ms`, `flush`, `round_settle`, `preview_every`, `seed` |
| `warp:` | `mode`, `work_size` (1280×720), `final_size` (640×360), `points` (20), `inset` (2), `debug` |
| `record:` | `clip`, `out_dir`, `screen`, `camera`, `cam_backend`, `background`, `codec`, `fps_probe`, `preview_every`, `max_queued` |

`live.yaml`의 한 섹션이 아니라 별도 파일인 이유는, 이 스크립트들이 torch 없이 돌고 파이프라인
패키지를 import하지 않기 때문이다.

카메라에 요청하는 해상도와 프레임레이트만 예외로 1280×960 @30fps 고정이다 —
`projector_distortion/pipeline/live.py`의 `CAM_WIDTH` / `CAM_HEIGHT` / `CAM_FPS`이고
`demo.py --live`와 공유한다. 드라이버가 요청을 무시하는 일이 잦아서, 카메라가 실제로 열린
값을 시작 시 출력하고 `collect_meta.json`에도 남긴다.

### 장면을 얼마마다 바꿔야 하나

`capture`는 **라운드**마다 `surface`를 1장 찍는다. 즉 스크린 위 물체는 라운드당 한 번
바꾸고, `--rounds`가 곧 만들어야 할 장면 개수다.

각 라운드는 `projected/` 폴더 **전체에서** `--limit`장을 새로 무작위 추출한다. 10라운드면
같은 목록을 10번 쓰는 게 아니라 서로 다른 10벌로 비춘다. 기본값은 `--rounds 10 --limit 50` =
**10장면에 500쌍**.

시간 제한은 없다. surface 샷이 `s` 키까지 블로킹하므로 장면 꾸미는 데 얼마가 걸리든
상관없다. 아래 숫자는 그 뒤 자동으로 도는 부분만이다.

| 명령 | 장면 수 | 장면당 촬영 | 쌍 | 라운드당 촬영 시간 |
|---|---|---|---|---|
| `capture` (기본값) | 10 | 50 | **500** | 약 1분 |
| `capture --rounds 20 --limit 25` | 20 | 25 | 500 | 약 30초 |
| `capture --rounds 4 --limit 250` | 4 | 250 | 1,000 | 약 5분 |

"라운드당 촬영 시간"은 촬영 루프만 계산한 값이고, 기본 `capture.settle_ms` 1200ms 기준이다.
실행 자체도 시작 전에 예상치를 출력한다. 장면을 꾸미는 시간은 사람 몫이고 여기 안 들어간다.

`--limit 0`은 풀의 **모든** 프레임을 뜻한다. 기본 `make_light` 후라면 수천 장이고, 한 장면에
몇 시간 촬영에 장면 다양성은 최저가 된다. 복원 학습에는 잘못된 교환이다. 모델에 필요한 건
여러 surface지, 한 surface 위의 여러 광원이 아니다.

추출은 시드를 걸지 않는다. 그래서 다른 날 세션을 또 돌리면 같은 조합을 반복하지 않고
다양성이 쌓인다. 재현이 필요하면 `collect.yaml`의 `capture.seed`를 설정한다.

### `check`

디스플레이 목록을 출력하고 웹캠을 연다. `capture`가 첫 시도가 되지 않게 하는 단계다.

| 옵션 | 기본값 |
|---|---|
| `--camera N` | `capture.camera` |

```
  --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
  --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

### `make_light` — 영상 → light 프레임

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--src <path>...` | `light.src` — `train_light_*.mp4` 3개 | 영상 파일과 폴더를 섞어서 여러 개 |
| `--out <dir>` | `<session.dir>/projected` | — |
| `--step N` | `light.step` (`30`) | N프레임마다 유지. 30이면 30fps에서 초당 1장 |
| `--augment <mode>` | `light.augment` (`full`) | 원본 프레임당 사본 수: `full` = 원본 + 반전 + 색상 회전 4개(6배), `invert` = 2배, `none` = 1배 |
| `--limit N` | `0` | 영상당 최대 N개 *원본* 프레임 (증강 전 기준) |

파일명은 `light_<tag>_<videoIdx>_<frameIdx>_<variant>.jpg`이고 `variant`는 `0`, `invert`,
또는 색상 회전 각도다. 폴더는 `projected/`인데 파일은 `light_*`로 남는 이유가 있다.
`projected_`는 이미 *distorted* 파일의 리네임 이전 접두사라서, 그렇게 이름 붙인 light
프레임은 distorted로 파싱된다.

프레임은 `light.size`(1280×720), `session.jpeg_quality`(95)로 저장되고 첫 영상 인덱스는
`light.first_video_index`(`1000`)다. `warp`에서 모델 입력 크기로 다시 줄이므로 여기서는
그보다 넉넉하기만 하면 된다. 색상 회전 각도를 고정하려면 `light.seed`를 설정한다.

### `capture` — 프로젝터 + 웹캠 → raw 촬영본

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--raw <dir>` | `<session.dir>/raw_<MMDD>`, 오늘자 | 특정 촬영 폴더에 덧붙일 때 |
| `--screen N` | `capture.screen` (`1`) | 프로젝터가 붙은 모니터. `check`가 표를 출력한다 |
| `--camera N` | `capture.camera` (`0`) | — |
| `--rounds N` | `capture.rounds` (`10`) | 장면 세팅 수. 각 라운드는 새 surface 샷으로 시작 |
| `--limit N` | `capture.limit` (`50`) | 라운드당 뽑을 light 프레임 수. 풀 전체에서 무작위 추출. `0`이면 전부 |
| `--settle-ms N` | `capture.settle_ms` (`1200`) | light 프레임을 몇 ms 띄워 두고 촬영본을 확정할지 |
| `--warp` | off | 촬영하면서 정류까지. `Data.py capture_warp`가 이 플래그 |

**라운드 동작.** `capture`는 배경을 투사한 뒤 `s` 입력을 기다린다. 그 샷이 `surface`이 된다 —
아무것도 투사되지 않은 장면이므로, 물체를 배치하고 프레임 밖으로 나간 뒤 눌러야 한다. 그
타임스탬프가 `surfaceId`가 되고 해당 라운드의 모든 촬영본이 그것을 물고 간다. 그래서 여러
`distorted`가 하나의 `surface`을 가리키게 된다. 그다음 각 light 프레임을 한 번 투사하고 한 번
촬영한다. `--rounds N`은 새 장면으로 반복한다.

**프레이밍 프리뷰.** 장면을 꾸미는 동안 스크린 경계가 실시간으로 검출되고 정류된 뷰가 카메라
뷰 옆에 표시된다. 워프가 안 잡히는 상황을 장면이 이미 해체된 warp 단계가 아니라 리그 앞에서
고치기 위한 것이다.

| 키 | 동작 |
|---|---|
| `s` · `enter` · `space` | surface 샷을 찍고 라운드 시작 |
| `c` | 프레임을 고정하고 네 코너를 직접 클릭. 검출이 못 찾을 때 |
| `r` | 클릭한 코너를 버리고 자동 검출로 복귀 |
| `q` · `esc` | 중단 |

`s`를 누른 시점에 유효한 코너가 `collect_meta.json`에 기록되고, `warp`는 다시 검출하는 대신
그것을 재사용한다 — 정류된 프리뷰를 사람이 보면서 확정한 값이고, 경계의 좋고 나쁨을 판단할 수
있는 순간은 그때뿐이기 때문이다. `warp --redetect`는 이 값을 무시한다.

**타이밍.** `capture.settle_ms`(1200ms)와 `capture.flush`(3)가 촬영본을 그 원인이 된 프레임에
맞춰준다. settle 구간 동안 sleep하지 않고 카메라를 계속 읽는다. 웹캠은 큐에 있던 것을 그대로
돌려주고, 자동 노출은 드라이버가 실제로 넘겨준 프레임에서만 조정되기 때문이다. 연속된 light
프레임은 서로 반전이거나 색상 회전본이라 거의 매 스텝이 큰 밝기 변화이고, AE 루프는 1초 가까이
필요하다. `distorted`가 *이전* light 프레임처럼 보이면 `settle_ms`를 올린다.

**`capture_warp`.** `--warp`를 주면 surface 샷 시점의 코너로 라운드마다 rectifier를 하나
만들고, 정류된 `surface/`·`distorted/`·`debug/`를 raw 옆의 `warp_<MMDD>/`에 쓴다. raw는 어느
쪽이든 남으므로, 잘못 나온 라운드는 리그로 돌아가지 않고 `Data.py warp --review`로 다시 할 수
있다. 코너를 못 쓰는 라운드는 raw만 촬영되고 그렇게 보고된다.

### `warp` — 정렬된 쌍으로 정류

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--raw <dir>` | 가장 최신 `raw_<MMDD>` | 정류할 촬영본 |
| `--out <dir>` | `<session.dir>/warp_<MMDD>`, raw 이름을 따름 | — |
| `--mode boundary` | `warp.mode` (기본) | 4코너 + 측정된 에지 휨 |
| `--mode homography` | — | 코너만. 평평한 스크린 + 깔끔한 경계일 때 |
| `--mode tps` | — | 레거시 thin-plate spline. `opencv-python<5` 필요 |
| `--limit N` | `0` | 최대 N개 장면만 처리 |
| `--review` | off | 장면마다 보여주고 판정을 기다림 |
| `--manual` | off | 자동 검출을 건너뛰고 모든 장면에서 코너 클릭 |
| `--redetect` | off | `capture`가 기록한 코너를 무시하고 다시 검출 |

정류는 `warp.work_size`(1280×720)에서 `warp.points`(20)개 경계 대응점으로, 밝은 투사 테두리를
피해 경계를 `warp.inset`(2)px 당겨서 수행하고, `warp.final_size`(640×360)로 저장한다.

**존재 이유.** 카메라에서 스크린은 사다리꼴이고 물체 스케일도 촬영마다 다르다. 이 단계 전에는
학습이 불가능하고, `raw_<MMDD>/`만 있는 세션은 로더가 인식하는 레이아웃 자체가 아니다.
`warp`는 surface 샷에서 스크린 경계를 찾고, 그 장면의 surface 프레임 *과* 해당 촬영본 전부를
동일한 매핑으로 정류한다. 결과적으로 쌍이 픽셀 그리드를 공유하게 되고, 남는 차이는 투사광뿐이다.

`boundary`는 비스듬히 본 평면 스크린에 대해 정확하고, 에지가 휘어도 유효하다. OpenCV 5는
`tps`가 필요한 shape 모듈을 제거했다.

**`--review`.** 경계 검출은 사진에 대한 컨투어 휴리스틱이다. 대체로 맞고, 틀릴 때는 요약
줄로는 드러나지 않는 방식으로 틀린다. `--review`는 장면마다 검출된 사각형과 정류 결과를
나란히 띄우고 기다린다:

| 키 | 동작 |
|---|---|
| `enter` · `space` · `a` | 이 장면 수락 |
| `m` | 대신 네 코너를 직접 클릭 |
| `r` | 검출기 재실행 |
| `s` | 이 장면 건너뛰기 |
| `A` | 이후 모든 장면을 확인 없이 수락 |
| `q` | 중단. 이미 쓴 장면은 유지된다 |

`--review` 없이는 경계를 못 찾은 장면이 보고되고 건너뛰어진다. 어느 쪽이든, 긴 세션에
들어가기 전에 짧게 시험하고 `warp_<MMDD>/debug/<surfaceId>_warp.jpg`를 확인할 것. 검출된
경계, 샘플링된 점, 정류 결과가 한 장에 함께 나온다.

### 라벨은 수집되지 않는다

`evaluate.py`로 mAP을 채점하려면 `surface/`를 손으로 라벨링해
`<session>/labels/surface_<surfaceId>.txt`로 — LabelMe로 주석했다면 `.json`으로 — 넣어야 한다.
복원 학습과 PSNR/SSIM에는 라벨이 필요 없고, 그래서 `sample_train`에는 라벨이 아예 없다.

---

## `Data.py record` — 모델 없이 투사 + 녹화

다른 스테이지는 *데이터셋*을 만든다 — 정류되고 id가 맞춰진 정지 이미지 쌍. `record`는
*영상*을 만든다. 클립을 원래 fps로 투사하고 카메라 시점을 mp4 하나로 녹화한다. 복원도 검출도
없고 가중치도 로드하지 않으므로, 가중치가 없는 기기에서도 돈다.

다른 스테이지가 공유하는 세션 폴더 밖에 있다. mp4 하나는 학습 세트를 이루는
distorted/light/surface 삼중쌍이 아니므로 `warp`나 `train.py`가 집어갈 것이 없다.
체크포인트가 나오기 전에 원본 왜곡 영상을 확보하거나, 리그가 끝까지 동작하는지 확인하는 데
쓴다.

| | 기본값 | 변경 |
|---|---|---|
| 투사 클립 | `record.clip` — `data/live/test_light.mp4` | `--clip <path>` |
| 배경 | `record.background` — `data/live/BaseBackGround.jpg` | `collect.yaml` |
| 출력 | `record.out_dir/rec_<MMDDHHMMSS>.mp4` + 옆에 `.json` | `--out <path>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--screen N` | `record.screen` (`1`) | 프로젝터가 붙은 모니터. `0` = 주 디스플레이 |
| `--camera N` | `record.camera` (`0`) | — |
| `--loop` | off | 클립 끝에서 멈추지 않고 다시 시작 |
| `--seconds F` | `0` | N초 후 중지. `0` = 무제한 |
| `--warp` | off | 한 번 캘리브레이션하고 정류된 스크린을 녹화 |

mp4는 `record.codec`(`mp4v`), 카메라 해상도 그대로이고, 레이트는 `record.fps_probe`(24)
프레임에 걸쳐 측정되며, 프리뷰는 `record.preview_every`(5) 투사 프레임마다 갱신되고,
인코더 대기열은 `record.max_queued`(32) 프레임까지 허용된다.

```bash
python Data.py record --screen 2 --seconds 30
python Data.py record --clip data/live/train_light_1.mp4 --loop
python Data.py record --warp
```

창이 뜨는 즉시 녹화가 시작된다 — 키 입력 없음. `q`로 조기 종료되고, 클립 끝과 `--seconds`도
종료 조건이다.

헤더 fps는 요청값이 아니라 **실측값**이다 — 지정 플래그를 일부러 두지 않았다. 웹캠은 요청과
다른 레이트를 내놓고 또 다른 숫자를 보고하는 일이 흔하며, 헤더가 틀리면 재생이 배속이나
슬로우가 된다. 프로브 프레임은 레이트가 확정될 때까지 버퍼되므로 측정에 영상 손실이 없다.

mp4 옆의 `.json`에는 실행이 실제로 사용한 클립·카메라·모니터 설정, 측정된 레이트, 인코더가
버린 프레임 수가 기록된다.

---

## 실데이터로 교체

```bash
# 1) data/SampleData/ 아래에 같은 구조로 채우고 그냥 실행
python demo.py
python evaluate.py

# 2) 또는 원본 데이터셋을 직접 지정
python demo.py --input /mnt/.../WarpData_0520
```

split 이름은 어디에도 하드코딩되어 있지 않다. `--input`/`--gt`와 `--data-root`는 인식되는
레이아웃의 아무 폴더나 받으므로, 실데이터는 자기 디렉터리 이름과 분할 방식을 그대로 유지해도
된다. `data/SampleData/`는 번들 데이터셋이 놓인 위치일 뿐이다.

학습은 세 디렉터리를
[configs/restoration.yaml](../projector_distortion/configs/restoration.yaml)의 `train.data`에서
읽는다. 각 항목은 디렉터리, glob, 리스트를 받는다. 우선순위, `--data-root` 폴백, surface 타깃
탐색 순서:
[README_running.ko.md](../README_running.ko.md#학습-데이터가-오는-곳).

---

[English](README_data.md) · [← README](../README.ko.md)

# data/

데이터셋에 관한 모든 것: 무엇이 들어 있는지, 파일명이 어떻게 세 시점을 묶는지, 라벨 포맷,
그리고 데이터를 더 만드는 도구 두 개.

`demo.py` / `evaluate.py` / `train.py`의 스크립트 옵션은
[README_running.ko.md](../README_running.ko.md)에 있다.

- [무엇이 들어 있나](#무엇이-들어-있나)
- [세 개의 split](#세-개의-split)
- [파일명 규약](#파일명-규약)
- [레이아웃](#레이아웃)
- [라벨 포맷](#라벨-포맷)
- [`collect.py` — 리그로 데이터셋 만들기](#collectpy--리그로-데이터셋-만들기)
- [`record.py` — 모델 없이 투사 + 녹화](#recordpy--모델-없이-투사--녹화)
- [실데이터로 교체](#실데이터로-교체)

---

## 무엇이 들어 있나

바로 실행되는 데이터셋. `SampleData/` 아래에 세 갈래로 분할되어 있다. git에 추적되므로
클론 직후 바로 있다.

```
data/
├── collect.py              프로젝터 + 웹캠으로 직접 데이터셋 수집
├── record.py               클립 투사 + 카메라 화면 녹화 (모델 불필요)
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
│   └── sample_video/       짧은 클립 2개, BeamVideo 대체용        23 MiB
│                           4,262 / 5,381프레임 @30fps, 854×480 / 856×480
├── live/                   demo.py --live 와 record.py 의 입력
│   ├── BeamVideo.mp4       프로젝터로 재생할 클립
│   │                       5,858프레임 @30fps = 3.3분, 854×480, 55 MiB
│   └── BaseBackGround.jpg  캘리브레이션 중 표시할 배경, 1280×960
└── recordings/             record.py 출력 위치 (git 무시)
```

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

`data/`는 추적 기준 305 MiB다. `live/BeamVideo.mp4`(55 MiB)와 `SampleData/sample_video/`
(23 MiB)는 `--live`와 `record.py`에만 쓰이므로, 둘 다 안 쓰면 지워도 된다.

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

## `collect.py` — 리그로 데이터셋 만들기

위 샘플 데이터셋도 이 방식으로 만들었다. 4단계, 순서대로. `check`와 `capture`는 프로젝터와
웹캠을 구동하고, `light`과 `warp`는 순수 파일 작업이라 어디서든 돈다.

```bash
python data/collect.py check                                   # 모니터 + 웹캠
python data/collect.py light   --src data/live/BeamVideo.mp4  # 영상 -> light 프레임
python data/collect.py capture --screen 2 --rounds 3            # 투사하며 촬영
python data/collect.py warp                                     # 정류해 쌍으로
```

모든 산출물이 세션 폴더 하나에, 저장소의 나머지가 그대로 읽는 레이아웃으로 떨어진다.
`--root`로 옮길 수 있고, 기본값은 하루 한 폴더다. 같은 단계를 다시 돌리면 기존 세션을 확장한다.

```
data/collected_<MMDD>/
├── light/      light_<lightId>.jpg                    [light]    프로젝터가 내보내는 것
├── raw/        surface/surface_<surfaceId>.jpg        [capture]  카메라 프레임, 정류 전
│               distorted/distorted_<surfaceId>_<lightId>.jpg     [capture]
├── surface/    surface_<surfaceId>.jpg                [warp]     정류됨, 640×360
├── distorted/  distorted_<surfaceId>_<lightId>.jpg    [warp]     정류됨, 640×360
├── debug/      <surfaceId>_warp.jpg                   [warp]     전/후 확인용
└── collect_meta.json                                  단계별 설정과 개수
```

```bash
python demo.py  --input data/collected_0803
python train.py --data-root data/collected_0803
```

### 카메라 플래그

`check`와 `capture`가 공유한다.

| 옵션 | 기본값 |
|---|---|
| `--camera N` | `0` |
| `--cam-width` · `--cam-height` | `1280` · `960` |
| `--cam-fps` | `30` |
| `--cam-backend` | `auto` · `any` · `dshow` · `msmf` · `v4l2` |

### `check`

디스플레이 목록을 출력하고 웹캠을 연다. `capture`가 첫 시도가 되지 않게 하는 단계다.

```
  --screen 0 -> 2560x1440 at (0,0) (primary)  \\.\DISPLAY1
  --screen 1 -> 1920x1080 at (2560,0)         \\.\DISPLAY2
```

### `light` — 영상 → light 프레임

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--src <path>` | `data/live/BeamVideo.mp4` | 영상 파일, 또는 영상이 든 폴더 |
| `--out <dir>` | `<root>/light` | — |
| `--step N` | `30` | N프레임마다 유지. 30이면 30fps에서 초당 1장 |
| `--size W H` | `1280 720` | `0 0`이면 원본 해상도 유지 |
| `--quality N` | `95` | JPEG 품질 |
| `--limit N` | `0` | 영상당 최대 N프레임 |
| `--video-index N` | `1000` | 첫 영상의 파일명 인덱스 |

### `capture` — 프로젝터 + 웹캠 → raw 촬영본

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--screen N` | `1` | 프로젝터가 붙은 모니터. `check`가 표를 출력한다 |
| `--light <dir>` | `<root>/light` | — |
| `--background <path>` | `data/live/BaseBackGround.jpg` | surface 샷을 찍는 동안 투사할 이미지 |
| `--rounds N` | `1` | 장면 세팅 수. 각 라운드는 새 surface 샷으로 시작 |
| `--limit N` | `0` | 라운드당 light 프레임 수 |
| `--shuffle` | off | light 순서 랜덤화. 짧은 라운드도 클립 전체를 커버 |
| `--seed N` | `42` | `--shuffle`용 |
| `--settle-ms N` | `150` | 프레임 표시 후 촬영까지 대기 |
| `--flush N` | `3` | 촬영 전 버릴 버퍼 카메라 프레임 수 |
| `--round-settle F` | `2.0` | surface 샷과 첫 투사 사이 대기 초 |
| `--preview-every N` | `10` | N프레임마다 촬영 프리뷰 갱신 |
| `--jpeg-quality N` | `95` | — |

**라운드 동작.** `capture`는 배경을 투사한 뒤 `s` 입력을 기다린다. 그 샷이 `surface`이 된다 —
아무것도 투사되지 않은 장면이므로, 물체를 배치하고 프레임 밖으로 나간 뒤 눌러야 한다. 그
타임스탬프가 `surfaceId`가 되고 해당 라운드의 모든 촬영본이 그것을 물고 간다. 그래서 여러 `distorted`가
하나의 `surface`을 가리키게 된다. 그다음 각 light 프레임을 한 번 투사하고 한 번 촬영한다.
`--rounds N`은 새 장면으로 반복한다.

**타이밍.** `--settle-ms`와 `--flush`가 촬영본을 그 원인이 된 프레임에 맞춰준다. `distorted`가
*이전* light 프레임처럼 보이면 둘 다 올린다.

### `warp` — 정렬된 쌍으로 정류

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--warp boundary` | 기본 | 4코너 + 측정된 에지 휨 |
| `--warp homography` | — | 코너만. 평평한 스크린 + 깔끔한 경계일 때 |
| `--warp tps` | — | 레거시 thin-plate spline. `opencv-python<5` 필요 |
| `--surface <dir>` · `--distorted <dir>` | `<root>/raw/surface` · `<root>/raw/distorted` | — |
| `--points N` | `20` | 경계 대응점 수 (tps와 디버그 오버레이용) |
| `--inset N` | `2` | 밝은 투사 테두리를 피해 경계를 당길 픽셀 |
| `--work-size W H` | `1280 720` | 정류 작업 해상도 |
| `--final-size W H` | `640 360` | 저장 해상도 — 모델 입력 |
| `--limit N` | `0` | 최대 N개 장면만 처리 |
| `--no-debug` | off | `<root>/debug/`의 전/후 오버레이 생략 |

**존재 이유.** 카메라에서 스크린은 사다리꼴이고 물체 스케일도 촬영마다 다르다. 이 단계 전에는
학습이 불가능하다. `warp`는 surface 샷에서 스크린 경계를 찾고, 그 장면의 surface 프레임 *과*
해당 촬영본 전부를 동일한 매핑으로 정류한다. 결과적으로 쌍이 픽셀 그리드를 공유하게 되고,
남는 차이는 투사광뿐이다.

`boundary`는 비스듬히 본 평면 스크린에 대해 정확하고, 에지가 휘어도 유효하다. OpenCV 5는
`tps`가 필요한 shape 모듈을 제거했다.

긴 세션에 들어가기 전에 짧게 시험하고 `<session>/debug/<surfaceId>_warp.jpg`를 확인할 것. 검출된
경계, 샘플링된 점, 정류 결과가 한 장에 함께 나온다.

### 라벨은 수집되지 않는다

`evaluate.py`로 mAP을 채점하려면 `surface/`를 손으로 라벨링해
`<session>/labels/surface_<surfaceId>.txt`로 — LabelMe로 주석했다면 `.json`으로 — 넣어야 한다.
복원 학습과 PSNR/SSIM에는 라벨이 필요 없고, 그래서 `sample_train`에는 라벨이 아예 없다.

---

## `record.py` — 모델 없이 투사 + 녹화

`collect.py`는 *데이터셋*을 만든다 — 정류되고 id가 맞춰진 정지 이미지 쌍. `record.py`는
*영상*을 만든다. 클립을 원래 fps로 투사하고 카메라 시점을 mp4 하나로 녹화한다. 복원도 검출도
없고 가중치도 로드하지 않으므로, 가중치가 없는 기기에서도 돈다.

체크포인트가 나오기 전에 원본 왜곡 영상을 확보하거나, 리그가 끝까지 동작하는지 확인하는 데
쓴다.

| | 기본값 | 변경 |
|---|---|---|
| 투사 클립 | `data/live/BeamVideo.mp4` | `--clip <path>` |
| 배경 | `data/live/BaseBackGround.jpg` | `--background <path>` |
| 출력 | `data/recordings/rec_<MMDDHHMMSS>.mp4` + 옆에 `.json` | `--out <path>` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--screen N` | `1` | 프로젝터가 붙은 모니터. `0` = 주 디스플레이 |
| `--loop` | off | 클립 끝에서 멈추지 않고 다시 시작 |
| `--seconds F` | `0` | N초 후 중지. `0` = 무제한 |
| `--max-frames N` | `0` | N프레임 투사 후 중지 |
| `--start-delay F` | `0` | 녹화 시작 전 배경을 유지할 초 |
| `--fps F` | `0` | mp4 헤더 fps. `0` = 카메라 실제 레이트 측정 |
| `--fps-probe N` | `24` | 레이트 측정에 쓰는 프레임 수. 버퍼되며 버려지지 않음 |
| `--codec <fourcc>` | `mp4v` | `avc1`, `XVID`, … |
| `--rec-size W H` | `0 0` | 인코딩 전 리사이즈. `0 0`이면 카메라 해상도 유지 |
| `--warp` | off | 한 번 캘리브레이션하고 정류된 스크린을 녹화 |
| `--manual-calib` | off | `--warp`와 함께: 네 코너를 직접 클릭 |
| `--calib-settle F` | `0.8` | 캘리브레이션 플래시 후 대기 초 |
| `--no-preview` | off | 작은 카메라 프리뷰 창 생략 |
| `--preview-every N` | `5` | N프레임마다 프리뷰 갱신 |

카메라 플래그는 `collect.py`와 동일하다.

```bash
python data/record.py --screen 2 --seconds 30
python data/record.py --clip data/SampleData/sample_video/mIni_Video_1.mp4 --loop
python data/record.py --warp --rec-size 640 360
```

창이 뜨는 즉시 녹화가 시작된다 — 키 입력 없음. `q`로 조기 종료되고, 클립 끝·`--seconds`·
`--max-frames`도 종료 조건이다.

헤더 fps는 요청값이 아니라 **실측값**이다. 웹캠은 요청과 다른 레이트를 내놓고 또 다른 숫자를
보고하는 일이 흔하며, 헤더가 틀리면 재생이 배속이나 슬로우가 된다. 프로브 프레임은 레이트가
확정될 때까지 버퍼되므로 측정에 영상 손실이 없다.

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

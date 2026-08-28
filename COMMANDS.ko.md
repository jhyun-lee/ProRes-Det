# 명령어 레퍼런스

이 프로젝트가 받는 모든 명령어를 한자리에 모았다. 옵션은 그것을 소유한 스크립트별로
묶었고, 기본값은 실제로 적용되는 값이다 — 플래그가 아니라 설정 파일에서 오는 것까지
포함해서.

어떤 설정이 *왜* 거기 있는지는 [README_running.ko.md](README_running.ko.md)를 봐라.
이 페이지는 설명이 아니라 조회표다.

- [한눈에](#한눈에)
- [설치](#설치)
- [`demo.py` — 복원 → 검출](#demopy--복원--검출)
- [`evaluate.py` — 복원 전후 점수 비교](#evaluatepy--복원-전후-점수-비교)
- [`train.py` — 복원기 재학습](#trainpy--복원기-재학습)
- [`Data.py` — 데이터셋 수집](#datapy--데이터셋-수집)
- [대화형 프롬프트 단축키](#대화형-프롬프트-단축키)
- [테스트](#테스트)
- [출력물이 떨어지는 곳](#출력물이-떨어지는-곳)
- [플래그가 없는 설정](#플래그가-없는-설정)

---

## 한눈에

전부 레포 루트에서 실행한다.

| 명령어 | 리그 필요 | 하는 일 |
|---|:---:|---|
| `python demo.py` | — | 번들 eval 22쌍 복원 + 검출 |
| `python demo.py --video-pair <mp4>` | — | 녹화본을 복원 영상으로 재생 |
| `python demo.py --live` | **필요** | 웹캠 + 프로젝터 실시간 구동 |
| `python evaluate.py` | — | 복원 전후 채점: mAP, PSNR, SSIM |
| `python train.py` | — | `sample_train`으로 복원기 재학습 |
| `python Data.py` | — | 수집 스테이지 목록 |
| `python Data.py check` | **필요** | 모니터 목록, 웹캠 미리보기 |
| `python Data.py make_light` | — | 영상 → 투사할 light 프레임 |
| `python Data.py capture` | **필요** | 투사·촬영 → raw 캡처 |
| `python Data.py capture_warp` | **필요** | 위와 같되 촬영하면서 정류 |
| `python Data.py warp` | — | raw 캡처 → 정류·정합된 쌍 |
| `python Data.py record` | **필요** | 클립 투사 + 녹화, 모델 없음 |
| `python -m pytest -q` | — | 테스트 151개 |

`demo` · `evaluate` · `train` 앞 세 개는 인자가 필요 없다. 샘플 데이터와 체크포인트
3개가 git에 추적되므로 클론 직후 바로 돌아간다.

editable 설치를 하면 콘솔 스크립트 3개가 PATH에 올라간다. 같은 파일을 감싼 얇은
래퍼라 플래그도 동일하다:

| 콘솔 스크립트 | 동일한 것 |
|---|---|
| `pdf-demo` | `python demo.py` |
| `pdf-evaluate` | `python evaluate.py` |
| `pdf-train` | `python train.py` |

---

## 설치

Python ≥ 3.9.

| 명령어 | 설치되는 것 |
|---|---|
| `pip install -e ".[all]" -r requirements-cuda.txt` | 전부, GPU 빌드 torch |
| `pip install -e ".[all]"` | 전부, CPU 전용 torch |
| `pip install -e ".[yolo]"` | `yolo` 검출 백엔드 |
| `pip install -e ".[train]"` | `train.py`에 필요한 것 |
| `pip install -e ".[live]"` | 비Windows에서의 `--live` |
| `pip install -e ".[test]"` | pytest |

대괄호는 따옴표로 감싸라 — zsh에서는 글롭 문자다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.9.0+cu128 True
```

Windows의 기본 PyPI `torch`에는 CUDA가 없어서 `requirements-cuda.txt`가 필요하다.
모든 진입점이 첫 줄에 자기가 잡은 device를 찍으므로, 조용히 CPU로 떨어진 실행은
스스로 그렇다고 말한다.

---

## `demo.py` — 복원 → 검출

한 스크립트에 모드 3개. 서브커맨드가 아니라 플래그로 갈린다: 아무것도 없으면 오프라인,
`--video-pair`면 재생, `--live`면 리그. `--live`와 `--video-pair`를 같이 주면 거부된다.

### 공통

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input <dir>` | `data/SampleData/sample_eval` | `distorted/` + `light/` 쌍 폴더 (오프라인) |
| `--output <dir>` | `output` | 실행 폴더가 생기는 곳 |
| `--limit N` | `0` | 최대 N프레임만 처리. `0` = 전부 |
| `--no-detect` | off | 복원만. `detection.yaml`을 고치지 않고 |
| `--save-every N` | `1` | N프레임마다 이미지 저장. `0` = csv만 |
| `--device cuda\|cpu` | 있으면 cuda | — |
| `--clip <mp4>` | 아래 참고 | 투사 클립. 라이브는 재생하고, 재생 모드는 이것과 짝짓는다 |

`--clip` 기본값은 모드마다 다르다. 라이브는 `data/live/test_light.mp4`를 틀고, 재생
모드는 녹화본 `.json`이 실제로 투사했다고 적어둔 것을 쓴다.

### 오프라인 (기본)

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--video` | off | 2×2 패널을 `result.mp4`로도 기록 |

```bash
python demo.py
python demo.py --input data/SampleData/sample_test
python demo.py --limit 5 --save-every 0
python demo.py --input /path/to/pairs --video
```

### 재생 — `--video-pair`

mp4 2개 입력, 1개 이상 출력. 하드웨어 불필요.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--video-pair <mp4>` | — (필수) | 재생할 `Data.py record` mp4 |
| `--video-out <kind>...` | `panel` | `panel`(2×2) · `restored`(복원본만) · `compare`(좌우 비교). 여러 개 가능 |
| `--offset-ms F` | `200` | 프로젝터 → 카메라 지연. `rig.offset`(6)을 `CAM_FPS`(30)로 환산 |
| `--manual-calib` | off | 화면 네 모서리를 자동 검출 대신 클릭 |
| `--no-review` | off | 캘리브레이션 확인 창 생략. 무인 실행용 |

```bash
python demo.py --video-pair data/recordings/rec_0828145219.mp4
python demo.py --video-pair rec.mp4 --no-detect --video-out restored
python demo.py --video-pair rec.mp4 --video-out restored panel
python demo.py --video-pair rec.mp4 --clip data/live/train_light_1.mp4
python demo.py --video-pair rec.mp4 --no-review --save-every 0
```

`--video-out`이 하나면 `result.mp4`, 여러 개면 한 번의 복원으로 `result_<kind>.mp4`를
각각 쓴다.

### 라이브 — `--live`

프로젝터가 쏘는 화면을 웹캠이 보고 있어야 한다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--live` | off | 리그 구동 |
| `--screen N` | `rig.screen` (`1`) | 프로젝터가 붙은 모니터. `0` = 주 모니터 |
| `--camera N` | `rig.camera` (`0`) | 웹캠 인덱스 |
| `--manual-calib` | off | 네 모서리를 자동 검출 대신 클릭 |
| `--debug-view` | off | 추가 창: 워프 전 카메라 피드 + 검출된 사각형 |

```bash
python demo.py --live
python demo.py --live --screen 2
python demo.py --live --save-every 30 --debug-view
```

---

## `evaluate.py` — 복원 전후 점수 비교

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input <dir>` | `data/SampleData/sample_eval` | 채점할 쌍 |
| `--gt <dir>` | `data/SampleData/sample_eval` | `surface/`와 `labels/`가 있는 곳 |
| `--output <dir>` | `output` | 리포트가 나오는 곳 |
| `--limit N` | `0` | 최대 N쌍만 채점. `0` = 전부 |
| `--iou F` | `0.5` | true positive 판정 IoU |

```bash
python evaluate.py
python evaluate.py --input data/SampleData/sample_test --gt data/SampleData/sample_test
python evaluate.py --iou 0.75 --limit 10
```

번들 split은 전부 `distorted/` 옆에 자기 `surface/`와 `labels/`를 갖고 있어서
`--input`과 `--gt`에 같은 경로를 준다.

검출기 여러 개를 한 리포트에서 비교하는 것은 플래그가 아니라 설정이다 —
`detector.backend`에 리스트를 넣으면 항목마다 한 행이 된다.

---

## `train.py` — 복원기 재학습

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--data-root <dir>` | `restoration.yaml`의 `train.data` | 학습 쌍 |
| `--gt <dir>` | `--data-root` 아래 | `surface/`가 `--data-root` 밖에 있을 때 |
| `--out <dir>` | `runs` | 체크포인트가 나오는 곳 |
| `--epochs N` | `30` | `train.epochs`에서 |
| `--batch-size N` | `4` | `train.batch_size`에서 |
| `--lr F` | `0.0002` | `train.lr`에서 |
| `--sample N` | `0` | 삼중쌍 개수 상한 — 시드 고정 무작위 추출. `0` = 전부 |
| `--resume <ckpt>` | — | 체크포인트에서 이어서 |
| `--no-amp` | off | 혼합 정밀도 끄기 |
| `--device cuda\|cpu` | 있으면 cuda | — |

```bash
python train.py
python train.py --epochs 60 --batch-size 8
python train.py --sample 200 --epochs 2          # 빠른 스모크 실행
python train.py --data-root data/Create_Data/warp_0813
python train.py --resume runs/<tag>/best.pt
```

아키텍처는 플래그가 **아니다**. 어블레이션은 `restoration.yaml`의 `ablation:` 블록이다.
무엇을 껐는지가 실행 디렉터리와 체크포인트 이름에 태그로 붙으므로, 변형은 경로만 봐도
식별된다.

---

## `Data.py` — 데이터셋 수집

`data/` 아래 독립 스크립트로 분기하는 단일 진입점. 각 스테이지가 자기 `--help`를
갖고 있고, 스크립트를 직접 실행해도 동일하다.

```bash
python Data.py                  # 스테이지 목록
python Data.py capture --help   # 그 스테이지의 옵션
python data/warp.py --review    # `python Data.py warp --review`와 동일
```

| 스테이지 | 스크립트 | 리그 필요 |
|---|---|:---:|
| `check` | `data/check.py` | **필요** |
| `make_light` | `data/make_light.py` | — |
| `capture` | `data/capture.py` | **필요** |
| `capture_warp` | `data/capture.py --warp` | **필요** |
| `warp` | `data/warp.py` | — |
| `record` | `data/record.py` | **필요** |

### `check` — 모니터 + 웹캠

| 옵션 | 기본값 |
|---|---|
| `--camera N` | `capture.camera` (`0`) |

```bash
python Data.py check
python Data.py check --camera 1
```

### `make_light` — 영상 → light 프레임

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--src <path>...` | `train_light_*.mp4` 3개 | 영상 파일 또는 그것이 든 폴더 |
| `--out <dir>` | `data/Create_Data/projected` | — |
| `--step N` | `30` | N프레임마다 하나. 30 = 30fps에서 초당 1장 |
| `--augment <mode>` | `full` | `full`(6×) · `invert`(2×) · `none`(1×) |
| `--limit N` | `0` | 영상당 최대 N개 *원본* 프레임, 증강 전 기준 |

```bash
python Data.py make_light
python Data.py make_light --src /path/clip.mp4 --step 15
python Data.py make_light --augment none --limit 100
```

`test_light.mp4`는 일부러 뺐다. 여기서 수집한 세트가 held-out split과 투사 프레임을
하나도 공유하지 않게 하기 위해서다.

### `capture` — 프로젝터 + 웹캠 → raw 캡처

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--raw <dir>` | `data/Create_Data/raw_<MMDD>` | 특정 캡처 폴더에 이어 담기 |
| `--screen N` | `capture.screen` (`1`) | 프로젝터가 붙은 모니터 |
| `--camera N` | `capture.camera` (`0`) | — |
| `--rounds N` | `10` | 장면 구성 횟수. 라운드마다 surface 촬영부터 |
| `--limit N` | `50` | 라운드당 뽑는 light 프레임 수. `0` = 전부 |
| `--settle-ms N` | `1200` | light 프레임을 띄우고 캡처를 채택하기까지 대기 |
| `--warp` | off | 촬영하면서 정류도 함 (이게 `capture_warp`) |

```bash
python Data.py capture --screen 2
python Data.py capture_warp --screen 2
python Data.py capture --rounds 20 --limit 25      # 20장면에 500쌍
python Data.py capture --rounds 4 --limit 250      # 4장면에 1,000쌍
```

기본값은 10장면 × 50 = **500쌍**. `--limit 0`은 풀에 있는 프레임 전부를 뜻하는데,
기본 `make_light` 뒤라면 수천 장을 한 장면에 쏟는 것이다 — 복원에는 틀린 선택이다.
모델에 필요한 건 한 장면 위의 많은 light가 아니라 많은 surface다.

### `warp` — raw 캡처 → 정합된 쌍

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--raw <dir>` | 가장 최근 `raw_<MMDD>` | 정류할 캡처 폴더 |
| `--out <dir>` | 그 raw에 대응하는 `warp_<MMDD>` | — |
| `--mode <m>` | `boundary` | `boundary` · `homography` · `tps`(OpenCV < 5 필요) |
| `--limit N` | `0` | 최대 N장면만 처리 |
| `--review` | off | 장면마다 화면으로 확인하고 잘못된 것을 고침 |
| `--manual` | off | 모든 장면에서 모서리를 직접 클릭 |
| `--redetect` | off | `capture.py`가 기록한 모서리를 무시하고 다시 검출 |

```bash
python Data.py warp
python Data.py warp --review
python Data.py warp --mode homography
python Data.py warp --raw data/Create_Data/raw_0813
```

이 스테이지는 선택이 아니다. 카메라 속 화면은 사다리꼴이라 이전 단계까지로는 학습이
불가능하고, `raw/`만 있는 세션은 로더가 인식하는 레이아웃 자체가 아니다.

### `record` — 모델 없이 투사 + 녹화

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--clip <mp4>` | `data/live/test_light.mp4` | 투사할 클립 |
| `--screen N` | `record.screen` (`1`) | 프로젝터가 붙은 모니터 |
| `--camera N` | `record.camera` (`0`) | — |
| `--seconds F` | `0` | N초 후 종료. `0` = 제한 없음 |
| `--loop` | off | 클립 끝에서 멈추지 않고 다시 재생 |
| `--warp` | off | 한 번 캘리브레이션하고 정류된 화면을 녹화 |
| `--out <path>` | `data/recordings/rec_<MMDDHHMMSS>.mp4` | — |

```bash
python Data.py record --screen 2 --seconds 30
python Data.py record --clip data/live/train_light_1.mp4 --loop
python Data.py record --warp
```

창이 뜨는 즉시 녹화가 시작된다 — 키 입력 없음. 녹화본을 `demo.py --video-pair`로
보낼 거라면 `--warp`는 끄고 찍어라. 거기서 알아서 정류한다.

---

## 대화형 프롬프트 단축키

실행을 멈추고 입력을 기다리는 모든 창과 받는 키. `enter`와 `space`는 둘 다 적힌
곳에서 `a`와 서로 바꿔 쓸 수 있다.

| 어디서 | 키 | 하는 일 |
|---|---|---|
| `capture` 프레이밍 미리보기 | `s` · `enter` · `space` | 이 장면을 라운드의 `surface`로 채택 |
| | `c` | 프레임을 정지시키고 네 모서리를 클릭 |
| | `r` | 클릭한 뒤 자동 검출로 복귀 |
| | `q` · `esc` | 세션 중단 |
| `warp --review` | `enter` · `space` · `a` | 이 장면 수락 |
| | `A` | 이 장면과 이후 전부 수락 |
| | `m` | 네 모서리를 직접 클릭 |
| | `r` | 검출기 재실행 |
| | `s` | 이 장면 건너뛰기 |
| | `q` | 중단. 이미 쓴 장면은 유지 |
| `demo.py --live` 캘리브레이션 | `enter` · `space` · `a` | 워프 수락하고 시작 |
| | `m` | 네 모서리 클릭 |
| | `r` | 캘리브레이션 재실행 |
| | `q` | 실행 중단 |
| `demo.py --video-pair` 캘리브레이션 | `enter` · `a` | 워프 수락하고 시작 |
| | `m` | 네 모서리 클릭 |
| | `q` | 실행 중단 |
| 모서리 클릭 (공통) | 좌클릭 ×4 | 순서 상관없이 네 점 지정 |
| | `u` | 마지막 모서리 취소 |
| | `q` | 중단 |
| `--live` / `record` 실행 중 | `q` | 조기 종료 |

---

## 테스트

| 명령어 | 실행 범위 |
|---|---|
| `python -m pytest -q` | 전체 151개 |
| `python -m pytest tests/test_replay.py -q` | 재생 파이프라인만 |
| `python -m pytest -q -k "align or clip"` | 표현식에 맞는 것만 |
| `python -m pytest -q --collect-only` | 실행 없이 목록만 |

전부 하드웨어가 필요 없다. 체크포인트나 선택 모듈이 있어야 하는 테스트는 그것이
없으면 실패가 아니라 깔끔하게 skip된다.

| 파일 | 다루는 것 |
|---|---|
| `tests/test_pipeline.py` | 파일명 id, 샘플 탐색, 지표, `RunRecorder`, argparse 기본값, 라이브 워커 계약 |
| `tests/test_replay.py` | 시간축 짝짓기, 사이드카, 화면 검출, 다중 영상 출력, 창 크기 |
| `tests/test_restoration.py` | 설정 태그, 모든 어블레이션 토글, 체크포인트 왕복, 기본 가중치 |
| `tests/test_detection.py` | 레지스트리, 라벨 정규화, 크기 게이트, 두 백엔드 |
| `tests/test_collect.py` | 리그 없이 되는 `data/warp.py`·`data/make_light.py` |

---

## 출력물이 떨어지는 곳

| 경로 | 쓰는 주체 | git 추적 |
|---|---|---|
| `output/<timestamp>/` | `demo.py` | 안 함 |
| `output/Eval_<dataset>/` | `evaluate.py` | 안 함 |
| `runs/<tag>/` | `train.py` | 안 함 |
| `data/Create_Data/projected/` | `Data.py make_light` | 안 함 |
| `data/Create_Data/raw_<MMDD>/` | `Data.py capture` | 안 함 |
| `data/Create_Data/warp_<MMDD>/` | `Data.py warp` | 안 함 |
| `data/recordings/` | `Data.py record` | 안 함 |

`demo.py` 실행 폴더 안:

| 파일 | 생기는 조건 |
|---|---|
| `run_meta.json` | 항상 |
| `detections.csv` | 항상 |
| `captures/` · `frames_all/` | `--save-every` > 0 |
| `calib/` | 라이브·재생 실행 |
| `result.mp4` | `--video`, 라이브, 또는 `--video-out`이 하나인 재생 |
| `result_<kind>.mp4` | `--video-out`이 여러 개인 재생 |

실행마다 자기 타임스탬프 폴더가 생기므로 반복 실행이 서로 덮어쓰지 않는다.

---

## 플래그가 없는 설정

**어떤 플래그도 모델을 지정하지 않는다.** 아래는 전부
`projector_distortion/configs/*.yaml`에서 읽는다. 두 실행 사이에 바뀌는 값이 아니기
때문이다:

| 설정 | 위치 |
|---|---|
| 복원 백엔드 · 체크포인트 · 입력 크기 | `restoration.yaml`의 `model:` |
| 아키텍처와 어블레이션 | `restoration.yaml`의 `ablation:` |
| 에폭·배치·lr·누적·손실 가중치 | `restoration.yaml`의 `train:` |
| 검출 백엔드 · 체크포인트 · 신뢰도 | `detection.yaml`의 `detector:` / `weights:` |
| 클래스 이름, 박스 크기 게이트, 추론 크기 | `detection.yaml` |
| 라이브 리그: 모니터·웹캠·지연·캘리브레이션 확인 | `live.yaml`의 `rig:` |
| 수집 리그, 세션 폴더, 워프 기하, 코덱 | `collect.yaml` |

덮어쓸 수 있는 셋 — `--screen`, `--camera`, 그리고 재생 모드 전용 `--offset-ms` — 은
세션 첫 실행에서 잘못 짚기 쉬운 것들이다. 그때 실행 하나 날리는 대가가 파일 수정이
되어서는 안 되기 때문에 남겨뒀다.

```yaml
# projector_distortion/configs/detection.yaml
detector:
  backend: ssd      # ultralytics 불필요
  # backend: none   # 모든 실행에서 복원만
```

```yaml
# projector_distortion/configs/live.yaml
rig:
  screen: 1
  camera: 0
  offset: 6                  # 프로젝터 -> 카메라 지연, 프레임 단위
  review_calibration: true   # 무인 실행이면 false
```

---

[English](COMMANDS.md) · [← README](README.ko.md)

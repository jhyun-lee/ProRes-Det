# data/

바로 실행 가능한 토이 데이터셋. 아래 파일명 규약만 지키면 실데이터를 그대로 부어넣어도
코드 수정이 필요 없다 (개수·경로 하드코딩 없음).

`weights/` 와 달리 이 디렉토리는 git 에 포함되어 있어 clone 직후 바로 있다.

```
data/
├── collect.py              프로젝터+웹캠으로 직접 데이터셋 수집
├── sample_input/           demo.py / evaluate.py / train.py 입력
│   ├── pro/    projected_<oriId>_<beamId>.jpg   22장, 640×360
│   └── beam/   output_video_<beamId>.jpg        22장, 크기 혼재
├── sample_gt/              정답 (없어도 demo 는 동작, 점수만 안 나옴)
│   ├── clean/  Ori<oriId>.jpg      10장, 640×360    ← 학습 타겟 / PSNR·SSIM 기준
│   └── labels/ Ori<oriId>.txt      10장, 박스 107개  ← mAP 기준 (YOLO 포맷)
└── live/                   demo.py --live 입력
    ├── BeamVideo.mp4       프로젝터로 재생할 클립
    │                       5,858 프레임 @30fps = 3.3분, 854×480, 55 MiB
    └── BaseBackGround.jpg  캘리브레이션 중 띄울 배경, 1280×960
```

이미지 크기는 서로 맞출 필요 없다. 파이프라인이 전부 모델의 `input_size`(기본 640×360)로
리사이즈한다. `beam/` 에 854×480 과 1280×720 이 섞여 있어도 별도 처리가 없는 이유다.

## 파일명 규약

세 시점의 뷰를 파일명 안의 id 로 묶는다.

```
projected_0409001429_0404023332_294_75.jpg
          └───┬────┘ └──────┬───────┘
            oriId         beamId

  → sample_gt/clean/Ori0409001429.jpg                     (정답 화면)
  → sample_gt/labels/Ori0409001429.txt                    (검출 정답)
  → sample_input/beam/output_video_0404023332_294_75.jpg  (프로젝터가 쏜 프레임)
```

| 역할 | 파일명 | 모델에서 |
|---|---|---|
| `pro` | `projected_<oriId>_<beamId>.jpg` | 입력 ch 0:3 — 투사된 화면을 찍은 것 |
| `beam` | `output_video_<beamId>.jpg` | 입력 ch 3:6 — 프로젝터가 쏜 원본 프레임 |
| `clean` | `Ori<oriId>.jpg` | 학습 타겟 / PSNR·SSIM 기준 |
| `label` | `Ori<oriId>.txt` | 검출 mAP 기준 |

- `oriId` 에 `_` 금지 — 첫 `_` 까지를 oriId 로 자른다. `beamId` 는 `_` 포함 가능
  (위 예시의 beamId 는 `0404023332_294_75`).
- 하나의 `clean` 에 여러 `pro` 가 붙는 게 정상이다. 여기서는 clean 10장이 pro 22장을
  받치고 있고, 하나당 1~3장이다.
- `beam` 은 `pro` 와 1:1.
- 인식 확장자: `.jpg` `.jpeg` `.png` `.bmp`
- 짝이 없는 `pro` 는 경고 후 건너뛴다. 실행 실패가 아니다.
- `clean` 과 `label` 은 선택이다. `clean` 이 없으면 PSNR/SSIM 만 빠지고, `evaluate.py` 는
  둘 다 있는 샘플만 채점한다.

## 레이아웃

두 가지를 폴더 존재 여부로 자동 판별한다.

| 이름 | 구성 |
|---|---|
| `flat` (여기서 쓰는 것) | `pro/` `beam/` (+ `clean/`) |
| `research` | `ProjectorImage/` `BeamImage/` `OriginalImage/` |

```bash
python demo.py --input data/sample_input --gt data/sample_gt      # flat
python demo.py --input /path/to/WarpData_0520                     # research 자동 인식
```

한 폴더에 이미지가 섞여 있으면 `mixed` 로 처리하고, `projected_` / `output_video_`
접두사로 구분한다.

## 라벨 포맷

YOLO 표준 — 한 줄에 박스 하나, `<cls_id> <cx> <cy> <w> <h>`, 전부 0~1 정규화.

```
0 0.139406 0.263969 0.122381 0.207129
1 0.505193 0.162010 0.127800 0.211946
5 0.311010 0.202954 0.125993 0.199101
```

`cls_id` 는 `0..16` 이고
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) 의 `names`
순서를 따른다 — 과일 11종(Apple … Watermelon) 다음 동물 6종(Cat … Snake).
번들 라벨에는 17개 클래스가 모두 등장한다.

## 직접 수집하기 — `collect.py`

위 샘플셋도 이렇게 만들었다. 4단계이고, `check` 와 `capture` 는 리그가 필요하지만
`beam` 과 `warp` 는 파일 작업뿐이다.

```bash
python data/collect.py check                                     # 모니터 + 웹캠 확인
python data/collect.py beam    --src data/live/BeamVideo.mp4     # 영상 -> beam 프레임
python data/collect.py capture --screen 2 --rounds 3             # 투사하고 촬영
python data/collect.py warp                                      # 정면화해서 쌍 생성
```

세션 폴더 하나에 전 단계가 쌓이고, 그대로 나머지 코드가 읽는 구조다.

```
data/collected_<MMDD>/
├── beam/   output_video_<beamId>.jpg           [beam]     프로젝터가 쏘는 프레임
├── raw/    ori/Ori<oriId>.jpg                  [capture]  카메라 원본, 정면화 전
│           pro/projected_<oriId>_<beamId>.jpg  [capture]
├── clean/  Ori<oriId>.jpg                      [warp]     정면화, 640×360
├── pro/    projected_<oriId>_<beamId>.jpg      [warp]     정면화, 640×360
├── debug/  <oriId>_warp.jpg                    [warp]     워핑 전후 근거
└── collect_meta.json                           단계별 설정과 개수
```

```bash
python demo.py     --input data/collected_0803 --gt data/collected_0803
python train.py --data-root data/collected_0803
```

**라운드 동작.** `capture` 는 배경을 투사한 채 `s` 를 누를 때까지 기다리고, 그때 찍힌 게
`clean` 이다. 아무것도 투사되지 않은 장면이므로 물체를 먼저 배치하고 프레임 밖으로
빠진 뒤 누를 것. 그 시각이 `oriId` 가 되고 해당 라운드의 모든 촬영이 그 id 를 달기 때문에
`clean` 하나에 `pro` 여러 장이 붙는다. 이후 beam 프레임을 한 장씩 투사하며 한 장씩 찍는다.
`--rounds N` 은 장면을 바꿔가며 반복한다.

**`warp` 가 필요한 이유.** 카메라 안에서 스크린은 사다리꼴이고 물체 크기도 촬영마다
다르다. 이 단계 전까지는 학습에 못 쓴다. `warp` 는 정답 샷에서 스크린 경계를 찾아,
그 장면의 clean 과 그 장면의 모든 촬영본을 **동일한 매핑**으로 정면화한다. 그래야 쌍이
픽셀 격자를 공유하고 남는 차이가 투사된 빛뿐이 된다. 기본 `--warp boundary` 는
코너 4점 호모그래피 + 실측 경계 곡선 보정이다. 비스듬히 본 평면 스크린에 대해 정확하고,
경계가 휘어도 맞는다. `--warp tps` 는 기존 thin-plate spline 재현용인데
`opencv-python<5` 가 필요하다 (OpenCV 5 에서 shape 모듈이 빠졌다).

본 세션 전에 짧게 시험 촬영해서 `debug/<oriId>_warp.jpg` 를 먼저 확인할 것.

**타이밍.** `--settle-ms`(프레임 표시 후 대기)와 `--flush`(버릴 카메라 버퍼 프레임 수)가
촬영본과 원인이 된 프레임을 맞춰준다. `pro` 가 **직전** beam 프레임처럼 보이면 둘 다 올린다.

**라벨은 수집되지 않는다.** `evaluate.py` 로 mAP 를 재려면 `clean/` 을 직접 라벨링해서
`<세션>/labels/Ori<oriId>.txt` 로 넣어야 한다. 복원 학습과 PSNR/SSIM 은 라벨이 필요 없다.

옵션 전체 표: [README_running.ko.md](../README_running.ko.md)

## 실데이터로 교체

```bash
# 1) 같은 구조로 채우고 그대로 실행
python demo.py
python evaluate.py

# 2) 또는 원본 데이터셋을 직접 지정
python train.py --data-root /mnt/.../0_ImageData/1_WarpData_0520 --epochs 30
```

학습은 `clean` 타겟이 필수다. `train.py` 는 아래 순서로 찾아서 있는 것을 그대로 읽는다.
심볼릭 링크나 복사는 필요 없다.

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```


## 용량

`data/` 는 59 MiB 이고 거의 전부가 `data/live/BeamVideo.mp4`(55 MiB)다.
`--live` 를 안 쓰면 그 파일은 지워도 된다.

> `data/sample_input/clean` 은 다른 머신의 절대 경로를 가리키는 **깨진 심볼릭 링크**다.
> 현재 코드는 무시하지만 `os.walk('data')` 같은 단순 순회는 여기서 깨진다.
> 정리 권장: `git rm data/sample_input/clean`

---

[English](README_data.md) · [← README](../README.ko.md)

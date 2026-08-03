# data/

바로 실행 가능한 토이 데이터셋. 아래 **파일명 규약**만 지키면 실데이터를 그대로 부어넣어도
코드 수정이 필요 없다 (개수·경로 하드코딩 없음).

```
data/
├── sample_input/           demo.py / evaluate.py / train.py 입력
│   ├── pro/    projected_<oriId>_<beamId>.jpg   22장
│   └── beam/   output_video_<beamId>.jpg        22장
├── sample_gt/              정답 (없어도 demo 는 동작, 점수만 안 나옴)
│   ├── clean/  Ori<oriId>.jpg                   10장  ← 학습 타겟 / PSNR·SSIM 기준
│   └── labels/ Ori<oriId>.txt                   10장  ← 검출 mAP 기준 (YOLO 포맷)
└── live/                   demo.py --live 입력
    ├── BeamVideo.mp4       프로젝터로 재생할 클립 (5,858 프레임 @30fps = 3.3분, 854×480)
    └── BaseBackGround.jpg  캘리브레이션 중 띄울 배경 (1280×960)
```

## 파일명 규약

세 시점의 뷰를 **파일명 안의 id** 로 묶는다.

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

- `oriId` 에 `_` 를 넣지 말 것 (첫 `_` 까지를 oriId 로 자른다). `beamId` 는 `_` 포함 가능
- 하나의 `clean` 에 여러 `pro` 가 붙는 게 정상 (여기선 clean 10장 → pro 22장)
- `beam` 은 `pro` 와 1:1
- 확장자 `.jpg` / `.jpeg` / `.png` / `.bmp` 인식
- 짝이 없는 `pro` 는 경고 후 건너뛴다 (실행 실패 아님)

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

폴더 안에 이미지가 섞여 있으면 `mixed` 로 처리한다 (`projected_` / `output_video_`
접두사로 구분).

## 라벨 포맷

YOLO 표준 — 한 줄에 `<cls_id> <cx> <cy> <w> <h>`, 전부 0~1 정규화. `cls_id` 는 `0..16` 이고
[configs/detection.yaml](../projector_distortion/configs/detection.yaml) 의 `names` 순서와
일치한다 (과일 11종 + 동물 6종).

## 실데이터로 교체

```bash
# 1) 같은 구조로 채우고 그대로 실행
python demo.py
python evaluate.py

# 2) 또는 원본 데이터셋을 직접 지정
python train.py --data-root /mnt/.../0_ImageData/1_WarpData_0520 --epochs 30
```

학습은 `clean` 타겟이 필수다. `train.py` 는 아래 순서로 찾아서 **있는 것을 그대로 읽는다**.
심볼릭 링크나 복사는 필요 없다.

```
--data-root/OriginalImage/   →   --data-root/clean/   →   --gt/clean/
```

## 용량

`data/live/BeamVideo.mp4` 가 55 MB 로 이 폴더의 대부분이다. `--live` 를 안 쓰면 지워도 된다.

> `data/sample_input/clean` 은 다른 머신의 절대 경로를 가리키는 **깨진 심볼릭 링크**다.
> 현재 코드는 무시하지만 `os.walk('data')` 같은 순회는 여기서 깨진다.
> 정리 권장: `git rm data/sample_input/clean`

# data/

바로 실행 가능한 토이 데이터셋. 아래 파일명 규약만 지키면 실데이터를 그대로 부어넣어도
코드 수정이 필요 없다 (개수·경로 하드코딩 없음).

`weights/` 와 달리 이 디렉토리는 git 에 포함되어 있어 clone 직후 바로 있다.

```
data/
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

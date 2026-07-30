# weights/

| 파일 | 용도 | 구조 | params | 크기 |
|---|---|---|---|---|
| `restorer_restormerlike.pt` | 복원 | 3-level U-Net of RestormerLikeBlock | 4,184,259 | 16 MB |
| `detector_yolo11s.pt` | 검출 (`--detector yolo`) | YOLO11s | — | 18 MB |
| `detector_ssdlite.pth` | 검출 (`--detector ssd`) | SSDLite320-MobileNetV3-Large | 4.42 M | 17 MB |

셋 다 17-class 프로젝터 왜곡 데이터셋에 파인튜닝됨. 클래스 목록은
[../projector_distortion/configs/detection.yaml](../projector_distortion/configs/detection.yaml).

## 체크포인트 형식

`train.py` 산출물은 config 를 가중치와 함께 저장한다:

```python
{"format": 2, "arch": "restormer_like",
 "cfg": {...RestorationConfig...}, "state_dict": {...},
 "epoch": 30, "loss": 0.1234, ...}
```

추론 시 `load_checkpoint()` 가 `cfg` 를 읽어 **동일 구조를 자동 재구성**하므로
ablation 변형을 써도 플래그를 기억할 필요가 없다.

`restorer_restormerlike.pt` 는 cfg 가 없는 **raw state_dict** 라, 기본값(= 모든 토글 ON,
`FULL`)으로 재구성된다. 실제로 strict 로드가 통과하므로 이 파일은 FULL 변형이 맞다.

## 다른 가중치 쓰기

```bash
python demo.py --restorer-weights path/to/other.pt
python demo.py --detector ssd --det-weights path/to/other.pth
```

기본 경로는 `projector_distortion/configs/*.yaml` 의 `weights:` 항목이다.

## 배포 시

`.gitignore` 가 `weights/*.pt`, `*.pth` 를 제외한다. 저장소에 올릴 때는 릴리스 에셋이나
외부 스토리지에 두고 링크만 남기는 것을 권장한다 (합계 52 MB).

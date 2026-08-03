# weights/

체크포인트 3개가 git 에 포함되어 있다(합계 51 MiB). clone 직후 별도 다운로드 없이
바로 실행된다. 다른 가중치를 쓰려면 이 파일들을 덮어쓰지 말고 `--restorer-weights` /
`--det-weights` 로 경로를 지정한다.

## 파일

| 파일 | 용도 | 구조 | params | 크기 |
|---|---|---|---|---|
| `restorer_restormerlike.pt` | 복원 | 3-level U-Net of RestormerLikeBlock | 4,184,259 | 16.1 MiB |
| `detector_yolo11s.pt` | 검출 (`--detector yolo`) | YOLO11s | 9,434,371 | 18.3 MiB |
| `detector_ssdlite.pth` | 검출 (`--detector ssd`) | SSDLite320-MobileNetV3-Large | 4,393,592 | 17.0 MiB |

셋 다 동일한 17-class 프로젝터 왜곡 데이터셋(과일 11종 + 동물 6종)에 파인튜닝됨.
클래스 목록은 [configs/detection.yaml](../projector_distortion/configs/detection.yaml).
YOLO 체크포인트는 같은 17개 이름을 자체적으로 들고 있고, `--classes` 를 주지 않으면
그쪽이 우선한다.

기본 경로는 config 에서 온다. 소스에 하드코딩된 경로는 없다.

```yaml
# projector_distortion/configs/restoration.yaml
model:
  weights: weights/restorer_restormerlike.pt

# projector_distortion/configs/detection.yaml
weights:
  yolo: weights/detector_yolo11s.pt
  ssd:  weights/detector_ssdlite.pth
```

경로는 작업 디렉토리가 아니라 프로젝트 루트 기준으로 해석된다.

## 다른 가중치 쓰기

```bash
python demo.py --restorer-weights path/to/other.pt
python demo.py --detector ssd --det-weights path/to/other.pth
```

## 체크포인트 형식

`train.py` 산출물은 구조 설정을 가중치와 함께 저장한다.

```python
{"format": 2, "arch": "restormer_like",
 "cfg": {...RestorationConfig...}, "state_dict": {...},
 "epoch": 30, "loss": 0.1234, ...}
```

추론 시 `load_checkpoint()` 가 `cfg` 를 읽어 동일 구조를 자동 재구성하므로,
ablation 변형을 써도 플래그를 다시 줄 필요가 없다.

```bash
python train.py --no-ca --epochs 30
python demo.py --restorer-weights runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt
#                                  ↑ --no-ca 불필요
```

`restorer_restormerlike.pt` 는 cfg 가 없는 raw state_dict 라 기본값(모든 토글 ON,
태그 `FULL`)으로 재구성된다. strict 로드가 통과하므로 이 파일이 FULL 변형인 게 맞다.
실행 로그에는 `cfg from legacy-raw` 로 표시된다.

## residual 규약

복원 네트워크는 복원 이미지를 직접 그리지 않는다. `pro` 에서 뺄 빛만 내놓는다.

```
입력  (B, 6, H, W) = cat([pro, beam])  in [-1, 1]
출력  (B, 3, H, W) = residual
restored = (pro - residual).clamp(-1, 1)
```

### 왜 clean 이 아니라 residual 인가

1) 원본 보존이 기본값이 된다.
투사광이 닿지 않은 영역은 residual ≈ 0 이면 되고, 그러면 `pro` 픽셀이 그대로 통과한다.
"아무것도 하지 않음"이 항등 함수라, 네트워크는 바뀌어야 하는 곳만 학습하면 된다.
clean 을 직접 회귀하면 멀쩡한 배경까지 전부 재생성해야 해서 손대지 말아야 할 곳이 뭉개진다.

2) 오브젝트를 그려 넣는 과적합을 막는다.
clean 을 직접 출력하게 두면, 입력을 거의 무시하고 학습셋에서 본 화면을 통째로 외워
그려내는 지름길이 손실을 가장 빠르게 낮춘다. 번들 데이터는 clean 10장에 pro 22장이
붙어 clean 이 반복 등장하므로, `oriId` 별 정답 화면을 암기하는 게 특히 유리해진다.
그렇게 학습된 모델은 검출기가 원래 없는 물체를 보게 만들어 평가 자체를 무의미하게 만든다.
`restored = pro − residual` 구조는 출력이 항상 실제 카메라 픽셀에서 파생되도록 강제해서
그 경로를 막는다.

3) 값이 폭주하지 않는다.
출력단 `tanh` 로 residual ∈ [-1, 1], 뺄셈 뒤 `clamp(-1, 1)`. 두 겹으로 범위가 잡힌다.
(`--no-tanh` 로 첫 겹을 없앨 수 있고, 그게 ablation 항목 중 하나다.)

### 어떻게 강제하나 — 손실은 residual 이 아니라 restored 에 건다

핵심은 뺄셈이 그래프 안에 있다는 것이다. residual 에 직접 정답을 주지 않고,
뺀 결과만 clean 과 비교한다. 그래서 "무엇을 빼야 하는가"는 네트워크가 스스로 찾는다.

```python
residual = net(torch.cat([pro, beam], dim=1))     # 네트워크 출력
restored = (pro - residual).clamp(-1, 1)          # 뺄셈이 그래프 안
loss = (0.93 * L1(restored, clean)
      + 2.04 * Perceptual(restored, clean)
      + 0.53 * (1 - SSIM(restored, clean))
      + 0.90 * WaveletHF(restored, clean))        # 4항 전부 restored 기준
```

| 손실 항 | 무엇을 재나 | 무엇을 벌하나 |
|---|---|---|
| `L1` | 픽셀 절대 오차 | 전역 색·밝기 어긋남 |
| `Perceptual` (VGG19 relu3_3) | 특징맵 거리 | 픽셀은 비슷한데 구조가 깨진 결과 |
| `1 − SSIM` | 국소 휘도·대비·구조 상관 | 평균만 맞춘 밋밋한 출력 |
| `WaveletHF` (Haar LH/HL/HH, LL 제외) | 엣지·텍스처만 | 흐릿하게 뭉개서 손실 낮추기 |

`WaveletHF` 가 저주파(LL) 를 빼고 보는 게 요점이다. 전체를 블러 처리해도 L1 은 잘 내려가지만
고주파 항은 안 내려간다. 이게 residual 이 실제 투사광 패턴의 경계를 따라가게 만든다.

가중치는 [configs/restoration.yaml](../projector_distortion/configs/restoration.yaml)
의 `train.loss` 에 있고, 이 데이터셋에 대한 Optuna 스윕 결과다.
구현은 [train.py](../train.py) 참고.

> clean 을 직접 뱉는 복원기도 `BaseRestorer` 인터페이스는 만족한다. 다만 그 경우
> `residual` 시각화와 `residual_mean` 지표가 의미를 잃고, 위 두 가지 이점도 사라진다.
> [모듈 교체](../README.ko.md#5-모듈-교체) 참고.

---

[English](README_weights.md) · [← README](../README.ko.md)

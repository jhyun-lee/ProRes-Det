# weights/

체크포인트 3개, 체크포인트 포맷, 그리고 복원 네트워크가 surface 이미지가 아니라 residual을
예측하는 이유.

- [파일](#파일)
- [경로가 오는 곳](#경로가-오는-곳)
- [체크포인트 포맷](#체크포인트-포맷)
- [residual 규약](#residual-규약)

---

## 파일

세 개 모두 git에 커밋되어 있다 — 합계 51 MiB. 클론 직후 별도 다운로드 없이 실행된다. 다른
가중치를 쓰려면 이 파일들을 덮어쓰지 말고 아래 config가 그쪽을 가리키게 한다.

| 파일 | 용도 | 구조 | 파라미터 | 크기 |
|---|---|---|---|---|
| `restorer_nafse_unet.pt` | 복원 | NAFSEBlock 3-level U-Net | 4,184,259 | 16.1 MiB |
| `detector_yolo11s.pt` | `yolo` 검출기 | YOLO11s | 9,434,371 | 18.3 MiB |
| `detector_ssdlite.pth` | `ssd` 검출기 | SSDLite320-MobileNetV3-Large | 4,393,592 | 17.0 MiB |

세 개 모두 동일한 17클래스 프로젝터 왜곡 데이터셋으로 파인튜닝됐다 — 과일 11개, 동물 6개.
클래스 목록은 [configs/detection.yaml](../projector_distortion/configs/detection.yaml)에 있다.
YOLO 체크포인트는 같은 17개 이름을 내부에 갖고 있고, config의 목록보다 그쪽이 우선한다.
torchvision에는 이름이 없으므로 `ssd`는 항상 config의 목록을 쓴다.

---

## 경로가 오는 곳

config에서만. 소스에 하드코딩된 경로도, 이 경로들을 덮는 CLI 플래그도 없다. 체크포인트
교체는 한 줄 수정이고 모든 엔트리포인트가 한 번에 따라간다.

```yaml
# projector_distortion/configs/restoration.yaml
model:
  backend: naf_se_unet
  weights: weights/restorer_nafse_unet.pt

# projector_distortion/configs/detection.yaml
weights:
  yolo: weights/detector_yolo11s.pt
  ssd:  weights/detector_ssdlite.pth
```

이 경로들은 작업 디렉터리가 아니라 프로젝트 루트 기준으로 해석된다.

---

## 체크포인트 포맷

`train.py`가 만드는 것은 가중치 옆에 아키텍처 config를 함께 저장한다:

```python
{"format": 2, "arch": "naf_se_unet",
 "cfg": {...RestorationConfig...}, "state_dict": {...},
 "epoch": 30, "loss": 0.1234, ...}
```

`load_checkpoint()`가 `cfg`를 읽어 맞는 아키텍처를 재구성한다. 그래서 ablation 변형도 추론
시점에 설정을 다시 줄 필요가 없다.

```bash
# projector_distortion/configs/restoration.yaml 의 ablation: { use_ca: false }
python train.py --epochs 30
# 그다음 model.weights를 runs/0730_1948_30ep_NoCA/restorer_NoCA_best.pt 로
python demo.py
#   ablation 블록을 그대로 둘 필요는 없다. config가 체크포인트에 같이 실려 있다
```

실행 시 config 출처가 출력된다:

| `cfg from` | 의미 |
|---|---|
| `embedded` | 체크포인트가 자기 `cfg`를 갖고 있음. `train.py` 산출물의 정상 경우 |
| `legacy-raw` | 순수 `state_dict`. 기본값으로 재구성 |
| `defaults` | format 2인데 `cfg` 키가 없음 |

`restorer_nafse_unet.pt`는 `legacy-raw`다. config가 없는 순수 state_dict이므로 기본값으로
재구성된다 — 모든 토글 ON, tag `FULL`. strict 로드가 성공하는 것이 이 파일이 실제로 FULL
변형임을 확인해 준다.

---

## residual 규약

복원 네트워크는 복원된 이미지를 그리지 않는다. `distorted`에서 **빼낼 빛**을 내놓는다:

```
input   (B, 6, H, W) = cat([distorted, light])  in [-1, 1]
output  (B, 3, H, W) = residual
restored = (distorted - residual).clamp(-1, 1)
```

### 왜 clean이 아니라 residual인가

**1) 원본 보존이 기본 동작이 된다.**
투사광이 닿지 않은 곳은 residual ≈ 0으로 충분하고, `distorted` 픽셀이 그대로 통과한다. "아무것도
하지 않음"이 항등 함수이므로, 네트워크는 바뀌어야 하는 것만 학습하면 된다. clean을 직접
회귀하면 아무 문제 없던 배경까지 다시 생성해야 하고, 건드리지 않아도 됐던 영역이 뭉개진다.

**2) "객체를 그려 넣는" 과적합을 막는다.**
네트워크가 clean을 직접 출력하면, 손실을 가장 빨리 낮추는 방법은 입력을 대체로 무시하고 학습
세트에서 외운 화면을 재현하는 것이다. 번들 데이터는 특히 그 유혹이 크다 — surface 10장이 distorted
22장을 받치니 clean이 반복되고 `surfaceId`별로 타깃을 외우는 것이 이득이 된다. 그렇게 학습된 모델은
존재하지 않았던 객체를 검출기에 보여주고, 평가의 의미 자체가 무너진다.
`restored = distorted − residual`은 출력이 항상 실제 카메라 픽셀에서 파생되도록 강제한다.

**3) 값이 발산할 수 없다.**
출력 `tanh`가 residual을 [-1, 1]로 묶고, 뺄셈 후의 `clamp(-1, 1)`이 결과를 다시 묶는다 —
범위 통제 2중. `use_tanh: false`가 첫 번째를 제거하고, 그것이 ablation 스위치 중 하나다.

### 어떻게 강제되나 — 손실은 residual이 아니라 `restored`에 걸린다

핵심은 뺄셈이 그래프 안에 있다는 점이다. residual에는 자기 타깃이 주어지지 않고, 뺀 결과만
clean과 비교된다. 무엇을 뺄지는 네트워크가 찾아낼 몫으로 남는다.

```python
residual = net(torch.cat([distorted, light], dim=1))     # 네트워크 출력
restored = (distorted - residual).clamp(-1, 1)          # 그래프 안의 뺄셈
loss = (0.93 * L1(restored, surface)
      + 2.04 * Perceptual(restored, surface)
      + 0.53 * (1 - SSIM(restored, surface))
      + 0.90 * WaveletHF(restored, surface))        # 네 항 모두 `restored`를 측정
```

| 손실 항 | 측정 대상 | 벌하는 것 |
|---|---|---|
| `L1` | 절대 픽셀 오차 | 전역 색·밝기 드리프트 |
| `Perceptual` (VGG19 relu3_3) | 특징맵 거리 | 픽셀은 가깝지만 구조가 깨진 결과 |
| `1 − SSIM` | 국소 휘도·대비·구조 | 평균만 맞춘 평탄한 출력 |
| `WaveletHF` (Haar LH/HL/HH, LL 제외) | 에지와 텍스처만 | 손실을 낮추려 전체를 흐리는 것 |

저주파 LL 밴드를 버리는 것이 `WaveletHF`의 요점이다. 이미지 전체를 흐리면 L1은 낮아지지만
고주파 항은 낮아지지 않는다. 그것이 residual을 투사광의 실제 경계에 붙게 만든다.

가중치는 [configs/restoration.yaml](../projector_distortion/configs/restoration.yaml)의
`train.loss`에 있고 이 데이터셋에 대한 Optuna 스윕 결과다. 구현: [train.py](../train.py).

> clean을 직접 내놓는 커스텀 복원기도 `BaseRestorer` 인터페이스는 만족한다. 다만 그 경우
> `residual` 시각화와 `residual_mean` 지표는 의미를 잃고, 위 장점 두 개도 사라진다.
> [모듈 교체](../README.ko.md#5-모듈-교체) 참고.

---

[English](README_weights.md) · [← README](../README.ko.md)

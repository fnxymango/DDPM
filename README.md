# DDPM

Denoising Diffusion Probabilistic Models([Ho et al., 2020](https://arxiv.org/abs/2006.11239))를 직접 구현해 MNIST에서 동작을 확인하고, 같은 코드를 의료 영상(PneumoniaMNIST)에 적용한 학습 프로젝트입니다.

학습 계획, 단계별 완료 기준, 진행 현황은 [GUIDELINE.md](GUIDELINE.md)에 있습니다.

## 결과 요약

| 데이터 | 학습 | 최종 loss | 샘플 |
|---|---|---|---|
| MNIST (60,000장) | 20k스텝, 배치 128 | 0.0218 | [samples_step020000.png](runs/mnist/samples_step020000.png) |
| PneumoniaMNIST (4,708장) | 20k스텝, 배치 128 | 0.0216 | [compare_real_vs_gen_5.png](runs/pneumonia/eval_ema_step020000/compare_real_vs_gen_5.png) |

- MNIST: 5k스텝부터 숫자 형태의 샘플이 나옵니다.
- PneumoniaMNIST: 흉부 X선 구도와 밝기 분포는 실제와 비슷하게 재현했지만, 세부 질감이 부족합니다. 학습을 오래 할수록 학습 영상에 가까운 샘플이 늘었습니다(0.3% → 2.7%).
- 평가 방법, 수치, 해석, 한계: [docs/stage6_evaluation.md](docs/stage6_evaluation.md)

## 구조

```
ddpm/
  diffusion.py      노이즈 스케줄, q_sample(식 4), L_simple(식 14), 샘플링(Algorithm 2)
  unet.py           28×28용 작은 U-Net (28→14→7, 시간 임베딩, 7×7 self-attention)
  data.py           MNIST / PneumoniaMNIST 로딩, [-1, 1] 스케일링
scripts/
  forward_vis.py        t별 forward 결과 시각화 → runs/forward/
  train.py              학습 + 중간 샘플·체크포인트 저장
  eval_pneumonia.py     생성 영상 평가 (최근접 학습 영상, 밝기 분포, 분류기)
  compare_pneumonia.py  실제 vs 생성 5장 비교 그림
  run_all.sh            forward → MNIST → Pneumonia → 평가 전체 실행
docs/
  notes_stage1-3.md     1~3단계(직관·수식·점검)용 질문 목록
  stage6_evaluation.md  PneumoniaMNIST 평가 정리
runs/                   실행 결과 (그림, loss 기록, 평가 수치)
```

## 설정

논문의 기본 설정을 따랐습니다.

- T = 1000, β_t는 1e-4에서 0.02까지 선형 (4절)
- 손실은 L_simple: ‖ε − ε_θ(x_t, t)‖² (식 14)
- 샘플링 분산은 σ_t² = β_t (3.2절)
- Adam lr 2e-4, EMA 0.9995, gradient clipping 1.0
- U-Net 11.25M 파라미터, 데이터 증강 없음

## 실행

```bash
uv venv .venv -p 3.12
uv pip install -p .venv/bin/python torch torchvision medmnist matplotlib

# 전체 실행 (데이터는 data/에 자동으로 받음)
scripts/run_all.sh

# 개별 실행
.venv/bin/python scripts/forward_vis.py
.venv/bin/python scripts/train.py --dataset mnist --steps 20000 --out runs/mnist
.venv/bin/python scripts/train.py --dataset pneumonia --steps 20000 --out runs/pneumonia
.venv/bin/python scripts/eval_pneumonia.py runs/pneumonia runs/pneumonia/ema_step020000.pt
```

- CUDA GPU가 필요합니다. 학습은 `torch.compile`과 bf16을 씁니다(첫 컴파일 약 5분).
- RTX A5000에서 데이터셋당 약 45분 걸렸습니다.
- 체크포인트(.pt), 데이터, 논문 PDF, 로그는 저장소에 포함하지 않습니다.

## 참고

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Yang, J. et al. (2023). MedMNIST v2. Scientific Data. [medmnist.com](https://medmnist.com/)

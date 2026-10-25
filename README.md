<div align="center">

# DecisionVLA

**Typed decisions · Parallel actions · Visible uncertainty**

관측 하나에서 행동과 확률을 함께 읽는 Vision–Language–Action 실험.

[English](README.en.md) · [아키텍처](docs/architecture.md) · [실험 결과](docs/experiments.md) · [대시보드](docs/dashboard.md)

![Status: Experimental](https://img.shields.io/badge/status-experimental-CE9147)
![Backbone: DiffusionGemma](https://img.shields.io/badge/backbone-DiffusionGemma-43789D)
[![License: MIT](https://img.shields.io/badge/code-MIT-398779)](LICENSE)

</div>

![DecisionVLA의 제안 아키텍처. 공유 Transformer, native denoising과 entropy-bound sampler, 마지막 행동·확률 readout.](figures/decisionvla-native/architecture.png)

<p align="center"><a href="https://www.figma.com/design/FbL9C7OnlyOZkiyGPOkcW0/?node-id=13407-92873">Editable Figma</a> · <a href="figures/decisionvla-native/">Figure source</a></p>

DecisionVLA는 DiffusionGemma의 내장 vision encoder와 공유 Transformer를 로봇 제어에 적응시키는 VLA 프로젝트입니다. 이미지와 지시문을 받아 **8 × 7 행동 chunk**, 행동별 확률분포와 불확실성을 반환하는 것이 목표입니다. 학습, 추론 최적화, 실패 분석 과정을 코드와 측정 기록으로 공유합니다.

> **개발 상태 · 2026-10-25**
> 전체 40-task 학습을 **46-step checkpoint 저장 후 중단**했습니다. 별도 native denoising·sampler·self-conditioning 경로와 typed decision API를 구현했고, 작은 실제 모델에서 CPU 수치·재개 검사를 통과했습니다. **26B GPU 검증·새 학습·성공률 평가는 아직 진행 전**입니다. 아래 실측은 기존 action-only pilot 결과입니다. [Native 구현·비교 명령](docs/native-reference.md) · [중단 검산](reports/openjev-redesign-20261025/stop-audit.json)

## 핵심 방향

| | DecisionVLA |
|---|---|
| **하나의 backbone** | 관측 인코딩과 행동 denoising에 같은 text Transformer·LoRA 사용 |
| **병렬 행동 생성** | 한 관측의 KV cache를 재사용해 여러 행동 위치를 함께 복원 |
| **확률을 함께 출력** | 최종 행동 분포, entropy, 선택 확률과 제어값 분산 확인 |
| **GPU 한 장의 추론** | RTX 5090에서 expert NVFP4 W4A16·fused kernels·CUDA Graph 검증 |
| **재현 가능한 실험** | 고정 데이터·설정·checkpoint hash, 원시 결과와 독립 검산 연결 |

새 경로는 중간 denoising에서 **전체 vocabulary를 유지**하고, 마지막 readout에서만 허용된 행동으로 변환합니다. Native entropy-bound sampler와 self-conditioning을 보존하며, 같은 native checkpoint의 1·2·4-pass 속도·행동 차이·LIBERO 성공률을 비교하는 도구를 제공합니다. [구체적인 데이터 흐름과 참고 구현](docs/architecture.md).

## 행동을 typed decision으로

공유 관측과 **질문·허용 답**을 입력하고, 텍스트 파싱 없이 모델 확률에서 `predicate / choice / score` 답을 읽습니다. [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)의 인터페이스 철학을 로봇 제어에 적용하며, 자체 DiffusionGemma backend를 사용합니다.

로봇 profile은 **48개 제어 bin 선택 + 8개 gripper predicate**입니다. 질문이 실제 모델 입력에 포함되고, 출력 슬롯·학습 target·실행 제어가 같은 schema로 연결됩니다. 임의의 의미 질문은 실험용으로만 허용하며, 진행률이나 성공 여부를 이미 학습했다고 주장하지 않습니다. [API·예제·정확한 의미](docs/typed-decisions.md).

## 지금까지 측정한 것

**RTX 5090 · 기존 4,500-step Spatial pilot · 4회 denoising · 8개 행동 + confidence**

| 측정 | 결과 |
|---|---:|
| W4A16 + fused + CUDA Graph, p50 / p95 | **188.43 / 190.35 ms** |
| PyTorch 최대 할당 메모리 | **18.815 GiB** |
| 20회 대응 LIBERO 평가, BF16 / W4A16 | **12 / 11 성공** |
| 별도 2k pilot, Spatial 500회 | **14 / 500 성공** |

[5090 측정·행동 차이·신뢰구간](reports/inference-nvfp4-5090-20261025/final/) · [2k 실패 분석](reports/libero-spatial-confidence/)

188ms는 입력 전처리부터 행동·confidence의 CPU 반환까지의 warm 측정이며, 네트워크·로봇 구동·최초 컴파일 시간은 포함하지 않습니다. 18.815GiB는 전체 장치 메모리가 아닌 PyTorch peak allocation입니다. 20회 비교의 W4A16−BF16 성공률 차이는 −5%p, 95% CI [−35, +25]로 동등성을 입증하지 못했습니다. **이 수치는 새 설계나 최종 40-task 모델의 결과가 아닙니다.**

## 추론을 들여다보기

로컬 대시보드는 실제 관측과 **56개 행동 분포**, 신뢰도 변화, 추론 지연을 보여줍니다. LIBERO 평가 진행률·suite별 성공/실패와 첫 신뢰도 대 실제 결과를 함께 확인할 수 있습니다. `--dashboard-session`으로 평가 worker에 연결하거나 저장된 trace를 명시적인 REPLAY 모드로 재생합니다. [사용 방법](docs/dashboard.md).

```bash
python -m decisionvla.dashboard --session results/recorded-session --mode replay
```

**신뢰도 0.9는 작업 성공률 90%를 뜻하지 않습니다.** 현재 값은 조건부 행동 분포의 집중도를 요약합니다. 실패 예측력과 calibration은 별도로 검증하며, 확률 표시만으로 행동의 인과적 설명을 주장하지 않습니다. [확률·신뢰도 계약](docs/confidence.md).

## 코드 살펴보기

현재 BF16 pilot의 재현 환경은 Linux / Python 3.12 / CUDA 13입니다. 새 native 경로는 별도 [requirements-native.txt](requirements-native.txt) 환경을 사용합니다. 아래 pilot 환경에 덮어 설치하지 않습니다.

```bash
git clone https://github.com/taewan2002/DecisionVLA.git
cd DecisionVLA
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -e '.[dev]'
pytest -q
ruff check src tests scripts benchmarks
```

Base model과 데이터는 각각의 upstream에서 받습니다. 현재 공개된 adapter release는 없습니다. 기존 pilot의 BF16 API는 50GiB 이상의 VRAM이 필요하며, 5090 최적화 경로는 별도 [NVFP4 재현 도구](benchmarks/NVFP4.md)를 사용합니다. [기존 학습 재현](docs/training.md) · [가중치 형식](docs/weights.md).

## 다음 단계

- [x] 행동·확률 API, 대시보드, LIBERO 평가·검산 구현
- [x] 기존 pilot의 RTX 5090 W4A16·fused·CUDA Graph 실측
- [x] 원본 LIBERO 40-task 데이터 정렬·split 검산
- [x] 학습 checkpoint 보존 및 DecisionVLA 새 설계 작성
- [x] 질문-conditioned native sampler·self-conditioning 및 typed decision API 구현, 작은 모델 CPU 수치·재개 검증
- [ ] 전체 26B 모델의 GPU 수치·실데이터 검증
- [ ] GPU0/1의 batch-64 학습 구성·재개·처리량 확인 후 새 학습
- [ ] 원본 LIBERO와 학습에 쓰지 않은 LIBERO Plus 평가
- [ ] 최종 5090 배포 검증과 adapter-only experimental release

새 학습 예산의 제안은 rank 32 · global batch 64 · 100k updates입니다. 기존 46-step 가중치를 이어 쓰지 않고 새 run으로 분리하며, 아직 시작하지 않았습니다. 성공률·수렴·완료 시간을 미리 보장하지 않습니다.

## 문서와 코드

| 경로 | 내용 |
|---|---|
| [Architecture](docs/architecture.md) / [Development](docs/development.md) | 새 설계, 변경 범위와 구현 검증 단계 |
| [Experiments](docs/experiments.md) | 완료·중단된 실험과 측정 범위 |
| [`src/decisionvla/`](src/decisionvla/) | 기존 policy, LoRA, 학습, 확률 API, 대시보드 |
| [`benchmarks/`](benchmarks/) | 5090 양자화·fused·Graph 변환 및 검산 |
| [`evaluation/`](evaluation/) / [`reports/`](reports/) | LIBERO·Plus 평가 도구와 공개 가능한 근거 |
| [Sources & licenses](docs/third-party.md) | DiffusionGemma, OpenJEV 계열 참고 구현과 데이터 출처 |

원본 코드는 **MIT**입니다. Base model·데이터·의존성의 라이선스는 별도이며 이 저장소에 가중치나 데이터셋을 포함하지 않습니다. 이 프로젝트는 OpenAI, Jev/TypeSafe 또는 Google의 공식 구현이 아닙니다.

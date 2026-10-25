# DecisionVLA 구현과 다음 단계

> **이전 구현 기록:** 2026-10-25에 40-task 학습을 46-step에서 저장·중단했습니다. 새 설계는 [DecisionVLA architecture](architecture.md), 현재 작업 범위는 [development plan](development.md)을 따릅니다. 아래의 진행 중·후속 자동 실행 설명은 당시 이력이며 재시작 지시가 아닙니다.

2026-10-24. 개인 오픈소스 프로젝트이며 기존 학위논문 코드·checkpoint·결과와 분리되어 있다.

## 현재 구현

공식 DiffusionGemma BF16 원본 위에 공유 attention LoRA와 action embedding delta를 학습한다. RGB 1장과 instruction을 입력하고, native Transformer가 8×7개의 action bins를 병렬 복원한다. 별도 action expert, world model 또는 후보 선택 모델은 없다.

첫 실행은 GPU 1장으로 고정했다. LIBERO-Spatial 10개 task의 450개 시연으로 학습하고 50개 시연을 검증에 쓴다. Rank 16, effective batch 8, 2,000 updates와 12시간 상한이다. 수치와 데이터 계약은 [training.md](training.md), 실행 설정은 [config](../configs/libero_spatial.json), 실제 진행은 [run 기록](../reports/libero-spatial-pilot/README.md)을 따른다.

이번 pilot은 proprioception과 self-conditioning을 제외했다. 전자는 관측 추가 비교, 후자는 정확한 soft embedding과 sampler를 함께 구현할 때 비교할 후속 항목이다. Action은 고정 controller 범위 `[-1, 1]`을 사용하므로 train statistics fit을 하지 않는다.

## 평가

첫 rank-32 2k checkpoint의 LIBERO 500회 평가는 14회 성공(2.8%)으로 완료했다. Open-loop MAE나 loss를 성공률로 해석하지 않는다. 현재 10만 update 실험은 1만 간격으로 같은 500개 초기상태를 재사용하는 개발 비교이며, 미사용 test로 소개하지 않는다. 별도 확인 평가는 아직 시작하지 않았다.

같은 정책의 denoising steps 1/2/4/8을 비교한다. 학습과 simulator 구현이 안정된 후 성공 수, task별 결과, 실제 latency, episode 수를 공개한다. 작은 pilot의 차이를 통계적으로 확정된 우위로 쓰지 않는다.

## NVFP4와 5090

1. 공유 LoRA matrix를 각각 한 번만 merge하고 tied action embedding rows를 반영한다.
2. Train/calibration subset의 이미지·지시문·다양한 noise 수준의 action canvas로 ModelOpt calibration을 수행한다.
3. 지원되는 expert linear layers부터 NVFP4로 export하고 민감한 경로는 높은 정밀도로 유지한다.
4. 동일 입력/noise에서 BF16 reference와 가속 runtime의 step별 logits 및 action을 비교한다.
5. BF16/NVFP4의 대응 rollout을 비교하고 실제 RTX 5090에서 peak memory와 p50/p95를 측정한다.

기존 [NVFP4 base 측정](../reports/base-latency-2026-10-24/report.ko.md)은 1-view/4-step p50 144.68ms, p95 146.38ms, 관측 전체 GPU 메모리 최대 22.26GiB였다. Synthetic image와 미학습 text 출력의 비용 기준선이다. 현재 action sampler와 다른 경로이며, 최종 VLA 또는 5090의 속도로 쓰지 않는다.

5090 목표는 batch 1, 1 camera, 8 actions, p95 ≤200ms와 peak device memory ≤28GiB이다. 두 수치는 아직 목표다. 8-action chunk를 200ms에 생성하는 것은 40Hz feedback 제어와 다르다. Camera age, 재계획 간격과 controller queue를 함께 평가한다.

## 두 GPU로 확대할 조건

사용자 요청으로 작은 배치의 후속 학습을 2,534 update에서 중단하고, 두 GPU에서 fresh LoRA·global batch 64·100k 구성으로 전환한다. Microbatch 4 × gradient accumulation 8 × GPU 2장이며, 실제 DDP gradient·저장/재개 검증을 통과해야 본학습을 시작한다. [고정 계약](training-b64-100k.md)에 LR·1만 간격 500회 평가·checkpoint 선택 기준을 기록했다. GPU 수와 배치 증가가 성능 향상을 보장하지는 않는다. Native 256→64 canvas, action 범위 정규화, self-conditioning, camera 추가는 아직 채택하지 않았다.

## 공개

코드는 MIT로 이 저장소에 공개한다. Adapter는 검증한 checkpoint의 safetensors/config/model card/SHA256을 GitHub Releases에 묶는다. Base weights, dataset, optimizer state와 개인 서버 운영 정보는 Git에 넣지 않는다. 별도 HF 배포 저장소는 만들지 않는다.

직접 discrete action diffusion과 NVFP4 적용 자체를 최초 기여로 주장하지 않는다. 프로젝트의 가치는 실제로 실행한 학습·변환·성공률/지연 검증과 재현 가능한 구현에서 확인한다.

## JEV 기반 판단과 설명가능성 (2026-10-24)

사용자가 제안한 핵심 방향은 행동 생성과 함께 불확실성을 드러내고, 실행 여부를 판단한 근거를 확인할 수 있게 하는 것이다. 아래는 후속 설계이며 현재 학습·평가 설정을 변경하거나 추가 GPU 작업을 시작한 기록이 아니다.

### 원래 방법과 현재 적용 범위

[JEV 공식 소개](https://docs.typesafe.ai/introduction)는 typed answers, probability distributions, 병렬 판단, 코드에서의 조합을 설명한다. [공식 confidence 문서](https://docs.typesafe.ai/confidence)는 confidence를 해당 출력분포에서 계산하는 통계량으로 정의하고 Choice와 순서가 있는 Score의 공식을 구분한다. Confidence는 별도 검증 모델의 판정이나 작업 성공 확률이 아니다. [JoshuaSP/open-jev](https://github.com/JoshuaSP/open-jev)는 DiffusionGemma의 허용 token readout 실험이며 Jev의 아키텍처·학습·calibration 재현이 아니라고 명시한다. 출처 확인일은 2026-10-24이며 이 확인만으로 선행연구 전체를 조사했거나 신규성을 확정한 것은 아니다.

현재 DecisionVLA는 native action-token vocabulary 제한, 56개 위치 병렬 복원, 관측 prefix KV 재사용을 구현했다. `predict_with_confidence()`는 마지막 denoising 단계의 행동별 확률·entropy·거리 기반 신뢰도를 반환하며 기존 action API와의 parity도 검증했다. 고정 4-step sampler는 confidence에 따른 조기 종료를 사용하지 않는다. 병렬 복원은 DiffusionGemma의 기능이며 JEV 고유의 속도 향상 효과가 입증된 결과로 소개하지 않는다. Action 위치들은 서로 조건화되므로 JEV의 독립된 typed question들을 그대로 복제한 것도 아니다.

### 첫 구현: 추가 neural module 없이 진단 가능한 출력

고정 checkpoint와 기존 action 출력을 유지한 상태에서 선택적으로 마지막 허용-bin 분포, gripper 확률, 연속 제어의 분포 폭, denoising 반복 간 변화량을 반환한다. 같은 seed에서 계측 전후 action과 RNG 진행이 유지되는지 먼저 확인한다. 여러 noise seed의 불일치는 추가 비용이 있는 보조 진단이며 그것만으로 epistemic uncertainty를 분리했다고 주장하지 않는다.

- Gripper: 두 유효 class의 확률과 peak probability를 확인한다.
- 연속 6축: 최고 확률·정규화 entropy 외에 순서가 있는 bin의 제어값 거리, 분산, 선택값 주변 허용오차 안의 probability mass를 비교한다. 인접 bin의 애매함과 반대 방향 행동의 애매함을 구분한다.
- JEV Choice confidence는 고정 선택지 수에서 peak probability의 선형 변환이므로 둘을 서로 독립적인 새 신호로 취급하지 않는다. 순서를 반영한 Score 방식은 비교 후보이며 로봇 제어에서의 유효성은 별도 검증한다.
- 56개 위치의 확률을 독립으로 가정해 곱하거나 평균한 수치를 chunk 성공 확률로 이름 붙이지 않는다. 마지막 denoising 조건부 분포의 날카로움도 초기 noise를 적분한 전체 정책 불확실성과 동일하지 않다.

기존 LoRA checkpoint로 분포를 읽는 데 추가 본학습은 필수가 아니다. Probability calibration에는 별도로 분리한 데이터가 필요하다. Token-level 정답 일치나 시연 행동 허용오차를 보정한 결과를 로봇의 closed-loop 성공 확률로 바꾸어 설명하지 않는다. 후자는 실패와 성공이 포함된 rollout label로 별도 검증해야 한다.

### 어느 요소가 유효했는지 보여줄 비교

우선 고정 정책에서 신뢰도와 행동 오차·실패의 관계를 분석한다. Task와 진행 단계별 결과를 나누어, 단순히 어려운 task 또는 후반 단계만 구분한 효과인지 확인한다. 성공·실패 한쪽 label만 존재하면 실패 구분 능력을 평가할 수 없다고 기록한다.

다음에는 같은 checkpoint, 초기상태, seed 규칙과 제어 예산에서 고정 denoising 1/2/4/8, 신뢰도에 따른 반복 수, 신뢰도에 따른 짧은 chunk 실행을 각각 비교한다. 짧게 실행하는 효과는 항상 짧은 chunk를 쓰는 대조군과 비교하고, 선택적 실행은 같은 실행 비율의 단순 또는 무작위 gate와 비교한다. 실행 전 추가 추론과 관측 비용도 전체 latency에 포함한다. 멈춤·개입한 episode를 성공률 분모에서 빼서 개선으로 보이지 않게 한다.

보고 항목은 task 성공률, p50/p95 latency, 실제 denoising 횟수, 실행 비율, 재관측·개입 횟수, calibration curve와 label 정의에 맞는 Brier/ECE 또는 tolerance coverage이다. Threshold와 calibration은 개발 데이터에서 결정한 뒤 평가 데이터에서는 고정한다. NVFP4 변환 후에는 action뿐 아니라 분포와 calibration의 변화도 다시 확인한다. 기존 공식 초기상태의 개발 사용 이력은 보존한다.

### 설명가능성의 범위

첫 목표는 불확실성과 실행 규칙을 추적할 수 있는 것이다. 예를 들어 실제 계산된 지표와 고정 threshold로부터 “gripper의 두 선택이 비슷해 실행 chunk를 줄임”이라는 reason code를 만들 수 있다. 이는 controller의 결정 근거이며, 신경망이 물체를 선택한 인과적 이유를 설명한 것은 아니다. 생성한 자연어 rationale를 근거로 삼지 않는다.

다음 후보는 같은 native canvas의 일부 위치에 현재 상태의 구조화된 판단을 함께 학습하는 것이다. 예시는 대상 식별·grasp 여부·목표 충족이며, 추가 외부 VLM 없이 구현할 수 있는 설계 후보다. 시뮬레이터 label의 정의, 입력 영상으로 관측 가능한지, action과 함께 예측하도록 할지 먼저 결정해야 한다. 현재 200개 padding을 재사용하려면 mask·target·loss가 바뀌므로 별도 학습 버전이 필요하다. 이 설계는 아직 채택·구현하지 않았다.

상태 설명과 action을 함께 출력하는 것만으로 설명의 충실성이 보장되지는 않는다. 관측·지시문 변화와 해당 판단에 대한 개입에서 action이 예상대로 달라지는지 확인해야 한다. 설명을 정책이 실제 사용하는 정보에 연결했다는 근거가 없으면 “상태 진단” 또는 “감사 가능한 실행 근거”로 소개한다. 본학습 전에 먼저 고정 pilot의 action 분포가 유용한 신호를 주는지 확인하는 것을 권장한다.


### 확률 API 구현 반영 (2026-10-24)

위 계획의 첫 단계는 `predict_with_confidence()`와 per-query trace 저장으로 구현했습니다. 마지막 readout에서 8×7×256 확률을 읽으며 추가 forward/RNG를 사용하지 않습니다. 연속 값은 mode 기준 거리 신뢰도, gripper는 choice 신뢰도이며 모두 미보정입니다. `configs/confidence_analysis.json`을 실제 rollout 결과를 보기 전에 고정했습니다. 500회 Spatial 평가 후 첫 query의 불확실성과 최종 실패를 task 내부 AUROC·bootstrap·permutation으로 검증합니다. 자세한 API/제약은 [confidence.md](confidence.md)에 있습니다. 실행 제어·성공확률 calibration·인과적 설명은 여전히 후속 과제입니다.

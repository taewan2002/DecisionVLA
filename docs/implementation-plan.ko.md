# DecisionVLA 구현과 다음 단계

2026-10-24. 개인 오픈소스 프로젝트이며 기존 학위논문 코드·checkpoint·결과와 분리되어 있다.

## 현재 구현

공식 DiffusionGemma BF16 원본 위에 공유 attention LoRA와 action embedding delta를 학습한다. RGB 1장과 instruction을 입력하고, native Transformer가 8×7개의 action bins를 병렬 복원한다. 별도 action expert, world model 또는 후보 선택 모델은 없다.

첫 실행은 GPU 1장으로 고정했다. LIBERO-Spatial 10개 task의 450개 시연으로 학습하고 50개 시연을 검증에 쓴다. Rank 16, effective batch 8, 2,000 updates와 12시간 상한이다. 수치와 데이터 계약은 [training.md](training.md), 실행 설정은 [config](../configs/libero_spatial.json), 실제 진행은 [run 기록](../reports/libero-spatial-pilot/README.md)을 따른다.

이번 pilot은 proprioception과 self-conditioning을 제외했다. 전자는 관측 추가 비교, 후자는 정확한 soft embedding과 sampler를 함께 구현할 때 비교할 후속 항목이다. Action은 고정 controller 범위 `[-1, 1]`을 사용하므로 train statistics fit을 하지 않는다.

## 평가

첫 학습이 끝나면 고정 checkpoint로 LIBERO closed-loop rollouts를 수행한다. 현재 open-loop MAE나 loss를 성공률로 해석하지 않는다. 개발에 사용한 초기상태는 별도로 기록하고, 개발 결과로 checkpoint/step 수를 정한 뒤 별도 초기상태에서 확인한다. 기존 학위논문에서 쓴 초기상태를 새 test라고 부르지 않는다.

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

현재 단일 GPU 경로의 학습·평가를 먼저 검증한다. 두 장 확대 시 같은 global batch와 sampler를 유지하고 DDP gradient/재개를 검증한 뒤 학습 예산을 늘린다. GPU 수 증가가 첫 pilot의 성공을 보장하지는 않는다. Native 256→64 canvas, action 범위 정규화, self-conditioning, camera 추가는 각각 분리해서 비교한다.

## 공개

코드는 MIT로 이 저장소에 공개한다. Adapter는 검증한 checkpoint의 safetensors/config/model card/SHA256을 GitHub Releases에 묶는다. Base weights, dataset, optimizer state와 개인 서버 운영 정보는 Git에 넣지 않는다. 별도 HF 배포 저장소는 만들지 않는다.

직접 discrete action diffusion과 NVFP4 적용 자체를 최초 기여로 주장하지 않는다. 프로젝트의 가치는 실제로 실행한 학습·변환·성공률/지연 검증과 재현 가능한 구현에서 확인한다.

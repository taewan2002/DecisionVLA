# DiffusionGemma 추론 비용 실측 — 2026-10-24

## 판단

RTX 5090 한 장을 목표로 하는 direct VLA의 첫 구현은 **NVFP4, 카메라 1개, 8-action chunk, 2–4 denoising steps**로 진행할 근거가 생겼다. 측정한 RTX PRO 6000에서는 4-step 입력→출력 중앙값이 카메라 1개 144.68ms, 2개 186.23ms였다. 양자화 조건의 전체 device memory 관측 최대는 22.26GiB였다.

이는 **로봇 행동을 학습하지 않은 base model의 workload proxy**다. 새 VLA의 성공률·최종 지연이나 RTX 5090의 실측 결과가 아니다. 이 측정에는 실제 action codec/PAD 고정/제한 readout이 포함되지 않았다. 후속 BF16 reference 구현은 별도 학습 run에서 검증한다. 다음 핵심 질문은 2–4 steps에서 충분한 행동 품질을 학습할 수 있는가이다.

## 측정 조건

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, GPU 1장, 기존 power limit **500W** 유지. 다른 GPU의 기존 작업은 실행 상태를 유지했다.
- CPU: AMD Ryzen Threadripper 9970X; 측정 컨테이너 8 CPU/112GiB host RAM 제한, OMP threads 8.
- NVIDIA driver 615.71.09, Torch 2.13.0+cu130, Transformers 5.17.0, vLLM 0.30.0 + 아래 호환성 patch.
- Batch 1, 256×256 synthetic RGB 1/2개, visual-token budget 280/image. 실제 prefix는 각각 **332/590 tokens**였다.
- 관측마다 서로 다른 이미지. Prefix cache와 processor cache off. 요청 간 동일 vision 결과 재사용 없음.
- Full-vocabulary native **256-token canvas**를 처리한 뒤 56 text token IDs 반환. 8×7 action의 출력 크기를 모사했으며 실제 action은 아니다.
- Fixed steps 1/2/4/8, adaptive confidence stopping off, synchronous scheduling, BF16 KV cache 512MiB, no CPU weight offload.
- 조건마다 warmup 5회 + 본측정 30회. 2 precisions × 2 views × 4 steps = **480 timed requests**, warmup 별도 80회.
- 입력 PIL image와 instruction부터 chat template, 전처리, vision, 새 prefill, denoising, commit, 로컬 engine 반환까지 포함. 카메라 acquisition, 원격 network, 실제 action 변환/제어는 제외. 단계별 시간을 따로 분해한 결과는 아니다.

## 결과

단위 ms. 각 셀은 **p50 / p95**이며 warmup/모델 로딩/컴파일은 제외했다.

| 이미지 수 | Denoising steps | NVFP4 | BF16 |
|---:|---:|---:|---:|
| 1 | 1 | 89.78 / 90.86 | 128.91 / 130.20 |
| 1 | 2 | 107.39 / 109.30 | 157.33 / 160.45 |
| 1 | 4 | 144.68 / 146.38 | 221.19 / 225.72 |
| 1 | 8 | 221.35 / 222.82 | 352.54 / 356.94 |
| 2 | 1 | 128.07 / 129.53 | 169.39 / 170.12 |
| 2 | 2 | 146.44 / 147.85 | 198.99 / 200.99 |
| 2 | 4 | 186.23 / 187.42 | 265.34 / 268.55 |
| 2 | 8 | 266.19 / 269.04 | 399.92 / 403.41 |

이미지 1개·4-step에서 NVFP4는 BF16 대비 **1.53배** 빠르다. 같은 이미지와 같은 요청 순서를 사용했지만, quantization에 따른 출력 차이는 별도다. 행동 정확도 비교를 한 것은 아니다.

| 메모리 지표 | NVFP4 | BF16 |
|---|---:|---:|
| PyTorch max allocated, 측정 조건 중 최대 | 20.45GiB | 50.83GiB |
| PyTorch max reserved, 측정 조건 중 최대 | 20.94GiB | 51.30GiB |
| 전체 GPU 사용량 관측 최대, startup 포함 | 22.26GiB | 53.02GiB |

전체 GPU 사용량은 `nvidia-smi` 200ms sampling의 최대이며, 모든 순간 peak를 포착한다는 뜻은 아니다. PyTorch 값은 worker allocator가 기록한 값이다. 96GB GPU에서 memory budget과 KV cache를 작게 설정한 측정이며, 5090의 실제 32GB 메모리를 강제로 에뮬레이션한 것은 아니다.

## 구현 결정에 주는 의미

1. 4-step은 측정 GPU에서 이미지 1/2개 모두 p95 200ms 목표 안에 들었다. 8-step은 이 목표를 넘었다. 먼저 2/4-step 행동 품질을 확인한다.
2. 4-step 모델 호출은 중앙값 기준 약 6.9회/초(1 view), 5.4회/초(2 views)다. 8 actions를 출력한다고 feedback이 그 8배 빨라지는 것은 아니다. 20Hz 저수준 제어와 chunk 재계획 주기를 구분한다.
3. 현재 NVFP4 memory footprint는 5090 배포를 검토하기에 여유가 있다. BF16 원본은 이 조건에서 32GB를 초과한다. 최종 LoRA merge/양자화 모델과 5090에서 재측정해야 한다.
4. RTX PRO 6000과 5090은 compute·clock·power·VRAM이 달라 이번 숫자를 그대로 5090 속도로 쓰지 않는다. 이번 결과로 정확한 환산 계수를 만들지 않았다.
5. Synthetic image와 미학습 text 출력의 MoE routing은 실제 로봇 입력/학습된 action 출력과 다를 수 있다. Native canvas의 PAD 고정과 constrained readout, 최종 commit 제거 같은 변경도 별도 검증한다.

## Runtime 호환성과 검산

원본 vLLM 0.30.0의 DiffusionGemma 이미지 경로는 Gemma4 image 함수가 참조하는 `_enable_mm_lora` 속성을 초기화하지 않아 실패했다. No-LoRA base/merged-weight 조건에서 기본값 `False`를 추가했다. 가중치나 tensor 계산식은 바꾸지 않았다. Patch와 변경 전후 source SHA를 공개 파일로 포함했다.

비동기 초기 probe는 한 요청에서 추가 256-token pass를 예약했다. 최종 측정은 synchronous scheduling으로 고정하여, 각 요청의 **prefill 1회 + 지정 denoise S회 + commit 1회**와 **256-token pass S+1회**를 모두 검사했다. 초기 호환성/측정 조건 검증에 실패한 run은 위 성능 표에서 제외했다.

독립적인 표준 Python 코드로 raw samples의 p50/p95, 모든 step/canvas 실행 횟수, unique image hashes, 출력 길이, 두 precision의 입력 일치를 재계산했다. 두 유효 run은 exit 0이며 원래 miner container가 복원됐고 보호 대상 작업의 container ID/PID/start time이 유지됐다. 이 benchmark run은 학습을 수행하지 않았다.

## 재현 자료

- [측정 스크립트](../../benchmarks/latency_vllm.py), [runtime patch wrapper](../../benchmarks/run_patched_vllm.py)
- [재현 명령과 측정 범위](../../docs/benchmark-protocol.md)
- [독립 검산 코드](../../benchmarks/audit_results.py), [검산 결과](audit.json)
- [NVFP4 raw results](nvfp4/results.json), [BF16 raw results](bf16/results.json)
- [NVFP4 GPU telemetry](nvfp4/gpu-telemetry.csv), [BF16 GPU telemetry](bf16/gpu-telemetry.csv)
- [Model revisions·container digest·hardware](environment-manifest.json)
- [자원 복원 검산](resource-audit.json)
- [다운로드한 15개 LFS 파일의 원본 SHA 일치 검산](weight-integrity.json)

개인 서버 식별정보가 들어 있는 운영 script와 전체 host audit/log는 공개 대상에서 제외했다. 학습·평가·5090 검증 계획은 [구현 계획](../../docs/implementation-plan.ko.md)에 있다.

GPU 0·2를 함께 사용하던 miner는 측정 동안 GPU 2 단독 임시 컨테이너로 전환했고, 종료 후 원래 공유 컨테이너로 복원했다. GPU 1·3의 기존 작업은 유지했다.

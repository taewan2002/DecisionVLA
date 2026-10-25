# Rank-32 attention, MLP and expert pilot

Configuration: [`libero_spatial_r32_all.json`](../../configs/libero_spatial_r32_all.json).
This is a fresh-base LIBERO-Spatial pilot with rank 32 / alpha 64, effective batch 8 and a 2,000-update budget. The completed rank-16 checkpoint is preserved. The dataset, episode split, seed and objective match the first pilot; changing both rank and placement means this comparison cannot isolate the contribution of either change.

## Adaptation scope

| Family | Scope | Trainable parameters |
|---|---|---:|
| Attention | 115 existing Q/K/V/O projections | 22,978,560 |
| Dense MLP | gate/up/down in all 30 text layers | 14,192,640 |
| MoE experts | Independent factors for 128 experts in each of 30 layers | 951,582,720 |
| Action embeddings/readout | 256 tied action rows | 720,896 |
| Total | | 989,474,816 |

The encoder and decoder share each adapter. Within an expert, fused gate/up uses one shared rank-32 input factor; down uses a separate factor pair. Original parameters, routers and vision remain frozen. No external action network is introduced.

## Checks before the main run

- Ten CPU tests passed, including independent merged-weight forward/gradient comparisons, unused-expert gradients, encoder/decoder sharing, single/sharded checkpoint round trips and release packaging. Ruff passed.
- A real-data GPU smoke run completed three updates, a save/reload comparison and a disposable 24-update fixed-frame fit. All four adapted families and the observation prefix received finite, nonzero gradients.
- Peak PyTorch allocation was **67.518 GiB**; peak reservation was **67.719 GiB**, on one RTX PRO 6000 Blackwell 96GB.
- Restarted step loss matched exactly. Maximum adapter parameter error was **4.61123e-6**, below the declared `1e-5` tolerance. Backward was not bitwise deterministic.
- Disposable fixed-frame CE decreased from **7.1854 to 0.7433**. These updates are discarded and do not demonstrate policy generalization.
- FP32 adapter data total **3,957,899,264 bytes** and are saved in four safetensors shards. Optimizer state is kept locally and excluded from release packaging.

Raw numerical smoke evidence: [`smoke-events.jsonl`](smoke-events.jsonl). The smoke used accumulation 1; its update timings are not the effective-batch-8 training throughput.

## Main execution

Run `libero-spatial-r32-all-v2` was launched on 2026-10-24 at 15:02 KST on GPU0. The first launch attempt ended before model loading or any update because a concurrent resource transition changed the miner's device layout. That attempt is preserved; acquiring the resource lock before inspecting the layout fixed the race. Other workloads were not stopped.

At **15:04 KST**, the main run had completed **9 / 2,000 updates**, with finite losses and gradients. The eight updates after the first had median duration **6.693 seconds/update** at effective batch 8. Peak PyTorch allocation/reservation was **71.119 / 71.350 GiB**. Extrapolating this initial throughput gives approximately **four hours including validation and checkpoint overhead**, subject to runtime variation. This is training throughput, not inference latency.

The first accumulated update confirmed nonzero gradients in attention, dense MLP, MoE experts and action rows, as well as the observation prefix. The original rank-16 run's 24 source files and four final-checkpoint files were rechecked against their saved SHA256 values. The timestamped evidence is [`start-audit.json`](start-audit.json).

The run uses a frozen source copy. All 15 local code/config/test/script files matched its recorded hashes at launch. [`source-sha256.json`](source-sha256.json) records the frozen tree. Checkpoints are saved every 500 updates and development validation runs every 100 updates on 20 fixed held-out frames. A 12-hour training deadline bounds the run. No 10k continuation is scheduled.

Closed-loop success, gains over rank 16, merged NVFP4 policy quality and actual RTX 5090 speed remain unmeasured.

## Automatic evaluation handoff

On 2026-10-24 at 15:14 KST, a server worker was registered to start evaluation on the same GPU0 after successful completion of update 2,000. Effective training batch remains 8. The pinned evaluation uses all ten LIBERO-Spatial tasks and official initial states 0–49, for 500 episodes, four denoising passes and eight-control chunks. A real-checkpoint preflight must pass before the scored run. Training failure or timeout blocks the handoff.

Sixteen CPU tests, five completion-gate cases, Ruff, all ten simulator tasks with a CPU stub policy, and video encoding/decoding passed. The CPU smoke is not a learned-policy result. The policy runtime, simulator image and 43 source files are frozen; the eight original training core source files are unchanged. A 15-minute follow-up check reports meaningful transitions, completion or errors. The server worker performs the immediate handoff without waiting for that check.

See the [evaluation contract](../../evaluation/README.md) and [registration evidence](evaluation-handoff.json). Real-policy rollout success is pending.

## Training power limit

On 2026-10-24 at approximately 18:27 KST, the owner authorized a 600W training ceiling. GPU0's limit was raised from 500W to its supported 600W maximum, with training at step 1,865 when recorded. The last 60 telemetry samples averaged 263.14W and peaked at 280.77W; a higher ceiling alone is not evidence of higher throughput. A run-specific worker restores the prior 500W limit after this training container exits. Other GPUs were unchanged. Earlier timing results retain their original conditions. [Recorded evidence](power-limit-change.json).


## Completion and confidence evaluation

The rank-32 run completed **2,000 / 2,000 updates**, with reported elapsed training time **13,260.69 seconds** and final training CE **1.5288857**. This loss is not a closed-loop success metric. The final four-shard adapter remains unchanged. GPU0 returned to its prior 500W limit.

The first automatic evaluation stopped before any real-policy rollout: a root-owned Unix socket was inaccessible to the host readiness check. A subsequent non-root policy setup exposed the existing root-only checkpoint-file permissions, again before any rollout. Both attempts are preserved. The corrected policy keeps the checkpoint permissions intact and transfers ownership only of its new private socket.

The confidence-enabled evaluation uses the same final checkpoint, sampling and rollout protocol; it adds observational statistics, not an intervention. See the [confidence evaluation record](../libero-spatial-confidence/README.md). No additional training has been started.

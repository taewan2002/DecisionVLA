# Training contract

> **Historical procedure — superseded on2026-10-25.** Training was checkpoint-stopped; the forty-task run ended at46 and its automatic final evaluation did not start. These commands describe the earlier action-only implementation. See the [current DecisionVLA design](architecture.md) and [development plan](development.md); do not restart old controllers.

For the current fresh forty-task, batch64/100k run, use
[the four-suite training contract](training-libero40.md). The original pilot
configuration and measurements below retain their historical scope.

The first run is a single-GPU LIBERO-Spatial pilot. It is not an OpenVLA reproduction, a JEV reproduction, or a measured NVFP4 policy. The earlier [implementation plan](implementation-plan.ko.md) records design alternatives; the checked-in config is the execution contract.

## Model and objective

The pinned DiffusionGemma model has a causal observation encoder and a bidirectional canvas decoder with tied text weights. DecisionVLA shares every adapted module between both paths. The original `libero_spatial.json` configuration adapts only attention with rank 16 / alpha 32. It freezes the original weights, vision tower, vision projection, MoE experts and routers.

The additional `libero_spatial_r32_all.json` configuration uses rank 32 / alpha 64 on:

| Family | Adapted projections | Trainable parameters |
|---|---|---:|
| Attention | 115 Q/K/V/O projections | 22,978,560 |
| Dense MLP | gate/up/down in all 30 text layers | 14,192,640 |
| MoE experts | gate_up/down for 128 experts in each of 30 layers | 951,582,720 |
| Action rows | Tied 256 × 2816 embedding/readout delta | 720,896 |
| Total | | 989,474,816 |

Each expert has independent rank-32 factors. Its fused gate/up projection shares one input factor across gate and up outputs; down has its own pair. The same factors are used by encoder and decoder. Routed computation uses grouped GEMMs for both base projections and low-rank branches, without materializing the full expert weight updates. The router, vision tower/projection, normalization layers and all original weights stay frozen.

This placement is motivated by [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), which found benefits from adapting MLP/MoE layers in language-model experiments. It does not establish a VLA gain. [OpenVLA Table 1](https://arxiv.org/html/2406.09246v3#S5.T1) reported the same mean success for ranks 32 and 64 in its setting. Our expanded pilot changes both rank and placement relative to the first run; it does not isolate either factor causally.

The expanded run starts from the pinned original base, with fresh adapters and the same data split/seed, for an initial 2,000-update comparison. Other training settings remain as below, except checkpoint saving every 500 updates and AdamW `foreach=False` to bound temporary optimizer memory. Validation remains every 100 updates. This is the historical 2k configuration. Its small-batch continuation was stopped at update2,534 with checkpoint2,500 preserved; the current experiment is a [fresh two-GPU batch64/100k run](training-b64-100k.md).

Action tokens use native vocabulary IDs `261888..262143`. Loading checks that all 256 IDs are unused and not special tokens. A small trainable delta updates only those input/output embedding rows. The constrained readout is a slice of the native tied LM head, not a separate action network. The adapter formats are `decisionvla-shared-lora-v1` (original) and `decisionvla-shared-lora-v2` (expert extension); these are **not** interchangeable PEFT adapters.

The first six action dimensions use nearest-neighbor quantization over 256 evenly spaced points in `[-1, 1]`. Maximum in-range quantization error is `1/255` in normalized control units. Gripper uses bins 0 and 255, preserving LIBERO's original control sign. No fitted statistics are needed for this fixed controller range. This recipe is specific to the Panda OSC_POSE controls in the pinned dataset.

The target is eight consecutive 7D controls, in timestep-major order. Remaining positions in the native 256-token canvas are fixed to PAD and masked as attention keys. Every sample has at least one valid future action; positions past the episode end are masked out of the loss. Inference always emits eight actions.

For each example, sample a uniform replacement probability, with an additional 20% chance of full corruption. Replace selected components with uniformly sampled allowed bins. The original 2k objective computes cross entropy on **all valid action positions**, including those left uncorrupted. The corrected continuation computes it only on valid positions selected for random replacement, including chance replacements equal to the target. This is an unweighted denoising objective; it is not claimed to reproduce the full upstream training recipe or ELBO. There is no causal next-token shift in the decoder.

Self-conditioning is disabled in both training and reference inference. The reference sampler predicts categorical clean actions and re-noises with a linearly decreasing replacement probability, returning argmax at the final pass. This sampler differs from vLLM's native entropy sampler; the earlier latency proxy cannot be relabeled as its measured speed.

## Observation/action alignment

Data source: the [official LIBERO mirror](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets), revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`. Downloaded files are checked against upstream SHA256 values.

The original [dataset creation script](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py) steps `action[t]` before saving `obs[t]`. The stored simulator state is the preceding state. We independently verified `obs/joint_states[t] == states[t+1, 1:8]` at all 61,750 available adjacent pairs across 500 demonstrations, with maximum absolute error 0. The policy therefore conditions on `obs[t]` and predicts actions beginning at `t+1`. The last frame, with no following target action, is excluded.

Use one `agentview_rgb` image, rotated 180 degrees, and the task instruction. Current proprioception is not included in this pilot. Real deployment must apply the same image transform. Do not add future observations or action targets to the prefix.

The split is deterministic: demonstration indices whose number modulo 10 equals 9 are validation data. This produces 450 training and 50 validation episodes over ten tasks. Frames are not split independently. Both the split manifest and original file hashes are recorded in `run.json`.

## Execution

`configs/libero_spatial.json` specifies:

| Parameter | Value |
|---|---|
| GPUs / microbatch | 1 / 1 |
| Accumulation / effective batch | 8 / 8 |
| Updates | 2,000 |
| Optimizer | AdamW, FP32 adapter parameters, BF16 model compute |
| Peak learning rate | 1e-4 |
| Warmup / decay | 100 updates / cosine to 10% of peak |
| Gradient clipping | 1.0 |
| Checkpoint and validation interval | 100 updates |
| Validation | 20 fixed held-out frames, four denoising passes |
| Training deadline | 12 hours, save at the next completed update |

Validation reports open-loop continuous action MAE, gripper accuracy, token accuracy and denoising CE. **None is a closed-loop task success rate.** Frame count is small, and repeated validation makes these development metrics.

Decoder activation checkpointing uses non-reentrant recomputation. Encoder prefix KV remains attached to autograd. An explicit vision attention mask handles the pinned Transformers encoder's discarded mask-builder return value. Grouped MoE GEMMs are selected through Transformers' existing `grouped_mm` implementation.

Run the commands in the README. To resume the same run, preserve the exact configuration and point to its latest complete checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m decisionvla.train \
  --model models/base --data data --output runs/libero-spatial \
  --resume runs/libero-spatial/checkpoint-000100
```

Adapter tensors are stored as `safetensors`. Local optimizer/RNG state is in `training_state.pt`; only load state you created or trust. Each checkpoint directory is atomically renamed after all files and checksums are written. A resumed run preserves the optimizer step, data sampler and noise generator.

The smoke run tests real image/action backward, prefix gradient, checkpoint reload, and a disposable 24-update single-frame overfit. Numerical restart is checked within a declared tolerance; GPU backward kernels are not claimed to be bitwise deterministic. Smoke adapters and overfit state are not released as trained policies.

## Deployment work still required

The BF16 reference API can load an adapter and output normalized LIBERO controls. The [LIBERO evaluator](../evaluation/README.md) is implemented for the final rank-32 checkpoint; the 2k pilot completed 500 episodes with 14 successes (2.8%), while corrected-checkpoint results remain pending. Hardware action checks, adapter merge, NVFP4 calibration/export and accelerated sampler parity remain separate unfinished stages. The public benchmark currently measures the untrained NVIDIA NVFP4 base with vLLM, not a DecisionVLA-trained quantized policy.
# Historical 10k continuation

The 2026-10-24 follow-up diagnosed confidence across denoising passes, removed loss on positions not selected for replacement, and started extending the preserved rank-32 checkpoint after its 2k evaluation. It was stopped by owner request at update2,534 on2026-10-25. See the preserved [diagnosis and continuation contract](../reports/denoising-correction/README.md). The corrected model is not yet validated.

## Superseded 50k plan and current 100k experiment

[The 50k contract](training-50k.md) is retained as history; its queued controller was stopped before continuation training. The replacement [batch64/100k contract](training-b64-100k.md) uses fresh LoRA, two-GPU DDP, a single100k LR schedule, and500-episode evaluations every10k. Publication waits for all ten evaluations and their audits.

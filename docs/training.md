# Training contract

The first run is a single-GPU LIBERO-Spatial pilot. It is not an OpenVLA reproduction, a JEV reproduction, or a measured NVFP4 policy. The earlier [implementation plan](implementation-plan.ko.md) records design alternatives; the checked-in config is the execution contract.

## Model and objective

The pinned DiffusionGemma model has a causal observation encoder and a bidirectional canvas decoder with tied text weights. DecisionVLA shares each attention LoRA module between both paths. It freezes the original weights, the vision tower, the vision projection, MoE experts and routers. LoRA rank is 16 and alpha is 32.

Action tokens use native vocabulary IDs `261888..262143`. Loading checks that all 256 IDs are unused and not special tokens. A small trainable delta updates only those input/output embedding rows. The constrained readout is a slice of the native tied LM head, not a separate action network. The adapter format is `decisionvla-shared-lora-v1`; it is **not** an interchangeable PEFT adapter.

The first six action dimensions use nearest-neighbor quantization over 256 evenly spaced points in `[-1, 1]`. Maximum in-range quantization error is `1/255` in normalized control units. Gripper uses bins 0 and 255, preserving LIBERO's original control sign. No fitted statistics are needed for this fixed controller range. This recipe is specific to the Panda OSC_POSE controls in the pinned dataset.

The target is eight consecutive 7D controls, in timestep-major order. Remaining positions in the native 256-token canvas are fixed to PAD and masked as attention keys. Every sample has at least one valid future action; positions past the episode end are masked out of the loss. Inference always emits eight actions.

For each example, sample a uniform replacement probability, with an additional 20% chance of full corruption. Replace selected components with uniformly sampled allowed bins. Compute cross entropy on **all valid action positions**, including those left uncorrupted. This is an unweighted denoising objective; it is not claimed to reproduce the full upstream training recipe or ELBO. There is no causal next-token shift in the decoder.

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

The BF16 reference API can load an adapter and output normalized LIBERO controls. Closed-loop simulation, action safety checks for hardware, adapter merge, NVFP4 calibration/export and accelerated sampler parity are separate unfinished stages. The public benchmark currently measures the untrained NVIDIA NVFP4 base with vLLM, not a DecisionVLA-trained quantized policy.

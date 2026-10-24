# DecisionVLA

**A personal experiment adapting DiffusionGemma to predict robot actions directly.**

[한국어](README.md) · [Training](docs/training.md) · [Weights](docs/weights.md)

One image and an instruction condition an 8 × 7 action chunk. The model uses DiffusionGemma's existing vision encoder and shared Transformer, with no separate action expert or world model.

This repository shares the implementation and experimental evidence. It does not yet provide a validated robot policy or a trained NVFP4 checkpoint.

## Implemented

- Shared attention LoRA across encoder and decoder; frozen vision, experts and router.
- 256 verified unused native vocabulary entries, with trainable tied action embedding rows.
- Uniform action corruption and parallel denoising CE, without self-conditioning in this pilot.
- 56 action positions on a native 256-token canvas; fixed padding is excluded from attention keys.
- LIBERO-Spatial episode split: 450 training demonstrations and 50 validation demonstrations.
- Explicit post-action alignment: `obs[t] → actions[t+1:t+9]`, verified on all 61,750 adjacent states.
- Atomic adapter/optimizer checkpoints, a resume check, and a BF16 reference inference API.

The initial configuration uses one 96GB GPU, rank 16 LoRA, effective batch 8 and 2,000 optimizer updates. See the [run record](reports/libero-spatial-pilot/README.md) for actual execution status.

## Reproduce

Use the installation and launch commands in the [main README](README.md#시작하기), followed by the [training contract](docs/training.md). The BF16 reference implementation needs roughly 51 GiB or more and does not fit a 32GB RTX 5090.

The planned deployment path is BF16 + LoRA merge → robot-domain calibration → NVFP4 export → numerical and paired rollout checks → measurement on an actual RTX 5090.

## Measured base-model cost

On one RTX PRO 6000 Blackwell at its existing 500W limit, the **untrained base-model workload proxy** measured:

| One image, 256-token canvas | NVFP4 p50 / p95 | BF16 p50 / p95 |
|---|---:|---:|
| 2 denoising steps | 107.4 / 109.3 ms | 157.3 / 160.5 ms |
| 4 denoising steps | 144.7 / 146.4 ms | 221.2 / 225.7 ms |

Sampled whole-device memory peaked at 22.26 GiB (NVFP4) and 53.02 GiB (BF16). These are synthetic-input vLLM measurements, not trained action-sampler latency, robot success, or RTX 5090 measurements. [Protocol](docs/benchmark-protocol.md) · [Raw evidence](reports/base-latency-2026-10-24/).

## Distribution and attribution

Code: MIT. Base weights, quantized weights and datasets retain their upstream terms; see [third-party sources](docs/third-party.md).

Validated adapters will be distributed as GitHub Release assets with their config, model card and checksums. No separate Hugging Face publishing repository is required. Base weights, datasets and optimizer states stay out of Git.

This is an independent project, unaffiliated with Google, NVIDIA, LIBERO or open-jev. It makes no first-of-its-kind claim for action diffusion or NVFP4.

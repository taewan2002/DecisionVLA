<div align="center">

# DecisionVLA

**Parallel actions · Visible uncertainty**

An experiment in reading robot actions and probabilities from one observation.

[한국어](README.md) · [Architecture](docs/architecture.md) · [Experiments](docs/experiments.md) · [Dashboard](docs/dashboard.md)

![Status: Experimental](https://img.shields.io/badge/status-experimental-CE9147)
![Backbone: DiffusionGemma](https://img.shields.io/badge/backbone-DiffusionGemma-43789D)
[![License: MIT](https://img.shields.io/badge/code-MIT-398779)](LICENSE)

</div>

![Proposed DecisionVLA architecture: shared Transformer, native denoising and entropy-bound sampling, final action and probability readout.](figures/decisionvla-native/architecture.png)

<p align="center"><a href="https://www.figma.com/design/FbL9C7OnlyOZkiyGPOkcW0/?node-id=13407-92873">Editable Figma</a> · <a href="figures/decisionvla-native/">Figure source</a></p>

DecisionVLA adapts DiffusionGemma's existing vision encoder and shared Transformer to robot control. The target interface returns an **8 × 7 action chunk**, per-action distributions and uncertainty from an image and instruction. This repository shares implementation, inference optimization and failure analysis with their measurement records.

> **Development status · 2026-10-25**
> Forty-task training was intentionally stopped after saving **checkpoint 46**. The figure describes a **proposed architecture** preserving native denoising, sampling and self-conditioning. Executable code and measurements below use the earlier action-only sampler; the new path is not implemented, trained or benchmarked yet. [Development plan](docs/development.md) · [Stop audit](reports/openjev-redesign-20261025/stop-audit.json)

## Direction

| | DecisionVLA |
|---|---|
| **One backbone** | Shared text Transformer and LoRA for observation encoding and action denoising |
| **Parallel actions** | Reuse one observation's KV cache across joint action-canvas refinement |
| **Visible distributions** | Inspect final probabilities, entropy, selected probability and control variance |
| **One-GPU inference** | Test expert NVFP4 W4A16, fused kernels and CUDA Graph on an actual RTX 5090 |
| **Auditable experiments** | Pin data, configurations and checkpoint hashes; independently verify results |

The proposed path retains the **full vocabulary during intermediate denoising** and restricts only the final action readout. It keeps the native entropy-bound sampler and self-conditioning, with planned 1/2/4-pass quality–latency comparisons. [Data flow, exact differences and upstream references](docs/architecture.md).

## Measured so far

**RTX 5090 · previous 4,500-step Spatial pilot · four denoising passes · eight actions + confidence**

| Measurement | Result |
|---|---:|
| W4A16 + fused + CUDA Graph, p50 / p95 | **188.43 / 190.35 ms** |
| Peak PyTorch allocation | **18.815 GiB** |
| Twenty paired LIBERO episodes, BF16 / W4A16 | **12 / 11 successes** |
| Separate 2k pilot, 500 Spatial episodes | **14 / 500 successes** |

[5090 timings, action differences and intervals](reports/inference-nvfp4-5090-20261025/final/) · [2k failure analysis](reports/libero-spatial-confidence/)

Warm timings cover preprocessing through CPU actions/confidence, excluding network, robot execution and first compilation. Memory is peak PyTorch allocation, not total device usage. The paired W4A16−BF16 success difference was −5 percentage points, 95% CI [−35, +25]; twenty episodes do not establish equivalence. **These are not results for the proposed architecture or a final forty-task policy.**

## Inspect inference

The local dashboard shows observations, the 8 × 7 action chunk, per-control distributions, confidence and latency. Replay saved traces or attach a recorder to an existing policy. [Dashboard guide](docs/dashboard.md).

```bash
python -m decisionvla.dashboard --session results/recorded-session --mode replay
```

**Confidence 0.9 does not mean 90% task success.** Current metrics summarize conditional action-distribution concentration. Failure discrimination and calibration require separate evaluation; a probability display is not a causal explanation. [Probability contract](docs/confidence.md).

## Explore the code

The existing BF16 pilot targets Linux, Python 3.12 and CUDA 13. The proposed native path will validate a separate runtime.

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

Obtain the base model and datasets from their upstream sources. No adapter release is available yet. The BF16 API needs more than 50GiB VRAM; the 5090 path uses separate [NVFP4 reproduction tools](benchmarks/NVFP4.md). [Pilot training](docs/training.md) · [Weight format](docs/weights.md).

## Next milestones

- [x] Action/probability API, dashboard and audited LIBERO evaluation
- [x] Actual RTX 5090 W4A16, fused-kernel and CUDA Graph pilot
- [x] Original forty-task LIBERO data alignment and split audit
- [x] Preserved training state and new DecisionVLA design
- [ ] Native sampler/self-conditioning implementation and reference parity
- [ ] GPU0/1 batch64 capacity, resume and throughput checks before fresh training
- [ ] Original LIBERO and evaluation-only LIBERO Plus
- [ ] Final 5090 deployment validation and adapter-only experimental release

The proposed new training budget is rank32, global batch64 and100k updates. It will use a separate fresh run, not resume checkpoint46, and has not started. Convergence, success rate and completion time remain unmeasured.

## Repository map

| Path | Contents |
|---|---|
| [Architecture](docs/architecture.md) / [Development](docs/development.md) | Proposed method, exact changes and implementation gates |
| [Experiments](docs/experiments.md) | Completed and stopped runs with measurement boundaries |
| [`src/decisionvla/`](src/decisionvla/) | Existing policy, LoRA, training, confidence API and dashboard |
| [`benchmarks/`](benchmarks/) | 5090 quantization, fused execution, Graph and audits |
| [`evaluation/`](evaluation/) / [`reports/`](reports/) | LIBERO/Plus tooling and curated evidence |
| [Sources & licenses](docs/third-party.md) | DiffusionGemma, OpenJEV-family references and data attribution |

Original code is **MIT** licensed. Base models, datasets and dependencies retain separate terms; weights and datasets are not included. This is not an official Jev/TypeSafe or Google implementation.

# Experimental SM120 inference

These scripts operate on an inference copy of a saved adapter. They do not
modify the training model, optimizer, RNG, schedule, or checkpoint. The measured
experiment uses the 4,500-step adapter and is separate from the 100k training run.
See [measured evidence](../reports/inference-nvfp4-5090-20261025/).

## Tested runtime

Use a separate Linux/Python 3.12 inference environment. The training environment
in `pyproject.toml` is intentionally different; do not upgrade it in place.

| Component | Measured version |
|---|---|
| GPU | RTX 5090, SM120, 32 GB |
| Torch / torchvision / Triton | 2.14.1+cu130 / 0.29.1 / 3.8.0 |
| Transformers / FlashInfer | 5.20.2 / 0.7.2.post1 |
| CUTLASS DSL | 4.7.1 |
| NumPy / Pillow / safetensors | 1.26.4 / 13.0.0 / 0.8.0 |
| NVCC / NVRTC / CUDA runtime | 13.4.92 / 13.0.88 / 13.0.96 |

The measured image uses FlashInfer's CuTe DSL quantizer. Its C++ quantizer failed
with this mixed compiler/runtime installation and is not used. The patch below
adds GeGLU-tanh to the pinned W4A16 backend, with before/after source hashes. It
must be applied once to an isolated inference installation, before importing the
backend in a policy process:

```bash
python benchmarks/patches/flashinfer_geglu.py
export PYTHONPATH=src:benchmarks
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

These version details describe the tested image, not a claim that every fresh
installation or newer library version passes the numerical checks below.

## Export and calibration

Paths below are caller-supplied local assets. Outputs must be new directories.
Only experts are quantized; vision, attention, dense MLPs and routers stay BF16.
Expert LoRA is merged into the inference copy. Converted base weights, input
images and dataset files are private and must not be committed.

```bash
python benchmarks/export_nvfp4.py \
  --model models/base --adapter runs/checkpoint-004500 \
  --mode w4a16 --output exports/w4a16

python benchmarks/prepare_calibration_inputs.py \
  --data data --adapter runs/checkpoint-004500 --output inputs/calibration
python benchmarks/calibrate_nvfp4.py \
  --model models/base --adapter runs/checkpoint-004500 \
  --packed exports/w4a16/packed-experts --manifest exports/w4a16/report.json \
  --inputs inputs/calibration --output exports/calibration.json
python benchmarks/export_nvfp4.py \
  --model models/base --adapter runs/checkpoint-004500 \
  --mode w4a4 --calibration exports/calibration.json --output exports/w4a4
```

W4A16 uses E2M1 weights, E4M3 block16 scales and BF16 activations with fused
weight decoding/BF16 Tensor Core computation. It requires no activation
calibration. W4A4 uses native FP4 Tensor Cores and static activation scales from
32 training observations: per-layer amax × 1.1 / 2688. Neither path quantizes the
entire model to four bits. The native FP4 path pads the expert intermediate
dimension 704→768 with zeros and preserves scale-factor strides across export.

## Correctness and timing

`latency_nvfp4.py` expects ten frozen inputs in `samples.json` (file, instruction,
seed). Supply the original eager output archive through `--expected-fixed` to
check export/reload. It runs ten eager references, then warmup/capture and two
measured repeats per input, with exact action **and probability** comparisons.

```bash
python benchmarks/latency_nvfp4.py \
  --model models/base --adapter runs/checkpoint-004500 \
  --packed exports/w4a16/packed-experts --manifest exports/w4a16/report.json \
  --mode w4a16 --inputs inputs/fixed --expected-fixed results/eager.npz \
  --graph --prefix-graph --output results/w4a16-graph
```

Use a different output directory and W4A4 export for the native FP4 comparison.
Verify the GPU has no other compute process for standalone timings. Report
capture/compilation separately from warmed latency. Timing includes preprocessing,
four denoising passes, eight 7D actions, and full probability/confidence conversion;
it does not include camera capture, network transport or robot execution.

`stable_routes.py` replaces atomic routing order with fixed expert/route ordering;
the W4A16 path also disables atomic output accumulation. `prefix_graph.py` captures
the prefix language model and `decoder_graph.py` captures action inference, copying
prefix KV tensors once per observation. Vision and sampling RNG remain outside
graphs. The original 256-position canvas is retained.

`audit_nvfp4.py` independently checks fixed-input errors and paired rollout
artifacts, actions, distributions and success intervals. `replay_policy_observations.py`
replays saved simulator controls and verifies every query-image hash;
`replay_graph_queries.py` checks graph outputs against those original eager
trajectory queries. `audit_graph_replay.py` independently connects their output hashes to the original trace arrays and checks probability support/normalization and action decoding. Replay is a parity audit, not an additional success cohort.

Changing Torch, merging LoRA, quantizing and switching kernels all contribute to
the measured differences from BF16. Small fixed-input mean error is insufficient
to claim retained success. Raw confidence is not calibrated success probability.

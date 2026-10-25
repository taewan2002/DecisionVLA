# Adapter distribution

No model release is available. Training was intentionally checkpoint-stopped at46
while the native DecisionVLA architecture is redesigned. The repository refresh
publishes code, documentation and evidence only. The examples below package
**historical action-only adapters**; a future native adapter requires a new config,
loader contract, complete validation and model card. [Current plan](development.md).

GitHub Releases can host this project's adapters without a separate Hugging Face repository. GitHub currently allows each release asset to be **smaller than 2 GiB**; see [GitHub's release quotas](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

Only adapters, their configuration, model card and hashes are distributed. Base weights are downloaded from their upstream pinned revision. Datasets, local optimizer/RNG state, server operations and credentials are excluded.

The adapter contains LoRA matrices shared across encoder and decoder, plus action embedding deltas. The original configuration adapts attention; the extended configuration also adapts dense MLPs and every routed expert. Use DecisionVLA's loader; this is not a generic PEFT checkpoint. Merging must update each shared matrix only once and preserve the tied action input/output rows.

The extended rank-32 adapter contains 989,474,816 FP32 parameters (3,957,899,264 raw bytes), saved in four safetensors shards below 1 GB each plus an index. The loader accepts both the original single-file format and this sharded format. Create separate GitHub Release assets with:

```bash
python scripts/package_adapter.py runs/libero-spatial-r32-all-v2/checkpoint-002000 \
  --model-card path/to/MODEL_CARD.md --output release-assets/r32-all --sharded-release
```

For the final release, substitute the fixed100k checkpoint under the [forty-task batch64/100k plan](training-libero40.md), after its2,000 original and10,030 Plus episodes, deployment validation and audits. The output directory contains only shards, index, adapter config, model card and SHA hashes; detailed training metadata and optimizer/RNG state remain local. Upload these files together after final review. The packaging command itself only writes local files.

After validating a checkpoint, create a release asset with:

```bash
python scripts/package_adapter.py runs/libero-spatial/checkpoint-002000 \
  --model-card reports/libero-spatial-pilot/model-card.md \
  --output release-assets/decisionvla-libero-spatial.tar.gz
```

The packaging tool writes an archive and SHA256 manifest. It does not publish automatically. Release notes must state the checkpoint step, original base revision, dataset split, evaluated metrics, and missing evaluations. A preliminary adapter may be labeled experimental, but open-loop imitation metrics must not be described as robot success.

No adapter release has been published yet. Do not use the three-step smoke checkpoint as the first public policy.

## NVFP4 and calibration

The NVIDIA DiffusionGemma checkpoint quantizes selected expert linear weights **and activations** to NVFP4. Other modules remain at higher precision. Its config specifies static calibration and groups of 16, and its [model card](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) lists the calibration data. It is not W4A16.

W4A16 only names weight/activation precision. It does not identify the quantization algorithm or imply calibration-free conversion. Weight-only round-to-nearest can avoid activation data; activation-aware or reconstruction-based methods may need representative data. See the [AWQ paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) for a W4A16 method that uses calibration.

The activation-quantized W4A4 alternative follows this sequence. The current W4A16 candidate described below omits activation calibration:

1. Train LoRA using the BF16 base.
2. Merge every configured trained adapter and the action embedding deltas.
3. Calibrate the merged model on a dedicated subset of training robot observations and representative noisy action canvases.
4. Quantize supported layers with ModelOpt, keeping sensitive paths at higher precision as needed.
5. Compare actions and paired closed-loop rollouts against BF16, then measure an actual RTX 5090.

The existing NVIDIA quantized base was sufficient for the initial latency probe. It does not replace the final trained-policy calibration and validation. No claim is made that NVFP4 conversion alone is a new research contribution.

## Interim resident W4 experiment (2026-10-25)

The 4,500-update adapter was also tested with its frozen expert weights packed to uncalibrated symmetric group64 INT4. Original LoRA and other modules retain their types; expert weights are dequantized on GPU before BF16 GEMM. The measured RTX5090 policy latency was 0.959s p50/0.961s p95 without CPU weight offload, with 24.237GiB maximum sampled whole-device memory. This is an in-memory benchmark conversion, not a packaged NVFP4 policy. Action differences include a 1.20784 maximum on normalized `[-1,1]` controls, so robot success preservation is unverified. No quantized weights were published. See the [complete evidence](../reports/inference-5090-20261025/).

## Experimental trained-policy NVFP4 conversion — 2026-10-25

The frozen 4,500-step inference copy has separate expert-W4A16 and expert-W4A4 implementations. Expert LoRA is merged in this copy only; the original adapter, optimizer and training process are unchanged. The weight representation is E2M1 with E4M3 block16 scales and FP32 global scales. W4A16 retains BF16 activations and requires no activation dataset for weight packing. W4A4 uses fixed activation scales measured on 32 training observations; this is post-training quantization, not NVFP4 training.

The implementation pins FlashInfer 0.7.2.post1 and an isolated Torch 2.14.1/CUDA 13 inference runtime. A local GeGLU-tanh patch and deterministic W4A16 route packing are included under `benchmarks/`. Original BF16 measurements use Torch 2.9.1, so total action differences must not be attributed solely to quantization. Converted base/expert weights remain private; the final publication plan remains adapter-only.

Reproduction helpers are `prepare_calibration_inputs.py`, `export_nvfp4.py`, `calibrate_nvfp4.py`, `latency_nvfp4.py` and `audit_nvfp4.py` under `benchmarks/`. They are experimental SM120 benchmark tools, not a replacement for the production BF16 API. See the [current result and validation scope](../reports/inference-nvfp4-5090-20261025/). Final 100k conversion, rollout validation and release are still required.

The selected deployment candidate is W4A16, whose weight packing needs no
activation calibration dataset. Its final100k weights still require export/reload,
graph/numerical checks,2,000 paired original LIBERO episodes and RTX5090 measurement under
the [final deployment contract](final-deployment.md). Only W4A4 would require the
activation-calibration stage used in this experiment. The release remains adapter-only;
users obtain the base separately and run the provided conversion recipe locally.

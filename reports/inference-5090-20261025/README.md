# RTX 5090 resident-policy latency — 2026-10-25

The actual **4,500-update DecisionVLA adapter** runs on one RTX 5090 32GB without CPU weight offload after experimental weight-only INT4 packing of its frozen MoE experts. This is a latency and numerical-difference experiment, **not an NVFP4 export or evidence that quantized robot success is preserved**.

| Same RTX 5090, 4 denoising passes, 8 × 7 actions | Measured calls | p50 | p95 |
|---|---:|---:|---:|
| Original BF16 policy, CPU weight streaming / CUDA arithmetic | 10 | 17.364 s | 17.478 s |
| Resident mixed-precision W4 policy | 20 | **0.959 s** | **0.961 s** |

The p50 ratio is **18.10×**. These are two different precision/placement paths, so this ratio does not isolate the causal effect of offload removal. The earlier 15.659-second pilot preflight was on a different GPU with concurrent training and is not the denominator here. The 0.961-second p95 remains about 4.8 times the project's 200ms target. Eight predicted controls per request must not be interpreted as eight closed-loop observations or an equivalent feedback rate.

## Workload and memory

- Ten evenly spaced, preselected held-out demonstration observations from LIBERO-Spatial, with seeds 42–51. They are development observations, not a new independent test. The resident measurements repeat these ten inputs twice; all repeated actions and probabilities match exactly.
- One image, `visual_tokens=280`, native canvas 256, action horizon 8, 7 dimensions, four denoising passes. The original sampler, action vocabulary, confidence API, LoRA and cuBLAS arithmetic path are retained.
- Timing starts with an in-memory PIL image and instruction and ends with decoded actions and full confidence diagnostics on CPU. It includes preprocessing, vision/prefix and denoising. Model load, disk image decode, warmups, network, image acquisition and robot execution are excluded. Warmups: BF16 one; resident three.
- Frozen expert weights only: symmetric round-to-nearest INT4, group64, FP32 scale, no calibration. LoRA remains FP32 and is cast as in the original forward; the other modules retain their original dtypes. No LoRA merge. Expert tensors occupy 12,846,366,720 packed bytes instead of 45,675,970,560 BF16 bytes.
- Packed tensors stay on CUDA. Each expert projection is dequantized on GPU to BF16 before the original grouped cuBLAS GEMM. **This is a reference implementation, not fused low-bit serving.** All parameters and buffers are verified on CUDA; no CPU weight streaming hook is installed in the successful resident process.
- Resident tensors: **21.212GiB**. Torch peak allocated: **22.500GiB**; reserved: **22.658GiB**. Maximum sampled whole-device memory: **24.237GiB**, including the desktop and CUDA runtime. Device telemetry is sampled every two seconds and may miss short transients.
- Actual RTX5090, driver 595.99.02, 600W existing power limit, Torch 2.9.1+cu130, Transformers 5.20.2. The benchmark runtime uses NumPy1.26.4 (the training project pins2.5.4); both compared paths use the same benchmark runtime. See [exact versions and image ID](environment.json). One request at a time. GPU1 mining and other workloads remain unchanged; the small desktop process on GPU0 is retained.

## Numerical differences and limitations

| Ten paired observations, same seed and original sampler | Value |
|---|---:|
| Mean absolute difference in continuous normalized controls `[-1, 1]` | **0.03966** |
| Largest absolute difference in any control | **1.20784** |
| Gripper agreement | 80/80 (100%) |
| Mean total variation of final token distributions | **0.32091** |

The differences are not negligible. Four-pass stochastic trajectories can diverge after a small quantization perturbation even with the same initial seed; these numbers measure final-policy differences, not an isolated one-layer quantization error. The two earlier successful BF16 rollouts **do not validate this W4 policy**. A paired quantized rollout evaluation is still required before deployment or a quality-preservation claim. Raw confidence remains uncalibrated, and changed distributions may change confidence behavior.

The independent CPU audit recomputed timing statistics, checked 30 requests / 1,680 distributions, normalization, allowed gripper support, action/argmax alignment, repeat equality and raw artifact SHA256. GPU unpacking matches an independent Torch quantization/dequantization reference exactly. These checks validate records and implementation consistency, not robot success.

## Recovery and resource preservation

The first attempt completed all ten BF16 measurements, then stopped before scored resident timings because an overly broad assertion rejected every forward hook. A CPU-only inspection identified the original Transformers `enable_input_require_grads` hook, which does not offload weights. The successful process preserved original hooks by identity, asserted every parameter/buffer is CUDA, and never installed weight-streaming hooks. The original BF16 arrays were reused with recorded SHA256; failure logs remain preserved privately.

GPU0 mining was temporarily stopped and restored; the temporary disable marker was removed. Training on the separate host continued, with all monitored training/serving process identities unchanged; the post-test snapshot was 4,854/100,000. The final 100k/500-episode evaluation contract remains unchanged. No commit, push, model export or release was performed.

## Evidence and reproduction

- [Full report](report.json), [independent audit](audit.json), [completion checks](completion.json).
- [BF16 timings](bf16_stream.json), [resident timings](resident_w4.json), [fixed observations](observations.json), [predeclared plan](plan.json).
- [Input file hashes](input-files-sha256.json), [executed script hashes](benchmark-source-sha256.json).
- Raw action/probability NPZs, host telemetry and process identities remain in the ignored `private/` folder. Base, dataset, adapter tensors and optimizer are not included here.
- Public scripts: [resident INT4 reference](../../benchmarks/resident_w4.py), [policy benchmark](../../benchmarks/latency_policy_w4.py), [CPU audit](../../benchmarks/audit_policy_w4.py), [input preparation](../../benchmarks/prepare_policy_inputs.py).

Run the benchmark in the pinned Torch/Transformers/CUDA13 environment with a C compiler and an SM120 GPU; match [environment.json](environment.json) for the measured runtime. Mount the repository at `/source`, the original base at `/base`, the matching adapter at `/adapter`, prepared images and `samples.json` at `/inputs`, and a fresh writable directory at `/output`. Set `PYTHONPATH=/source/src`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`, `XDG_CACHE_HOME=/output/cache` and `TRITON_CACHE_DIR=/output/triton-cache`. Run `python /source/benchmarks/latency_policy_w4.py`, followed by `python /source/benchmarks/audit_policy_w4.py`. The script loads the full BF16 model on host RAM before conversion, so provision approximately 100GiB host memory. It caps its CUDA allocator at 29GiB and does not control miners or other jobs. GPU isolation must be arranged before running it.

To reuse an already-completed BF16 comparison, mount its output read-only at `/reference` and set `DECISIONVLA_BF16_REFERENCE=/reference`. Match the base, adapter, source and observation hashes first; the script records reference file hashes and checks workload metadata. The successful run used this option after the documented assertion failure. Its public benchmark, packing helper and audit files match the executed source byte-for-byte.

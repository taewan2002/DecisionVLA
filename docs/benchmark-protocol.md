# Latency protocol

This is a base-model workload proxy, not a trained VLA evaluation.

- One GPU, one request at a time, no CPU weight offload.
- Disable asynchronous scheduling for this single-request policy workload. The initial asynchronous probe scheduled an additional 256-token pass beyond the requested canvas; it is retained as a failed workload-validation attempt rather than silently relabeled.
- Native 256-token canvas. Return 56 text token IDs to match an 8×7 action shape.
- 1 or 2 deterministic synthetic RGB images (256×256), 280 visual-token budget per image.
- Every request uses new image bytes; prefix caching and processor caching are disabled.
- Fixed denoising counts 1, 2, 4, 8; confidence stopping disabled.
- Five warmups then thirty timed requests per case. New step values may compile during warmup.
- Timing starts before chat-template processing and ends after synchronous `LLM.generate` returns IDs. It includes image preprocessing, image encoding, prefix prefill, denoising, canvas commit and local engine overhead.
- Image synthesis/acquisition, remote network transport, robot control and trained action decoding are excluded.
- The engine execution counter must equal `(warmups + repeats) × (steps + 2)`: one prefill, S denoise passes and one commit. This validates the actual fixed-step workload.
- The scheduler token histogram must contain `(warmups + repeats) × (steps + 1)` passes of 256 tokens, verifying that the 56-token return limit did not shorten the denoise/commit canvas.
- Use BF16 KV cache explicitly for both weight formats. NVFP4 refers to the checkpoint's quantized weights/activations; this is mixed precision, not every tensor in 4-bit.
- Record raw latency samples, input/output hashes, actual prompt lengths, allocator peaks, whole-device memory telemetry, engine settings and pinned model/container versions.
- Report p50/p95 using NumPy's default linear percentile estimator. Thirty requests are a short latency probe, not a production tail-latency guarantee.

Synthetic input content and untrained outputs can route different MoE experts
than trained robot actions. These results establish runtime feasibility and
workload costs; they do not establish final VLA latency, success, or 5090 speed.

Example after downloading the pinned local checkpoint and ensuring the selected
GPU is available:

```bash
docker run --rm --gpus device=0 --cpus 8 --memory 112g --shm-size 4g \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=8 \
  -v "$PWD/benchmarks:/workspace:ro" \
  -v /absolute/path/to/checkpoint:/model:ro \
  -v "$PWD/reports/run:/output" \
  --entrypoint python3 \
  vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
  /workspace/run_patched_vllm.py /workspace/latency_vllm.py \
  --model /model --output /output
```

The default memory fraction 0.30 is for the **96GB measurement GPU**, not a
portable 5090 launch setting. KV cache is explicitly 512MiB; the fraction is not
a hard 32GB physical-memory emulation. On a 5090, choose the fraction according
to available memory and verify startup/warmup/steady peaks on that device.
Retain the resolved quantization and KV-cache dtypes from the engine log.

The wrapper applies a one-attribute compatibility fix to the disposable
container's vLLM 0.30.0 source: DiffusionGemma delegates image processing to
Gemma4, which expects `_enable_mm_lora`; the DiffusionGemma constructor omits
that attribute. This no-LoRA benchmark sets it to `False` and records the
before/after source hashes. This is a locally patched runtime, not an
unmodified upstream performance result. No mathematical operation or weight
is changed by the patch. It must be reviewed before using a different version
or unmerged adapters.

Only this repository's private operations helper knows the measurement host or
miner lifecycle. It is excluded from Git. The public benchmark does not stop
other processes or change GPU power settings.

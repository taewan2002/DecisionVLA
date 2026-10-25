# Inference benchmarks

These tools reproduce experiments for the **earlier action-only policy** and
untrained base proxy. The proposed native denoising/sampler path must pass new
reference and deployment checks before inheriting any optimization or claim.

- [NVFP4 reproduction](NVFP4.md): expert W4A16/W4A4 export, fused SM120 execution,
  prefix/decoder CUDA Graph, reload and numerical audits.
- [Measured5090 pilot](../reports/inference-nvfp4-5090-20261025/final/):4500-step
  checkpoint, four passes,20 paired robot episodes;188.43ms W4A16 Graph p50.
- `latency_policy_w4.py` / `resident_w4.py`: earlier group64 INT4-dequantization baseline.
- `latency_vllm.py`: untrained base-model proxy, not the VLA action sampler.
- `audit_results.py`: independent recomputation of the base-proxy report.

The optimized inference runtime is isolated from the BF16 training environment.
`patches/flashinfer_geglu.py` intentionally patches its pinned dependency only in
that isolated runtime; do not run it against an unrelated Python installation.
No benchmark imports or commands are a background training launcher.

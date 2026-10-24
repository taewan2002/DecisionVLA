# Adapter distribution

GitHub Releases can host this project's adapters without a separate Hugging Face repository. GitHub currently allows each release asset to be **smaller than 2 GiB**; see [GitHub's release quotas](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

Only small adapters, their configuration, model card and hashes are distributed. Base weights are downloaded from their upstream pinned revision. Datasets, local optimizer/RNG state, server operations and credentials are excluded.

The adapter contains attention LoRA A/B matrices shared across encoder and decoder, plus action embedding deltas. Use DecisionVLA's loader; this is not a generic PEFT checkpoint. Merging must update each shared matrix only once and preserve the tied action input/output rows.

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

W4A16 only names weight/activation precision. It does not identify the quantization algorithm or imply calibration-free conversion. Weight-only round-to-nearest can avoid activation data; activation-aware or reconstruction-based methods may need representative data.

The intended deployment sequence is:

1. Train LoRA using the BF16 base.
2. Merge the trained attention adapters and action embedding deltas.
3. Calibrate the merged model on a dedicated subset of training robot observations and representative noisy action canvases.
4. Quantize supported layers with ModelOpt, keeping sensitive paths at higher precision as needed.
5. Compare actions and paired closed-loop rollouts against BF16, then measure an actual RTX 5090.

The existing NVIDIA quantized base was sufficient for the initial latency probe. It does not replace the final trained-policy calibration and validation. No claim is made that NVFP4 conversion alone is a new research contribution.

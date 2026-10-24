# First LIBERO-Spatial pilot

Status: initial training launched on 2026-10-24; ongoing. This file will be updated with an observed progress snapshot before publication.

The original 500 demonstrations were independently downloaded and checked against upstream SHA256. Across all 61,750 available adjacent states, post-action observation joint coordinates matched the following simulator state exactly. Training and validation episodes are disjoint (450 / 50).

The real-GPU smoke run passed image-conditioned backward, a nonzero observation-prefix gradient, adapter save/reload and a numerical resume check. Peak PyTorch allocation was 50.70 GiB. Single-example optimizer updates after warmup took about 0.4 seconds.

The resume loss matched, but repeated backward was not bitwise identical: maximum adapter parameter difference was `4.4248990889173e-6` under the declared `1e-5` tolerance. This is recorded rather than described as exact replay.

A disposable 24-update overfit check reduced fixed-image/fixed-noise action CE from **7.19 to 1.18**. This verifies that the learning path can fit a sample. It is not a robot success result, generalization result, or released policy.

Main run: one RTX PRO 6000 Blackwell 96GB, rank-16 shared attention LoRA, trainable action rows, effective batch 8, 2,000 updates, checkpoint/validation every 100 updates, 12-hour training limit. The main run starts from the original base and newly initialized adapters; smoke/overfit updates are discarded.

No closed-loop rollout success, trained-policy NVFP4 result or RTX 5090 result has been measured. Adapter release is pending validation.

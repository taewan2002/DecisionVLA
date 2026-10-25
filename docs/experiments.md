# Experiment ledger

All completed measurements below belong to the **earlier action-only sampler**.
The proposed DecisionVLA native sampler has no trained checkpoint, success result
or latency measurement yet. This ledger retains unsuccessful and stopped runs.

| Experiment | Outcome / scope | Evidence |
|---|---|---|
| Base latency probe | Untrained proxy; not a trained robot policy | [Report](../reports/base-latency-2026-10-24/report.ko.md) |
| Rank32 /2k Spatial |14/500 success; within-task failure AUROC0.431,95% CI[0.287,0.590] | [Confidence evaluation](../reports/libero-spatial-confidence/) |
| Denoising diagnosis | Confidence increased while action error and copying also increased; motivated objective investigation | [Diagnostic records](../reports/denoising-correction/) |
| Training throughput | Audited cuBLAS forward / SM120 backward changes, with numerical and resume checks | [Kernel records](../reports/training-throughput/) |
| Rank32 /4500 Spatial,5090 | W4A16 Graph188.43ms p50,18.815GiB peak allocation; BF16/W4A16/W4A4 success12/11/9 out of20 each | [Final pilot report](../reports/inference-nvfp4-5090-20261025/final/) |
| Spatial batch64 /100k plan | Intentionally stopped at6,681; checkpoint and history preserved | [Scope correction](libero-40-scope-correction.md) |
| Fresh40-task batch64 /100k plan | Started from base, then intentionally stopped at46 to redesign the method; no final rollout | [Startup](../reports/training-libero40-20261025/) · [Stop audit](../reports/openjev-redesign-20261025/stop-audit.json) |
| Native DecisionVLA proposal | Read-only source review and tokenizer feasibility audit; no training or model inference | [Design evidence](../reports/openjev-redesign-20261025/) |

The5090 pilot used a fixed4500-step checkpoint and20 paired development initial
states. W4A16−BF16 success difference was−5percentage points,95% CI[−35,+25].
Graph execution reproduced863 candidate observations including confidence;
48,328 distributions were independently checked. These repeated observations
are not additional independent success trials. W4A16 expert packing is
post-training quantization, not NVFP4 training. Original BF16 and optimized
runtime versions also differ; total action error cannot be attributed only to
quantization. See the original reports for exact workload and timing exclusions.

## Historical procedures

[Initial pilot](training.md), [50k continuation](training-50k.md),
[batch64 plan](training-b64-100k.md) and [stopped forty-task plan](training-libero40.md)
preserve earlier implementation and configuration contracts. They do not authorize
restarting those runs. [Current development plan](development.md) supersedes their
active execution status. OMY simulator preparation is deferred; scripted results
are not model zero-shot results and are outside the current LIBERO release scope.

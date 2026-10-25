# Evidence index

Reports describe their own frozen model, runtime, dataset and protocol. A startup
snapshot is not live status, a preflight is not a success benchmark, and repeated
policy queries are not independent robot episodes.

| Start here | What it establishes |
|---|---|
| [5090 W4A16 / W4A4 pilot](inference-nvfp4-5090-20261025/final/) | Measured4500-step latency, memory, action differences and20 paired outcomes |
| [2k confidence evaluation](libero-spatial-confidence/) |500 Spatial episodes and within-task failure association |
| [Denoising diagnosis](denoising-correction/) | Recorded confidence/error/copying behavior |
| [Training numerical and throughput checks](training-throughput/) | Kernel comparisons and resume checks under their original runtimes |
| [Forty-task startup](training-libero40-20261025/) | Data audit, disposable resume check and startup; later stopped at46 |
| [Preservation and design review](openjev-redesign-20261025/) | Verified46-step stop, inspected upstream revisions and tokenizer-only feasibility |
| [Untrained base latency](base-latency-2026-10-24/report.ko.md) | Proxy inference cost, not trained VLA quality or5090 policy speed |

See [experiments](../docs/experiments.md) for the interpretation and chronology.
Native DecisionVLA training and performance remain unmeasured. Reports contain
curated aggregates and provenance; base weights, datasets, optimizer/RNG and
private operational records are excluded.

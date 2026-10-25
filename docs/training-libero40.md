# Fresh forty-task DecisionVLA training

> **Historical procedure — superseded on2026-10-25.** Training was checkpoint-stopped; the forty-task run ended at46 and its automatic final evaluation did not start. These commands describe the earlier action-only implementation. See the [current DecisionVLA design](architecture.md) and [development plan](development.md); do not restart old controllers.

The owner authorized replacement of the Spatial-only experiment on2026-10-25.
The old run checkpoint-stopped at6,681 and retains its adapter, optimizer,
all-rank RNG, logs and frozen source. The new run is
`libero40-r32-b64-100k-v1`: fresh rank32 LoRA and action embeddings initialized
from the pinned pretrained DiffusionGemma base, with a fresh optimizer/RNG.
No Spatial pilot or disposable smoke weights initialize production.

## Fixed scope

| Item | Setting |
| --- | --- |
| Training | Original Spatial, Object, Goal, Long (`libero_10`), ten tasks each |
| Demonstrations |2,000 total:1,800 train /200 validation; demo IDs ending in9 held out |
| Sampling | Uniform task, then uniform valid observation within that task; with replacement |
| GPUs / global batch | Two RTX PRO6000 Blackwell; micro4 × accumulation8 ×2 =64 |
| Budget |100,000 optimizer updates;6.4million observations total |
| LR | Warmup1,000 updates from5e-6 to5e-5, cosine to5e-6 at100k |
| Objective | Valid randomly resampled positions, including chance matches |
| Validation | Every500 updates; same200 observations, five per task; suite metrics reported |
| Checkpoints | Every500; retain all10k boundaries and latest two recovery points |
| Selection | Fixed100k, no checkpoint selection from rollout outcomes |
| Original evaluation |40 tasks ×50initial states =2,000 episodes |
| Plus evaluation |10,030 official variants ×1 trial; no Plus training |

Long tasks have more frames, so uniformly sampling pooled frames would overweight
them. Task-first sampling makes expected task and suite weights equal. This is a
single mixed policy; comparisons to four separately trained policies must state
the different training budget.100k is the declared budget, not proven convergence.

The accepted cuBLAS forward and SM120 Triton backward remain in use. The fresh
three-update versus2+1 restart gate passed bitwise adapter/optimizer comparison
and exact all-rank RNG. The40-file data audit passed336,575 observation/action
alignment checks with zero joint error. All2,000 demonstrations were checked;
train/validation episode sets are disjoint. These are implementation checks, not
robot success measurements.

Training runs in ten10k process segments with immediate full-state resume and no
intermediate rollout. After a segment, the private controller records manifests
for intermediate recovery checkpoints before removing them according to the
retention rule above. Existing Spatial and smoke evidence is preserved.
The retention rule bounds disk use; it does not select checkpoints by performance.

## Reproduce the training configuration

Use the pinned dependencies and a licensed local copy of the base model. Fetch
and audit all four original suites before launching:

```bash
python scripts/download_libero.py --output data-libero40 \
  --suites libero_spatial libero_object libero_goal libero_10
python scripts/audit_data.py --data data-libero40 \
  --config configs/libero_40_r32_b64_100k.json \
  --output data-libero40/audit.json
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 \
  --max-restarts=0 -m decisionvla.train_distributed \
  --model /path/to/base --data data-libero40 \
  --config configs/libero_40_r32_b64_100k.json \
  --output runs/libero40-r32-b64-100k --stop-at-step 10000
```

Resume the same output/config at each boundary with
`--resume runs/libero40-r32-b64-100k/checkpoint-010000 --stop-at-step 20000`,
continuing through100k. The configured total budget stays100k throughout;
segmentation preserves the LR schedule, optimizer and all-rank RNG. Production
uses the audited private controller for these consecutive segments and the final
evaluation; the commands above alone do not create a background controller.

## Final evaluation

The controller invokes the pinned evaluator automatically after100k training.
Two resident BF16 policy processes use the same final checkpoint on GPU0/1;
CPU simulators own disjoint whole tasks. Original suite horizons are220/280/300/
520 controls, plus10settling controls. Plus uses the same respective horizons,
128RGB, four denoising passes and eight-action execution. This declares our
protocol; it is not an exact reproduction of models using256RGB, wrist images
or proprioception. The semantic BDDL instruction is used, excluding virtual
filename perturbation identifiers. Upstream image corruptions remain active.

Every task stores contract/source/initial-state hashes, complete action/state/
probability traces and an independent probability/action audit. Final collection
recomputes these audits, verifies the exact expected episode keys and creates
report.json and episodes.csv. Original videos are retained privately. An error
stops the chain and preserves partial results; incomplete original tasks are not
resumed by skipping previously executed initial states.

Success is reported overall, by suite and by Plus perturbation category.
Confidence/failure AUROC uses first-query uncertainty and within-original-task
comparisons across40 groups with95% bootstrap intervals. A Plus variant has one
trial, so its own AUROC is undefined. Single-class groups remain undefined.
Raw confidence is not a calibrated success probability or causal explanation.
Some original Spatial initial states have prior development use; no untouched
holdout-test claim is made.

The automatic chain here is **BF16 reference evaluation**. Final expert NVFP4
W4A16 export, eager/Graph parity, paired deployment-quality validation and
RTX5090 measurements remain additional publication gates. Existing4500-step
inference numbers do not validate these new weights. Publish one combined
code/report/figure commit and adapter-only experimental Release after those
checks and review. No base/dataset/optimizer/private operations are release assets.
OMY and other robot arms are excluded.

Configuration: `configs/libero_40_r32_b64_100k.json` and
`configs/eval_libero40_plus_100k.json`. Original task/data revisions and Plus
code/assets revisions are pinned there. Current execution status belongs in the
run artifacts, not inferred from this document.

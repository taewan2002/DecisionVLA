# Four-suite scope correction — 2026-10-25

**Historical preparation record.** The owner subsequently authorized the replacement.
The Spatial run stopped at6,681; fresh forty-task training and final LIBERO/Plus
evaluation are now connected. See the [current contract](training-libero40.md).
The remaining text records the earlier15:27KST preparation snapshot.

The running `libero-spatial-r32-b64-100k-finalonly-v1` trains **Spatial's ten
tasks only**. At15:27KST it had6,624 updates and checkpoint6,500. Its remote
dataset manifest contains ten Spatial files. Earlier descriptions of this as
simply "LIBERO training" did not make the limitation sufficiently clear.

The intended publication covers Spatial, Object, Goal and Long (`libero_10`),
forty original tasks, plus LIBERO Plus robustness evaluation. LIBERO-90 and OMY
are outside this scope. Existing Spatial results remain Spatial evidence.

## Preparation; replacement not launched

Local checks: eighteen focused data/schedule tests and Ruff passed. All forty
candidate task names and language prompts match the pinned official task map.
A broader continuation test run had22 passes and one import failure in the
host's pre-existing SciPy macOS binary; that DDP checkpoint test did not execute.
No new GPU gradient/resume result is claimed.

`configs/libero_40_r32_b64_100k_candidate.json` specifies the exact forty task
names from original LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
All forty exist in dataset revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`
(33,784,856,577 bytes). The loader rejects missing suites, task-list mismatches
and incomplete declared demonstrations. It handles Long scene prefixes and
suite-qualified sample identities. Spatial keys/manifests retain compatibility.

```bash
python scripts/download_libero.py --output data-libero40 \
  --suites libero_spatial libero_object libero_goal libero_10
python scripts/audit_data.py --data data-libero40 \
  --config configs/libero_40_r32_b64_100k_candidate.json \
  --output data-libero40/audit.json
```

The candidate retains rank32, batch64, SM120 kernels, resampling-only loss and
the100k LR schedule. Its current sampler is uniform over pooled frames, **not
balanced by task**. Finalize sampling and suite-stratified fixed validation
before production. Fresh LoRA/optimizer/RNG from the pinned base is the proposed
release baseline; continuing Spatial weights would be a different, explicitly
two-stage experiment. Ordinary resume correctly rejects data-config changes.

100k ×64 remains6.4million observation presentations. Mixed training reduces
per-task exposure relative to the Spatial run; neither adequate convergence
nor equal cost to four separate suite-specific models is established. There
is no automatic400k increase. Recheck mixed-data throughput.

Before replacement: checkpoint-stop the owned Spatial run and preserve all
state; verify new data SHA/splits/action alignment/language; pass real gradient
and continuous-versus-resumed checks on GPU0/1. No remote changes have been
made by this preparation, and the original run remains active.

The intended original evaluation covers40tasks ×50inits =2,000 episodes per
precision; declare suite-specific horizons instead of copying Spatial's220.
Plus covers10,030 variants, one trial each, with suite/perturbation reporting.
Keep Plus out of this training dataset. Existing NVFP4/Graph code is reusable,
but numerical and success validation must be repeated on final weights.

Spatial-only completion cannot satisfy this expanded publication gate. No
replacement training, expanded rollout, automation, commit, push or Release
has occurred. The earlier deployment contract is historical pending replacement.

Sources: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
[pinned task map](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/libero_suite_task_map.py),
[LIBERO Plus](https://github.com/sylvestf/LIBERO-plus).

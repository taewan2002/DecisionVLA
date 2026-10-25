# DecisionVLA contributor context

- This is an independent project. Preserve all other projects, jobs and datasets.
- Answer the owner in Korean; use English for code and API names.
- **Current state (2026-10-25): training intentionally stopped.** The forty-task run `libero40-r32-b64-100k-v1` saved `checkpoint-000046`; all eight checkpoint payload hashes were verified. Its early-boundary controller error is expected. Do not recover or restart that controller or its final evaluator. The earlier Spatial run remains preserved at step6,681.
- The owner requested a new DecisionVLA architecture preserving the reference denoiser, entropy-bound sampler and self-conditioning. The separate `native_*` reference and typed readout are now implemented with tiny-model CPU checks; see `docs/native-reference.md` and `docs/typed-decisions.md`. Full26B GPU validation, real-data learning, DDP capacity/resume and training remain pending. Historical `model.py` and measurements use the earlier action-only sampler. Native tests require isolated `requirements-native.txt`; do not replace the pilot runtime.
- The owner subsequently authorized a repository/README refresh **before** training resumes. This supersedes the earlier hold on intermediate repository publication. It does not authorize a weight release or a new training launch in this documentation task.
- Keep the project name **DecisionVLA**. Name OpenJEV implementations in technical comparisons and attribution, not as the project's brand.
- Public descriptions should not label it a personal project. Keep architecture figures focused on data flow; put implementation status and caveats in the accompanying prose, not corner badges or footers.
- Preserve frozen run source, adapters, optimizer/RNG, protocols and logs. New architecture work uses a separate configuration and fresh run; never silently change or relabel old evidence.
- Public scope is original LIBERO forty tasks and evaluation-only LIBERO Plus. OMY is deferred and is not a publication gate. No new recurring automation.
- Use only owner-authorized GPUs. Other training/serving jobs remain untouched. GPU1 mining must remain stopped; do not restore it automatically. Check live ownership before later GPU work.
- Never present raw confidence as task-success probability or causal explanation. A native sampler does not establish speed or robot success. The existing188ms/18.815GiB measurement belongs only to the old4500-step RTX5090 W4A16 Graph pilot.
- Keep credentials, private operations, datasets, base weights, optimizer/RNG, raw sessions and model assets out of Git. Publish only reviewed source, documentation and curated evidence. A later adapter-only experimental release needs its own completed validation.
- CPU checks: `pytest -q`; lint: `ruff check src tests scripts benchmarks`. GPU and optional simulator tests may skip in CPU-only environments; report their scope honestly. Keep upstream licenses separate from this repository's MIT license.

## Typed decision API publication — 2026-10-25

- User requests one combined code/docs/dashboard/native commit and push before training. No new training, GPU execution or schedule is authorized by this repository-refresh step.
- Canonical interface: predicate (noul alias), choice, score; questions condition the native prefix, label slots and future training preparation. Robot profile libero.actions.v1 is 48 choices + 8 command predicates. Custom questions require explicit experimental opt-in and emit no robot controls.
- Architecture tag decisionvla.native-decisions.v1 rejects older layouts. Confidence is entropy concentration, not OpenAI's proprietary formula or calibrated success probability. No trained native decision checkpoint exists.
- Deferred OMY files remain local and excluded from this publication. Preserve historical pilot sources/results and stopped checkpoints.

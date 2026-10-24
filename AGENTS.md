# Project scope

- This is an independent personal project. Do not import or modify the thesis/RetoVLA project, its jobs, checkpoints, evaluation results, or schedules.
- Answer the owner in Korean. Use English for public-facing code and API names.
- Current scope: implement and publish DecisionVLA, and start a LIBERO-Spatial LoRA pilot on the owner-authorized GPU. Track actual run status in reports; no closed-loop success or 5090 measurement has been established.
- Keep measured results, implementation targets, and hypotheses distinct. RTX PRO 6000 timings are not RTX 5090 measurements.
- Use the exact revisions and workload boundaries in configs/reports. Never relabel synthetic images or base text tokens as validated robot inputs/actions.
- Follow the user's current execution scope. The owner authorized this initial training run and publication to taewan2002/DecisionVLA. Do not silently expand to other GPUs or projects.
- Keep private server operations, credentials, datasets and weights out of Git. Respect each upstream license separately.
- Before GPU use, inspect live jobs. Isolate authorized GPUs and automatically restore any temporary resource changes.

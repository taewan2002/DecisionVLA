# Repository refresh checks

The owner requested a DecisionVLA README/repository refresh before another training
launch. This publishes the existing pilot implementation, curated measurements and
a separately labeled proposed architecture. No training or model release was started.

The staged public snapshot passed80 CPU tests (26 GPU-only skips), Ruff, relative-link
and JSON checks, source/payload review, a wheel build and the existing base-latency
independent recomputation. The wheel includes dashboard assets and the native C
source; it excludes weights and deferred OMY modules. Details and source hashes are
in [qa.json](qa.json). These checks do not establish the proposed model parity,
GPU kernel equivalence, task success or speed.

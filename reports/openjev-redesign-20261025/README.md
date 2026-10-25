# DecisionVLA redesign evidence

- `stop-audit.json`: owner-requested stop at46; eight complete checkpoint payload
  hashes; old frozen source unchanged; protected jobs unchanged; no final rollout.
- `canvas-audit.json`: tokenizer-only direct-token JSON feasibility;56slots,
  243template tokens,13EOS fill positions and14,336 slot/label checks. Unused
  token spellings do not re-encode to the inserted IDs; use direct-ID construction.
- `upstream-sources.json`: seven pinned primary source repositories, including
  five OpenJEV-family implementations and the native HF/Google implementations.

These artifacts establish preservation and design feasibility, not model parity,
learning quality, calibrated confidence or speed. All weights, optimizer/RNG,
base files and private operations remain outside Git. The repository refresh is
separately authorized before training; no new training or model release occurred.

[Architecture](../../docs/architecture.md) · [Development plan](../../docs/development.md).

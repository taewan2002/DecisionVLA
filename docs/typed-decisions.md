# Typed decisions

DecisionVLA follows the interface principle of the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions): shared evidence plus typed questions produce ordered answers read from allowed-label probabilities. It uses its own DiffusionGemma backend and probability definitions. It is not an OpenAI client, a wire-compatible replacement, or a reproduction of proprietary Jev training.

## From a question to a decision

1. Validate each question's type, instructions, name and allowed options.
2. Include the question semantics and label bindings in the image/task prefix. Repeated option legends are shared. The fixed robot profile has an exact compact arithmetic description, rather than repeating 256 choices 48 times.
3. Compile one single-token answer slot per question into a 256-token canvas. Native denoising, the entropy-bound sampler and soft self-conditioning retain the full vocabulary at variable slots.
4. Read the final softcapped logits **before sampler temperature**, normalize over that question's allowed labels, and return its typed answer. No generated text is parsed.
5. For the declared robot profile only, map the answers to executable controls. The same prefix and label mapping prepare future training examples.

Changing custom instructions or options changes the model prefix and schema hash. These are actual question-conditioned slots, not arbitrary names attached to an existing action distribution. The schema, canvas, checkpoint and generation settings are recorded with each prediction.

## Contract

| Type | Question fields | Answer |
|---|---|---|
| `predicate` | `instructions`, optional `name` | `probability = P(true)`; label order is true, false |
| `choice` | `instructions`, 2–256 `choices: [{value, description?}]` | `choice`, complete `probabilities`, `confidence` |
| `score` | `instructions`, 2–10 ordered `levels: [{label, description?}]` | `score = Σ i·pᵢ`, complete level probabilities, `confidence` |

There are 1–56 questions per request. Names, when supplied, must be unique. Choice values are distinct strings or booleans; boolean `true` and string `"true"` remain distinct. Answers preserve request order. A score uses indices starting at zero and may lie between levels. `noul` is accepted as a spelling alias for `predicate`; responses use `predicate`.

These are DecisionVLA's limits. OpenAI's service has additional input types, refusal behavior, caching and other capabilities that this local service does not implement. Unknown fields and malformed requests fail explicitly. A protocol-valid answer is not a guarantee of a correct action.

## Robot profile

`libero.actions.v1` defines 56 decisions in timestep-major order:

- Six `choice` questions per timestep: `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, each selecting bin 0–255. Its normalized control is `-1 + 2*i/255`.
- One `predicate`: whether the gripper command is +1 rather than −1. It is **not** whether a grasp succeeded.

Execution uses each continuous mode; the gripper is +1 when `P(true) > 0.5`, otherwise −1. Ties choose the first continuous bin and the negative gripper command. The control mean is not substituted for the mode. No artificial task-progress score is derived from the control bins. A genuine progress/phase/grasp question requires appropriate labels and validation; LIBERO behavior cloning alone does not supply that capability.

The 56 distributions are conditional marginals from a joint denoising canvas. Multiplying them does not give an established joint trajectory probability.

## Python and local service

The interface is implemented, but there is **no trained native decision checkpoint yet**. The following is usage for a future verified checkpoint, not a runnable pretrained-model demo. Use the isolated [native runtime](native-reference.md).

```bash
PYTHONPATH=src .venv-native/bin/python -m decisionvla.native_serve \
  --adapter /absolute/path/to/native-checkpoint --socket /exchange/native.sock
```

```python
from PIL import Image
from decisionvla.decision_client import DecisionClient

client = DecisionClient("/exchange/native.sock")
# Already transformed to the documented 128x128 LIBERO policy convention.
image = Image.open("policy-observation.png").convert("RGB")
decision = client.create(image, "put the bowl on the plate", steps=2, seed=42)
actions = decision["actions"]          # 8 x 7, from the same model call
answers = decision["answers"]          # 48 choice + 8 predicate, ordered
print(answers[0]["choice"], answers[0]["probabilities"])
print(answers[6]["probability"])        # positive gripper command probability
print(decision["diagnostics"]["allowed_mass"])
```

For an in-process caller, `NativeDecisionVLA.decide(image, instruction, ...)` returns `decisions`, `actions`, `diagnostics` and `timing`. The socket client flattens the `decisions` envelope into its reply. The evaluator's `predict_with_confidence()` keeps the existing action/diagnostics contract; `typed=True` also returns the decision envelope. Formatting answers adds CPU and serialization cost, but no model forward pass.

Custom question schemas go through the same compiler and native model. They require **`experimental=True`**, return `semantic_validation: unvalidated_custom_questions`, and never return robot actions. For example:

```python
questions = [{
    "type": "score", "name": "height", "instructions": "Rate the object's height above the table.",
    "levels": [
        {"label": "On table", "description": "No visible separation."},
        {"label": "Raised", "description": "Clear separation above the table."},
    ],
}]
# Exploratory mechanics only: no semantic accuracy or calibration has been established.
result = client.create(image, "inspect the bowl", questions=questions, experimental=True)
```

The local transport is a length-prefixed JSON Unix socket, owner-only permissions, with a 2 MiB message limit. There is no public HTTP `/v1/decisions` endpoint, hosted image fetch, OpenAI key, or cloud dependency. Invalid requests return errors rather than fabricated answers. Custom semantics are not silently promoted to a trained feature.

## Confidence and evidence

For choices/scores, DecisionVLA defines `confidence = 1 − H(p)/log(K)`. This follows an inspected open-source entropy summary; it is **not a claim about OpenAI's confidence formula**. It measures concentration, not correctness or episode success. Every decision envelope declares `calibrated: false`.

`allowed_mass` and `allowed_log_mass` retain the full-vocabulary normalizer. A sharp conditional distribution can still have tiny total probability on its allowed labels. Preserve both when investigating overconfidence. Calibration must be measured against declared targets on representative labeled data, using reliability, NLL/Brier and ECE where applicable. Confidence is not a causal explanation.

Historical `choice_confidence` (normalized mode probability), `ordered_confidence` and old dashboard reports keep their definitions. They are not silently relabeled as the new entropy confidence.

## References and scope

- [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions), inspected 2026-10-25: interface inspiration, not backend implementation.
- [razorback16/openjev](https://github.com/razorback16/openjev/tree/b4339c640f5331ef75098637d1be2c787715ee88): `noul`/`choice`/`score` and direct-probability serving. Its adaptive independent noisy reads are not sequential self-conditioned passes.
- [JoshuaSP/open-jev](https://github.com/JoshuaSP/open-jev/tree/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c): native denoising/sampler reference.

Typed interfaces, native diffusion, calibration and latency are separate evidence axes. Tiny CPU numerical checks pass; full-model semantic quality, GPU speed, training and robot success remain unmeasured for this path.

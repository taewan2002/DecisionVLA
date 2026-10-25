# DecisionVLA architecture

**Status: separate native reference implemented and CPU-verified; untrained (2026-10-25).**
`model.py` remains the earlier action-only pilot. The new `native_*` modules use an
isolated pinned runtime. Full26B GPU execution and robot performance are still
unverified. [Implementation and commands](native-reference.md). Existing checkpoints
and benchmark results retain their original meaning.

![DecisionVLA architecture proposal](../figures/decisionvla-native/architecture.png)

## Typed decisions and robot control

An agent-view RGB observation and task instruction condition eight timesteps of
seven LIBERO controls. Six controls use256 bins on `[-1,1]`; the gripper uses two
endpoints. Return actions, the56 conditional distributions, selected probabilities,
entropy, ordered-control variance, allowed-set probability mass and provenance.
No added action expert, world model or learned confidence head is required.

The [typed interface](typed-decisions.md) follows the shared evidence + questions
+ ordered answers principle of the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).
The default robot profile comprises 48 control-bin choices and 8 positive-command
predicates. Question meaning and allowed-label mappings enter the model prefix.
The same schema binds training targets, output slots and controls; it is not a
post-hoc renaming of arbitrary action probabilities. General score/choice/predicate
queries compile through the same backend but remain explicitly unvalidated experiments.

The backbone is `google/diffusiongemma-26B-A4B-it`, revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`. The prefix and decoder share the native
text Transformer and rank32 LoRA. The vision encoder and router remain frozen.
Prefix KV is reused within the current observation, then invalidated when the
image or instruction changes.

## Preserve the reference inference algorithm

The primary reference is [JoshuaSP/open-jev](https://github.com/JoshuaSP/open-jev/tree/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c).
Its inference harness keeps fixed structure, leaves variable positions unrestricted
through intermediate denoising and reads allowed values from final logits. It
retains DiffusionGemma's sampler while overriding canvas initialization and disabling
adaptive stopping for an exact pass budget. It provides no VLA training recipe.
Relevant source: [canvas](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/parallel_canvas.py),
[sampler wrapper](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/json_canvas.py)
and [generation call](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/inference.py).

The DecisionVLA reference path calls native `model.generate()` with the
structured canvas processor. It must preserve:

1. A256-token canvas; variable action slots initialize from **uniform full vocabulary**.
2. Native full-vocabulary logits, softcapping and temperature processing.
3. Native categorical proposals, entropy-bound acceptance and re-noising of
   unaccepted slots. No custom decreasing Bernoulli replacement schedule.
4. Native self-conditioning from the previous **processed full-vocabulary logits**.
   The first pass has no self-conditioning; later passes use the native soft-embedding path.
5. Exact1/2/4-pass budgets, with adaptive stopping disabled as in the reference.
6. Action restriction only in the terminal readout. Fixed syntax remains pinned.

The reference runtime is Transformers5.11.0, pinned source
`e7b5b964e6f64923f2770208178f3ed367978895`; the existing pilot uses5.20.2.
The reference harness also pins Torch2.14.0. Establish a separate runtime rather
than changing old environments. Export the resolved generation config with every
run: current base defaults include entropy bound0.1 and temperature endpoints
0.8/0.4. Use the upstream step indexing, not a separately guessed schedule.
See the [native generation implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)
and [self-conditioning implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py).

### Changes from the executable pilot

| Component | Previous pilot | Native DecisionVLA reference |
|---|---|---|
| Intermediate vocabulary |256 action bins; gripper endpoints | Full262,144 vocabulary entries |
| Proposal / re-noising | Custom action-only loop | Native entropy-bound sampler |
| Self-conditioning | Disabled | Native processed-logit soft embeddings |
| Canvas structure |56 action slots; remaining200 masked as padding keys | Structured256-token native canvas; normal native attention semantics |
| Final output | Action argmax and conditional probabilities | Terminal allowed-token readout plus full-vocabulary allowed mass |
| Training noise / normalizer | Action-only corruption and CE | Proposed full-vocabulary corruption and CE |

This is a material method change. Existing188ms timings and14/500 success results
remain evidence for their original implementation only.

## Robot-specific canvas adapter

Use 56 single-token labels in a flat `{"answers":[...]}` canvas, with a
one-to-one correspondence to the ordered typed questions. The action-token IDs
remain 261888–262143. Continuous slots admit 256 labels; gripper slots bind true
to 262143 (+1) and false to 261888 (−1). All variable slots still admit the full
vocabulary during denoising. The robot decoder applies the established modal-bin
rule and resolves a gripper tie to −1.

The compiler inserts unused label IDs directly; their printable aliases do not
re-encode to a single token. It validates the decoded layout under the pinned
base tokenizer. The earlier nested 243-token action template and its audit are
preserved as historical design evidence, not claimed as the current layout.
The current flat answer layout and compact robot prefix have their own
[verification record](../reports/native-reference-20261025/verification.json).

A compact numeric JSON representation also fits for the inspected constant-value
examples, but numbers can occupy multiple tokens. We choose single-token labels
to obtain an unambiguous per-control256-way distribution without inventing a
sequence probability from greedy trie selection. Action positions condition on
one another; the56 marginals do **not** constitute an independent joint policy.

The tied action embedding deltas must affect the input lookup, full output
projection and self-conditioning expectation consistently. Keep the base matrix
frozen and add the action-row delta contribution in each path; test all three.
A separate256-row head cannot implement the intermediate full-vocabulary path.

## Proposed adaptation objective

OpenJEV inference code does not specify robot fine-tuning. This section is our
training design, informed by Google's
[DiffusionGemma SFT adapter](https://github.com/google-deepmind/gemma/tree/9e1f74131ba9d45e5fe6ae7bb3941d9f2abb39c3/gemma/diffusion/hackable_diffusion_adapter).
That code uses full-vocabulary uniform corruption, a stop-gradient first pass
and50% self-conditioning. Its auxiliary encoder loss and task-specific settings
are not silently inherited here.

For valid action positions, draw replacement mask `m` and uniform vocabulary noise
`u`, then use `x_t = where(m, u, x_0)`. Hold syntax/EOS fixed. The proposed initial
noise mixture is80% samples with a uniform replacement rate and20% fully noisy
samples. This mixture is a DecisionVLA choice, not an OpenJEV training prescription.

Minimize full-vocabulary cross-entropy at valid positions **selected for
replacement**, including accidental matches between noise and target. Average
within an observation before averaging the batch; never condition supervision on
`noisy_token != target_token`. Keep syntax and unavailable future timesteps out
of the action loss. Use a fixed valid-slot mask for short chunks.

With probability0.5, obtain stop-gradient first-pass predictions and supply the
native layout/temperature-processed logits to the supervised pass. Otherwise use
true absent self-conditioning, which yields zero soft embeddings in HF. Zero
logits instead produce a uniform embedding expectation and are not interchangeable.
The supervised pass sees the same corrupted canvas in this SFT proposal; native
accepted/re-noised inference trajectories are an explicit diagnostic, not a claim
that the training distribution is identical. No auxiliary encoder AR loss is
proposed initially; action gradients still flow through prefix KV into shared LoRA.

Initialize a **fresh** rank32 adapter and action-row deltas from the pinned base.
Do not resume the old46-step optimizer or import Spatial pilot weights. Retain
the audited original40-task split (1,800 train /200 validation demonstrations),
uniform task-first sampling, post-action alignment and camera convention.
The proposed budget remains global batch64 /100k updates, subject to the
capacity, resume and training checks in [development](development.md).

## Probabilities and confidence

The [typed API](typed-decisions.md) returns `predicate`, `choice` and `score`
answers from question-specific logits; `noul` is a predicate spelling alias. Typed APIs, native sampling and calibration are
separate concerns; maintaining a sampler is not a definition of all Jev-like models.

For raw final logits `z` before sampler temperature, expose
`p_allowed = softmax(z[allowed])` and
`allowed_mass = exp(logsumexp(z[allowed]) - logsumexp(z))`.
They answer different questions: a sharp conditional distribution can coexist
with very little full-vocabulary support for any allowed action. Save this
provenance rather than conflating it with the temperature-scaled sampling distribution.

Retain selected probability, entropy and distance-aware summaries. Record per-pass
values, action error, accepted masks and seed-to-seed disagreement for diagnostics.
A sharper distribution after denoising does not prove a better action. These
quantities are not calibrated task-success probabilities or causal explanations.
Calibration, confidence-triggered replanning and early exit remain separate,
unimplemented validation targets.

## Other inspected implementations

| Reference | Actual mechanism | Role in this design |
|---|---|---|
| [JoshuaSP/open-jev](https://github.com/JoshuaSP/open-jev/tree/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c) | DiffusionGemma native loop; final typed readout | Primary inference oracle |
| [razorback16/openjev](https://github.com/razorback16/openjev/tree/b4339c640f5331ef75098637d1be2c787715ee88) | DiffusionGemma decision server using modified vLLM read-only endpoints | Serving/interface reference; not assumed equivalent to the HF oracle |
| [Saik0s/diffusiongemma-jev-macos](https://github.com/Saik0s/diffusiongemma-jev-macos/tree/070a792df210e5a5139622e049d0040988502873) | MLX label readout; optional two-pass structured trajectory | Terminal projection and uncertainty reference; different runtime/trajectory details |
| [Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev/tree/bd4118882f733574a3250a4b65fe4d884130c08b) | Qwen candidate scoring with LoRA/scalar head and calibration | Alternative decision-model baseline; no diffusion sampler |
| [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev/tree/23cf1f39fc9534fe81437200959b6dfc7106e45a) | Direct choice logits from open language models | Simple decision-readout baseline; no DiffusionGemma loop |

Similar names do not imply the same architecture. These projects are independent
implementations, not disclosures of proprietary Jev training or calibration.
[Inspected revisions and file hashes](../reports/openjev-redesign-20261025/upstream-sources.json).

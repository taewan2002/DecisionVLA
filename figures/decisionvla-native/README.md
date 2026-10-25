# DecisionVLA architecture

![DecisionVLA](architecture.png)

[Editable Figma](https://www.figma.com/design/FbL9C7OnlyOZkiyGPOkcW0/?node-id=13407-92873) · [SVG](architecture.svg) · [Authoring script](build-figma.js)

This 1360 × 424 editable vector figure describes the proposed architecture,
not the implementation used for existing results. The PNG is exported at 2×.
The earlier pilot figure remains preserved in
[decisionvla-architecture](../decisionvla-architecture/). The frame stays inside
Frame 2117910921; its previous design is kept as a hidden Figma backup.
Neighboring project figures are unchanged.

The blue blocks represent the vision encoder and the shared Transformer path.
The image enters the encoder; the instruction conditions the prefix. Cached
context feeds the denoiser, and the green loop shows iterative sampling with
self-conditioning and re-noising. The final readout returns parallel actions and
their probability diagnostics. Intermediate predictions retain the full vocabulary;
only the final action readout is restricted. The 1/2/4-pass choices require
quality–latency validation.

Cells and bars are schematic, not measured probabilities. Raw confidence is not
success probability. Implementation status and interpretation stay in this text
and the repository README rather than badges or footnotes within the figure.
See the [exact design](../../docs/architecture.md).

The observation is reused from the existing LIBERO-Spatial figure: demo0/frame0,
`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
rotated180° under the pilot convention. Its shortened instruction is illustrative.
Dataset attribution and terms remain in [third-party](../../docs/third-party.md).

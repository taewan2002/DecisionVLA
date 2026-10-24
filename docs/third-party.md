# Sources and licenses

DecisionVLA's original code is MIT licensed. That license does not replace the licenses of base models, datasets or dependencies.

| Source | Use and terms |
|---|---|
| [Google DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | Base model. The pinned model card identifies Apache-2.0 and links the [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license). Base weights are not included here. |
| [NVIDIA NVFP4 DiffusionGemma](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) | Untrained base used for the latency proxy. Its card lists Apache-2.0 and additional Gemma terms links. Retain and review that card for any redistribution. |
| [Transformers](https://github.com/huggingface/transformers) | Native DiffusionGemma model implementation and grouped MoE kernels, Apache-2.0. Imported as a dependency. |
| [vLLM](https://github.com/vllm-project/vllm) | Base-model inference benchmark, Apache-2.0. The small compatibility patch is explicitly recorded. |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | Code is MIT; the official repository identifies dataset licensing as CC BY 4.0. The HF mirror's metadata labels Apache-2.0; preserve this distinction and credit the original authors. Dataset bytes are not redistributed here. |
| [open-jev](https://github.com/JoshuaSP/open-jev) | Conceptual reference for constrained canvas inference; no claim that this project reproduces JEV or contains a trained OpenJEV VLA. |
| [Google's DiffusionGemma training recipes](https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion/hackable_diffusion_adapter) | Reference for fine-tuning feasibility. DecisionVLA uses its own simpler PyTorch action objective, not an exact recipe reproduction. |
| [NVIDIA Alpamayo quantization example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/alpamayo) | Existing FP8/NVFP4 VLA quantization with calibration. A direct precedent against claiming that NVFP4 VLA alone is novel. |

LIBERO attribution: Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu and Peter Stone, *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, 2023.

Prior work includes [Discrete Diffusion VLA](https://arxiv.org/abs/2508.20072). Direct discrete action diffusion and quantization are not presented as new concepts here.

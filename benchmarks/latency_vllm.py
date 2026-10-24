"""Untrained DiffusionGemma workload proxy; never controls a robot.

Pinned for vLLM 0.30.0. Full-vocabulary, one 256-token canvas, new image
encoding and prefix on every request. No claim of learned VLA performance.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image


def worker_control(worker, steps=None, reset=False):
    import torch
    runner = worker.model_runner
    assert runner.sampler.diffusion_states is runner.model_state.diffusion_states
    if steps is not None:
        runner.model_state.diffusion_states.max_denoising_steps = int(steps)
        # Negative entropy threshold prevents confidence-based early termination.
        runner.sampler.confidence_threshold = -1.0
    if not hasattr(runner, "_benchmark_execute_count"):
        runner._benchmark_execute_count = 0
        runner._benchmark_token_counts = {}
        original = runner.execute_model
        def counted(*args, **kwargs):
            scheduled = args[0] if args else kwargs["scheduler_output"]
            tokens = int(scheduled.total_num_scheduled_tokens)
            if tokens:
                runner._benchmark_execute_count += 1
                key = str(tokens)
                runner._benchmark_token_counts[key] = runner._benchmark_token_counts.get(key, 0) + 1
            return original(*args, **kwargs)
        runner.execute_model = counted
    if reset:
        runner._benchmark_execute_count = 0
        runner._benchmark_token_counts = {}
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return {
        "execute_count": runner._benchmark_execute_count,
        "scheduled_token_histogram": runner._benchmark_token_counts,
        "steps": runner.model_state.diffusion_states.max_denoising_steps,
        "confidence_threshold": runner.sampler.confidence_threshold,
        "allocated_GiB": torch.cuda.memory_allocated() / 2**30,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
    }


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


class BenchmarkWorkerExtension:
    """Named local RPC avoids serializing Python callables with pickle."""
    def benchmark_control(self, steps=None, reset=False):
        return worker_control(self, steps, reset)


def make_observation(serial, views):
    # Different deterministic images defeat both processor and encoder reuse.
    # Synthetic images measure workload shape only; MoE routing is content-dependent.
    rng = np.random.default_rng(10000 + serial)
    images = [Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
              for _ in range(views)]
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", default="1,2,4,8")
    parser.add_argument("--views", default="1,2")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--visual-tokens", type=int, default=280)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.30)
    args = parser.parse_args()
    import torch
    import transformers
    import vllm
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    engine_args = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=1024,
        async_scheduling=False,
        gpu_memory_utilization=args.memory_fraction, kv_cache_memory_bytes=512 * 2**20,
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False, mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 2, "video": 0},
        mm_processor_kwargs={"max_soft_tokens": args.visual_tokens},
        attention_config={"backend": "TRITON_ATTN"},
        diffusion_config={"canvas_length": 256, "max_denoising_steps": 4},
        override_generation_config={"confidence_threshold": -1.0, "max_new_tokens": None},
        worker_extension_cls="latency_vllm.BenchmarkWorkerExtension",
        enforce_eager=args.eager, seed=42,
        compilation_config={"cudagraph_capture_sizes": [256]},
    )
    report = {
        "status": "loading", "benchmark_kind": "untrained_base_model_workload_proxy",
        "weights_trained_for_actions": False, "robot_execution": False,
        "synthetic_images": True, "input_image_size": [256, 256],
        "action_shape_target": [8, 7], "returned_text_tokens": 56,
        "actual_canvas_tokens": 256, "action_constrained_readout_implemented": False,
        "timing_boundary": "PIL images + instruction -> vLLM preprocessing, vision, fresh prefill, denoise, commit, 56 text IDs returned",
        "excluded": ["image acquisition", "network/RPC to a remote client", "robot actuation", "trained action detokenization"],
        "warmups": args.warmups, "repeats": args.repeats,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
        "gpu_total_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "engine_args": engine_args, "cases": [],
    }
    write_json(output / "results.json", report)
    t = time.perf_counter()
    llm = LLM(**engine_args)
    report["load_and_engine_init_seconds"] = time.perf_counter() - t
    report["status"] = "running"
    write_json(output / "results.json", report)
    params = SamplingParams(max_tokens=56, ignore_eos=True,
                            temperature=1.0, detokenize=False)
    serial = 0
    for views in map(int, args.views.split(",")):
        for steps in map(int, args.steps.split(",")):
            control = llm.collective_rpc("benchmark_control", args=(steps, True))[0]
            assert control["steps"] == steps and control["confidence_threshold"] == -1.0
            rows = []
            for i in range(args.warmups + args.repeats):
                images = make_observation(serial, views)
                serial += 1
                image_hash = hashlib.sha256(b"".join(im.tobytes() for im in images)).hexdigest()
                content = [{"type": "image"} for _ in images] + [{
                    "type": "text",
                    "text": "Move the red block into the tray. Robot state: "
                            "x=0.10 y=-0.05 z=0.20 rx=0.00 ry=0.00 rz=0.00 gripper=1. "
                            "Predict the next 8 actions, each with 7 dimensions.",
                }]
                start = time.perf_counter()
                prompt = processor.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                result = llm.generate({"prompt": prompt, "multi_modal_data": {"image": images}},
                                      params, use_tqdm=False)[0]
                elapsed_ms = (time.perf_counter() - start) * 1000
                ids = result.outputs[0].token_ids
                assert len(ids) == 56, len(ids)
                if i >= args.warmups:
                    rows.append({"ms": elapsed_ms, "image_sha256": image_hash,
                                 "prompt_tokens": len(result.prompt_token_ids),
                                 "output_tokens": len(ids),
                                 "output_ids_sha256": hashlib.sha256(str(list(ids)).encode()).hexdigest()})
            control = llm.collective_rpc("benchmark_control")[0]
            # One prefill + S denoise iterations + one canvas commit per request.
            expected = (args.warmups + args.repeats) * (steps + 2)
            values = np.array([r["ms"] for r in rows])
            case = {"views": views, "steps": steps, "visual_token_budget_per_image": args.visual_tokens,
                    "p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "min_ms": float(values.min()), "max_ms": float(values.max()),
                    "mean_ms": float(values.mean()),
                    "worker": control, "expected_execute_count": expected, "samples": rows}
            report["cases"].append(case)
            write_json(output / "results.json", report)
            print(json.dumps({k:v for k,v in case.items() if k != "samples"}), flush=True)
            assert control["execute_count"] == expected, (control, expected)
            assert control["scheduled_token_histogram"].get("256") == (args.warmups + args.repeats) * (steps + 1), control
    report["status"] = "completed"
    write_json(output / "results.json", report)


if __name__ == "__main__":
    main()

"""All-rank/global-batch forward replay under a bounded-memory inference process.

Does not replace the full DDP gradient, optimizer/resume or timing gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from decisionvla.adapters import LoRAExperts
from decisionvla.checkpoints import load_adapter
from decisionvla.data import LiberoDataset
from decisionvla.distributed import prepare_microbatch, restore_rng, sample_rank_indices
from decisionvla.model import action_logits, denoising_loss, encode_prefix, load_model
from diagnose_forward_numerics import stream_modules


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "data", "parent", "output", "reference_audit"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    config = json.loads((args.parent/"adapter_config.json").read_text())
    config["expert_kernel"] = "torch_grouped_mm"
    model, processor, _ = load_model(str(args.model), config, "cpu")
    load_adapter(model, args.parent)
    model.eval()
    model.gradient_checkpointing_disable()
    placement, handles = stream_modules(model)
    data = LiberoDataset(args.data, config)
    state = torch.load(args.parent/"training_state.pt", map_location="cpu", mmap=True, weights_only=False)
    rngs = state["rank_rng"]
    del state
    expected = json.loads(args.reference_audit.read_text())["input_hashes"][0]
    activation_hashes = []

    def observe(module, inputs, output):
        values = output if isinstance(output, tuple) else (output,)
        activation_hashes.extend(tensor_sha(value) for value in values)

    for module in model.modules():
        if isinstance(module, LoRAExperts) or module.__class__.__name__ == "DiffusionGemmaTextRouter":
            handles.append(module.register_forward_hook(observe))
    rows, start = [], time.time()
    with (args.output/"events.jsonl").open("x", buffering=1) as events:
        for rank in range(config["world_size"]):
            sampler = np.random.default_rng()
            noise = torch.Generator(device="cuda")
            restore_rng(rngs[rank], sampler, noise)
            batches = sample_rank_indices(sampler, len(data), config, rank)
            for micro, indices in enumerate(batches):
                inputs, noisy, targets, masks, keys, _ = prepare_microbatch(
                    model, processor, data, config, indices, noise, torch.device("cuda"))
                digest = hashlib.sha256()
                for name, value in [*sorted(inputs.items()), ("noisy", noisy), ("targets", targets), ("masks", masks)]:
                    digest.update(name.encode())
                    digest.update(value.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                assert digest.hexdigest() == expected[rank][micro]
                results, internal = [], []
                for kernel in ("triton_sm120_backward", "cublas_sm120_forward"):
                    for module in model.modules():
                        if isinstance(module, LoRAExperts):
                            module.kernel = kernel
                    activation_hashes.clear()
                    cache = encode_prefix(model, inputs)
                    logits = action_logits(model, cache, noisy, config,
                                           prefix_attention_mask=inputs["attention_mask"])
                    results.append({"logits_sha256": tensor_sha(logits),
                                    "loss": float(denoising_loss(logits, targets, masks))})
                    internal.append(list(activation_hashes))
                    del cache, logits
                row = {"rank": rank, "microbatch": micro, "sample_keys": keys,
                       "input_sha256": digest.hexdigest(), "outputs": results,
                       "activation_hashes": internal[0],
                       "all_router_expert_tensors_exact": internal[0] == internal[1],
                       "readout_exact": results[0] == results[1]}
                events.write(json.dumps(row)+"\n")
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in {"activation_hashes", "sample_keys"}}),
                      flush=True)
                assert row["all_router_expert_tensors_exact"] and row["readout_exact"]
    report = {"status": "passed", "rows": rows, "global_batch": sum(len(r["sample_keys"]) for r in rows),
              "mean_loss": sum(r["outputs"][0]["loss"] for r in rows)/len(rows),
              "placement": placement, "elapsed_seconds": time.time()-start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "optimizer_updates": 0, "training_gradient_resume_timing_gate": "not_run",
              "contended_diagnostic_not_benchmark": True}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    data.close()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()

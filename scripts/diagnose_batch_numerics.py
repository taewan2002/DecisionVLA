"""Separate padding effects from batch-shaped BF16/MoE arithmetic; no optimizer updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from decisionvla.batching import pack_observations
from decisionvla.checkpoints import load_adapter
from decisionvla.data import LiberoDataset, prepare_observation
from decisionvla.model import action_logits, encode_prefix, load_model, training_corruption
from decisionvla.train import to_device


def difference(reference, actual):
    continuous = torch.arange(reference.shape[1]) % 7 != 6
    delta = (reference[:, continuous] - actual[:, continuous]).abs()
    return {"continuous_logit_mae": float(delta.mean()), "continuous_logit_max_abs": float(delta.max()),
            "probability_mae": float((reference.softmax(-1) - actual.softmax(-1)).abs().mean()),
            "argmax_agreement": float((reference.argmax(-1) == actual.argmax(-1)).float().mean())}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    model, processor, _ = load_model(args.model, config)
    load_adapter(model, args.checkpoint)
    model.eval()
    dataset = LiberoDataset(args.data, config)
    state = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    sampler = np.random.default_rng()
    sampler.bit_generator.state = state["sampler"]
    noise = torch.Generator(device="cuda")
    noise.set_state(state["noise"])
    del state
    items = [dataset[int(sampler.integers(len(dataset)))] for _ in range(8)]
    processed = [prepare_observation(processor, item["image"], item["task"], config["visual_tokens"])
                 for item in items]
    noisy = [training_corruption(torch.from_numpy(item["tokens"])[None].cuda(),
                                torch.from_numpy(item["valid"])[None].cuda(), config, noise)[0]
             for item in items]
    routes = []
    handles = [module.register_forward_hook(lambda module, inputs, output: routes.append(output[2].detach().cpu()))
               for module in model.modules() if module.__class__.__name__ == "DiffusionGemmaTextRouter"]

    def forward(inputs, action, masked):
        routes.clear()
        inputs = to_device(inputs, "cuda")
        cache = encode_prefix(model, inputs)
        logits = action_logits(model, cache, action, config,
                               **({"prefix_attention_mask": inputs["attention_mask"]} if masked else {}))
        return logits.cpu(), [value.clone() for value in routes]

    reference = [forward(p, n, False)[0] for p, n in zip(processed, noisy, strict=True)]
    length = max(p["input_ids"].shape[1] for p in processed)
    padded_single = []
    for p, n in zip(processed, noisy, strict=True):
        p = {k: F.pad(v, (0, length-v.shape[1]), value=model.config.text_config.pad_token_id if k == "input_ids" else 0)
             if k in {"input_ids", "attention_mask", "mm_token_type_ids"} else v for k, v in p.items()}
        padded_single.append(forward(p, n, True)[0])
    result = {"updates": 0, "prefix_lengths": [p["input_ids"].shape[1] for p in processed],
              "padded_single_vs_unpadded": difference(torch.cat(reference), torch.cat(padded_single)),
              "duplicate_same_length": {}}
    for batch in (2, 4):
        batched = []
        for start in range(0, 8, batch):
            p = pack_observations(processed[start:start+batch], model.config.text_config.pad_token_id)
            batched.append(forward(p, torch.cat(noisy[start:start+batch]), True)[0])
        result[f"batch_{batch}_vs_unpadded"] = difference(torch.cat(reference), torch.cat(batched))
        # Identical-length copies remove padding as an explanation for any residual difference.
        original, ref_routes = forward(processed[0], noisy[0], False)
        duplicate, batch_routes = forward(pack_observations([processed[0]] * batch,
                                            model.config.text_config.pad_token_id), noisy[0].repeat(batch, 1), True)
        mismatch, total = 0, 0
        for a, b in zip(ref_routes, batch_routes, strict=True):
            a = a.reshape(-1, 8).sort(-1).values
            b = b.reshape(batch, -1, 8)[0].sort(-1).values
            mismatch += int((a != b).any(-1).sum())
            total += a.shape[0]
        result["duplicate_same_length"][str(batch)] = {
            **difference(original, duplicate[:1]), "changed_expert_set_token_layers": mismatch,
            "total_token_layers": total, "changed_expert_set_fraction": mismatch / total}
    for handle in handles:
        handle.remove()
    dataset.close()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Bounded safetensors shards for large expert adapters, with v1 compatibility."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_adapter(model, path, max_shard_bytes=1_000_000_000):
    path = Path(path)
    groups, group, size = [], [], 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        nbytes = parameter.numel() * 4
        if nbytes > max_shard_bytes:
            raise ValueError(f"Parameter exceeds shard limit: {name}")
        if group and size + nbytes > max_shard_bytes:
            groups.append(group)
            group, size = [], 0
        group.append((name, parameter))
        size += nbytes
    if group:
        groups.append(group)
    if not groups:
        raise ValueError("No trainable adapter parameters")
    weight_map, total = {}, 0
    for i, group in enumerate(groups, 1):
        filename = ("adapter.safetensors" if len(groups) == 1 else
                    f"adapter-{i:05d}-of-{len(groups):05d}.safetensors")
        state = {name: p.detach().float().cpu().contiguous() for name, p in group}
        save_file(state, path / filename)
        weight_map.update({name: filename for name in state})
        total += sum(p.numel() * 4 for p in state.values())
        del state
    if len(groups) > 1:
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (path / "adapter.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return {"shards": len(groups), "total_size": total}


def adapter_files(path):
    path = Path(path)
    index_path = path / "adapter.safetensors.index.json"
    if not index_path.exists():
        return [path / "adapter.safetensors"]
    index = json.loads(index_path.read_text())
    filenames = sorted(set(index["weight_map"].values()))
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in filenames):
        raise ValueError("Invalid adapter shard path")
    return [path / name for name in filenames]


def load_adapter(model, path):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    seen = set()
    with torch.no_grad():
        for filename in adapter_files(path):
            state = load_file(filename)
            if seen.intersection(state) or not set(state).issubset(expected):
                raise ValueError("Unexpected or duplicated adapter keys")
            for name, value in state.items():
                if value.shape != expected[name].shape:
                    raise ValueError(f"Adapter shape mismatch: {name}")
                expected[name].copy_(value)
            seen.update(state)
            del state
    if seen != set(expected):
        raise ValueError("Missing adapter parameters")

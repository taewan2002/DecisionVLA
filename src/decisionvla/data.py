"""Episode-level split and action chunks from the original LIBERO HDF5 files."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .codec import ActionCodec


class LiberoDataset:
    def __init__(self, root, config, split="train"):
        if split not in ("train", "validation"):
            raise ValueError(split)
        self.config = config
        self.codec = ActionCodec(config["bins"])
        self.episodes = []
        self.frames = []
        self.handles = {}
        for path in sorted((Path(root) / config["suite"]).glob("*.hdf5")):
            with h5py.File(path, "r") as f:
                for demo in sorted(f["data"], key=lambda x: int(x.split("_")[-1])):
                    index = int(demo.split("_")[-1])
                    is_val = index % config["validation_demo_modulus"] == config["validation_demo_remainder"]
                    if is_val != (split == "validation"):
                        continue
                    group = f["data"][demo]
                    length = len(group["actions"])
                    if len(group["obs"][config["camera"]]) != length:
                        raise ValueError("Observation/action length mismatch")
                    task = path.stem.removesuffix("_demo").replace("_", " ")
                    episode = (path, demo, length, task)
                    self.episodes.append(episode)
                    self.frames.extend((len(self.episodes) - 1, t)
                                       for t in range(length - config["action_offset"]))
        if not self.frames:
            raise ValueError(f"No {split} episodes found in {root}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        episode_index, start = self.frames[index]
        path, demo, length, task = self.episodes[episode_index]
        if path not in self.handles:
            self.handles[path] = h5py.File(path, "r")
        group = self.handles[path]["data"][demo]
        image = group["obs"][self.config["camera"]][start]
        if self.config["rotate_image_180"]:
            image = image[::-1, ::-1].copy()
        horizon = self.config["horizon"]
        action_start = start + self.config["action_offset"]
        valid_steps = min(horizon, length - action_start)
        actions = np.zeros((horizon, 7), dtype=np.float32)
        actions[:valid_steps] = group["actions"][action_start:action_start + valid_steps]
        actions[valid_steps:] = actions[valid_steps - 1]
        valid = np.repeat(np.arange(horizon) < valid_steps, 7)
        return {"image": Image.fromarray(image), "task": task, "actions": actions,
                "tokens": self.codec.encode(actions).reshape(-1), "valid": valid,
                "key": f"{path.name}/{demo}/{start}"}

    def manifest(self):
        episodes = [{"file": p.name, "demo": d, "frames": n} for p, d, n, _ in self.episodes]
        digest = hashlib.sha256(json.dumps(episodes, sort_keys=True).encode()).hexdigest()
        return {"episodes": episodes, "frames": len(self), "sha256": digest}

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def prepare_observation(processor, image, task, visual_tokens=280):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": f"Robot task: {task}. Predict the next 8 robot actions."},
    ]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt",
                                         processor_kwargs={"images_kwargs": {"max_soft_tokens": visual_tokens}})

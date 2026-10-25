"""Episode-level split and action chunks from the original LIBERO HDF5 files."""
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .codec import ActionCodec


def dataset_files(root, config):
    """Resolve every declared suite, failing instead of silently using a subset."""
    if "suites" in config and "suite" in config:
        raise ValueError("Specify suite or suites, not both")
    suites = config.get("suites", [config.get("suite")])
    if (not isinstance(suites, list) or not suites
            or any(not isinstance(s, str) or not s or Path(s).name != s or s in {".", ".."}
                   for s in suites) or len(set(suites)) != len(suites)):
        raise ValueError("Expected distinct suite directory names")
    expected = config.get("expected_tasks")
    if expected is not None and set(expected) != set(suites):
        raise ValueError("Expected task manifest must cover exactly the declared suites")
    paths = []
    for suite in suites:
        selected = sorted((Path(root) / suite).glob("*.hdf5"))
        if not selected:
            raise ValueError(f"Missing dataset suite: {suite}")
        if expected is not None:
            actual = {p.stem.removesuffix("_demo") for p in selected}
            names = expected[suite]
            if len(names) != len(set(names)) or actual != set(names):
                raise ValueError(f"Task manifest mismatch: {suite}")
        paths.extend(selected)
    return paths


def task_instruction(path):
    """Match the pinned original LIBERO language for standard suite filenames."""
    name = path.stem.removesuffix("_demo")
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)
    return name.replace("_", " ")


class LiberoDataset:
    def __init__(self, root, config, split="train"):
        if split not in ("train", "validation"):
            raise ValueError(split)
        self.config = config
        self.codec = ActionCodec(config["bins"])
        self.episodes = []
        self.frames = []
        self.handles = {}
        self.task_ranges = []
        for path in dataset_files(root, config):
            first = len(self.frames)
            with h5py.File(path, "r") as f:
                expected_demos = config.get("expected_demonstrations_per_task")
                if expected_demos is not None and set(f["data"]) != {
                        f"demo_{i}" for i in range(expected_demos)}:
                    raise ValueError(f"Demonstration manifest mismatch: {path}")
                for demo in sorted(f["data"], key=lambda x: int(x.split("_")[-1])):
                    index = int(demo.split("_")[-1])
                    is_val = index % config["validation_demo_modulus"] == config["validation_demo_remainder"]
                    if is_val != (split == "validation"):
                        continue
                    group = f["data"][demo]
                    length = len(group["actions"])
                    if len(group["obs"][config["camera"]]) != length:
                        raise ValueError("Observation/action length mismatch")
                    task = task_instruction(path)
                    episode = (path, demo, length, task)
                    self.episodes.append(episode)
                    self.frames.extend((len(self.episodes) - 1, t)
                                       for t in range(length - config["action_offset"]))
            if len(self.frames) == first:
                raise ValueError(f"No {split} frames for task: {path}")
            self.task_ranges.append((first, len(self.frames), path.parent.name, path.name))
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
                "key": f"{self.file_key(path)}/{demo}/{start}", "suite": path.parent.name}

    def sample_indices(self, rng, size):
        mode = self.config.get("sampling", "uniform_frames")
        if mode == "uniform_frames":
            return rng.integers(len(self), size=size)
        if mode != "uniform_task":
            raise ValueError(f"Unknown sampling mode: {mode}")
        tasks = rng.integers(len(self.task_ranges), size=size)
        starts = np.array([r[0] for r in self.task_ranges], dtype=np.int64)[tasks]
        ends = np.array([r[1] for r in self.task_ranges], dtype=np.int64)[tasks]
        return rng.integers(starts, ends)

    def validation_indices(self):
        rng = np.random.default_rng(9001)
        count = self.config.get("validation_samples_per_task")
        if count is None:
            return rng.choice(len(self), self.config["validation_samples"], replace=False)
        if type(count) is not int or count < 1 or count * len(self.task_ranges) != self.config["validation_samples"]:
            raise ValueError("Validation count must cover every task equally")
        return np.concatenate([rng.choice(end - start, count, replace=False) + start
                               for start, end, *_ in self.task_ranges])

    def file_key(self, path):
        # Keep historical single-suite keys/manifests byte-compatible.
        return f"{path.parent.name}/{path.name}" if "suites" in self.config else path.name

    def manifest(self):
        episodes = [{"file": self.file_key(p), "demo": d, "frames": n} for p, d, n, _ in self.episodes]
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

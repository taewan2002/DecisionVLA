"""Verify episode separation and the official HDF5 post-action observation alignment."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from decisionvla.data import LiberoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/libero_spatial.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train, val = (LiberoDataset(args.data, config, split) for split in ["train", "validation"])
    assert set((p.name, d) for p, d, *_ in train.episodes).isdisjoint(
        set((p.name, d) for p, d, *_ in val.episodes))
    errors, episodes, frames = [], 0, 0
    for path in sorted((Path(args.data) / config["suite"]).glob("*.hdf5")):
        with h5py.File(path) as f:
            for group in f["data"].values():
                # In the pinned Panda dataset, qpos starts at state index 1.
                observed = group["obs/joint_states"][:-1]
                next_state = group["states"][1:, 1:8]
                error = float(np.max(np.abs(observed - next_state)))
                assert error < 1e-10, (path.name, group.name, error)
                actions = group["actions"][:]
                assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.0001
                errors.append(error)
                episodes += 1
                frames += len(observed)
    output = {"episodes_checked": episodes, "transitions_checked": frames,
              "max_observation_to_next_state_joint_error": max(errors),
              "action_offset": config["action_offset"], "split_disjoint": True,
              "train": train.manifest(), "validation": val.manifest()}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k not in ["train", "validation"]}))


if __name__ == "__main__":
    main()

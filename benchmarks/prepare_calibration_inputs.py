"""Freeze 32 evenly spaced training observations for activation calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from decisionvla.data import LiberoDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    data = LiberoDataset(args.data, config, "train")
    rows = []
    try:
        for i, index in enumerate(np.linspace(0, len(data) - 1, 32, dtype=int)):
            item = data[int(index)]
            name = f"observation-{i:02d}.png"
            item["image"].save(args.output / name)
            rows.append(
                {
                    "file": name,
                    "key": item["key"],
                    "instruction": item["task"],
                    "seed": 42 + i,
                    "image_sha256": hashlib.sha256(np.asarray(item["image"]).tobytes()).hexdigest(),
                }
            )
    finally:
        data.close()
    (args.output / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

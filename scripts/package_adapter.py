"""Create a small GitHub Release asset, without optimizer state or base weights."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    args = parser.parse_args()
    files = [args.checkpoint / name for name in ["adapter.safetensors", "adapter_config.json", "training_metadata.json"]]
    files.append(args.model_card)
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=file.name)
    if args.output.stat().st_size >= 2 * 1024**3:
        raise ValueError("GitHub Release assets must be smaller than 2 GiB")
    manifest = {"asset": args.output.name, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    args.output.with_suffix(args.output.suffix + ".sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Create a small GitHub Release asset, without optimizer state or base weights."""
import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

from decisionvla.checkpoints import adapter_files, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--sharded-release", action="store_true",
                        help="Write separate release assets under --output; each is below 2 GiB")
    args = parser.parse_args()
    files = adapter_files(args.checkpoint)
    index = args.checkpoint / "adapter.safetensors.index.json"
    if index.exists():
        files.append(index)
    files.append(args.checkpoint / "adapter_config.json")
    files.append(args.model_card)
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {file.name: file_sha256(file) for file in files}
    checksum_bytes = (json.dumps(hashes, indent=2) + "\n").encode()
    if args.sharded_release:
        for file in files:
            if file.stat().st_size >= 2 * 1024**3:
                raise ValueError(f"Release asset exceeds 2 GiB: {file.name}")
        if len({file.name for file in files}) != len(files):
            raise ValueError("Release asset names collide")
        args.output.mkdir(parents=True)
        manifest = {"format": "decisionvla-release-assets-v1", "files": {}}
        for file in files:
            shutil.copyfile(file, args.output / file.name)
            manifest["files"][file.name] = {"sha256": hashes[file.name], "bytes": file.stat().st_size}
        (args.output / "sha256.json").write_bytes(checksum_bytes)
        manifest["files"]["sha256.json"] = {"sha256": hashlib.sha256(checksum_bytes).hexdigest(),
                                             "bytes": len(checksum_bytes)}
        print(json.dumps(manifest, indent=2))
        return
    if sum(file.stat().st_size for file in files) >= 2 * 1024**3:
        raise ValueError("Use --sharded-release for large adapters")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for file in files:
            archive.add(file, arcname=file.name)
        checksum_info = tarfile.TarInfo("sha256.json")
        checksum_info.size = len(checksum_bytes)
        checksum_info.mode = 0o644
        archive.addfile(checksum_info, io.BytesIO(checksum_bytes))
    if args.output.stat().st_size >= 2 * 1024**3:
        raise ValueError("GitHub Release assets must be smaller than 2 GiB")
    manifest = {"asset": args.output.name, "sha256": file_sha256(args.output),
                "files": {**hashes, "sha256.json": hashlib.sha256(checksum_bytes).hexdigest()}}
    args.output.with_suffix(args.output.suffix + ".sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

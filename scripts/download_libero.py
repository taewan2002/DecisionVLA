"""Download the official LIBERO mirror at a pinned revision, verifying every file."""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--suite", choices=SUITES)
    group.add_argument("--suites", nargs="+", choices=SUITES)
    args = parser.parse_args()
    suites = args.suites or [args.suite or "libero_spatial"]
    if len(suites) != len(set(suites)):
        parser.error("Duplicate suites")
    manifest = args.output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        previous_suites = {Path(item["file"]).parts[0] for item in previous["files"]}
        if (previous["repo"] != REPO or previous["revision"] != REVISION
                or previous_suites != set(suites)):
            raise ValueError("Use a new output directory when changing the dataset scope")
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    metadata = json.load(urllib.request.urlopen(url))
    assert metadata["sha"] == REVISION
    files = []
    for suite in suites:
        selected = [f for f in metadata["siblings"]
                    if f["rfilename"].startswith(suite + "/") and f["rfilename"].endswith(".hdf5")]
        if len(selected) != 10:
            raise ValueError(f"Expected ten tasks in {suite}, got {len(selected)}")
        files.extend(selected)

    def download(item):
        path = args.output / item["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".partial")
            uri = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['rfilename']}"
            with urllib.request.urlopen(uri, timeout=300) as response, temporary.open("wb") as out:
                while chunk := response.read(8 << 20):
                    out.write(chunk)
            with temporary.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item["lfs"]["sha256"]:
                    raise ValueError(f"Downloaded file hash mismatch: {item['rfilename']}")
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == item["lfs"]["sha256"], path.name
        result = {"file": item["rfilename"], "bytes": path.stat().st_size, "sha256": digest}
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, files))
    temporary_manifest = args.output / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps({"repo": REPO, "revision": REVISION,
                                               "files": verified}, indent=2) + "\n")
    temporary_manifest.replace(manifest)


if __name__ == "__main__":
    main()

"""Download the official LIBERO mirror at a pinned revision, verifying every file."""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    args = parser.parse_args()
    url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true"
    metadata = json.load(urllib.request.urlopen(url))
    assert metadata["sha"] == REVISION
    files = [f for f in metadata["siblings"] if f["rfilename"].startswith(args.suite + "/") and f["rfilename"].endswith(".hdf5")]
    assert len(files) == 10, len(files)

    def download(item):
        path = args.output / item["rfilename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".partial")
            uri = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['rfilename']}"
            with urllib.request.urlopen(uri, timeout=300) as response, temporary.open("wb") as out:
                while chunk := response.read(8 << 20):
                    out.write(chunk)
            temporary.replace(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == item["lfs"]["sha256"], path.name
        result = {"file": item["rfilename"], "bytes": path.stat().st_size, "sha256": digest}
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, files))
    (args.output / "manifest.json").write_text(json.dumps({"repo": REPO, "revision": REVISION, "files": verified}, indent=2) + "\n")


if __name__ == "__main__":
    main()

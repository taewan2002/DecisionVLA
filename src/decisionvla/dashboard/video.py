"""Attach audited LIBERO videos to an existing replay without running inference."""
import argparse
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .import_trace import sha


def video_entries(directory):
    directory = Path(directory).resolve()
    manifest = directory / "videos.json"
    if not manifest.exists():
        return []
    entries = json.loads(manifest.read_text())["videos"]
    for entry in entries:
        path = (directory / entry["file"]).resolve()
        if (not path.is_relative_to(directory) or path.suffix != ".mp4"
                or not re.fullmatch(r"[a-f0-9]{64}", entry["id"])
                or entry["fps"] != 20 or entry["timebase"] != "simulation"):
            raise ValueError("Invalid replay video manifest")
    return entries


def public_videos(directory):
    return [{key: value for key, value in entry.items() if key != "file"}
            | {"url": f'/api/video?id={entry["id"]}'} for entry in video_entries(directory)]


def byte_range(header, size):
    """One inclusive HTTP byte range; malformed/unsatisfiable ranges raise."""
    if not header:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("Unsupported byte range")
    first, last = match.groups()
    if not first:
        if int(last) <= 0:
            raise ValueError("Empty suffix range")
        return max(0, size - int(last)), size - 1
    start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start > end or start >= size:
        raise ValueError("Unsatisfiable byte range")
    return start, end


def attach_videos(directory, results, audit_path, episodes):
    directory, results = Path(directory), Path(results)
    metadata = json.loads((directory / "session.json").read_text())
    if metadata["contract_sha256"] != sha(results / "contract.json"):
        raise ValueError("Video source contract differs from the replay")
    hashes = json.loads(Path(audit_path).read_text())["artifacts_sha256"]
    # Full on-disk history, not the bounded UI index.
    recorded = [json.loads(path.read_text()) for path in sorted((directory / "frames").glob("*.json"))]
    entries = []
    for episode in episodes:
        folder = (results / episode).resolve()
        if not folder.is_relative_to(results.resolve()):
            raise ValueError("Episode path outside results")
        for name in ("rollout.mp4", "trace.npz", "result.json"):
            if sha(folder / name) != hashes[f"{episode}/{name}"]:
                raise ValueError(f"Audited {name} SHA mismatch")
        result = json.loads((folder / "result.json").read_text())
        queries = [f for f in recorded if f["context"].get("episode") == episode]
        with np.load(folder / "trace.npz", allow_pickle=False) as trace:
            if len(queries) != result["policy_queries"] or len(trace["actions"]) != result["policy_steps"]:
                raise ValueError("Incomplete replay query/step mapping")
            if sorted(f["context"]["query"] for f in queries) != list(range(result["policy_queries"])):
                raise ValueError("Replay queries are duplicated or missing")
            for frame in queries:
                context, q = frame["context"], frame["context"]["query"]
                if (context["trace_sha256"] != hashes[f"{episode}/trace.npz"]
                        or context["control_step"] != int(trace["query_steps"][q])
                        or frame["seed"] != int(trace["query_seeds"][q])
                        or context["image_sha256"] != str(trace["image_sha256"][q])
                        or context["recorded_success"] != result["success"]):
                    raise ValueError("Replay query identity disagrees with video source")
            timings = [{"query": q, "control_step": int(step),
                        "call_seconds": float(trace["query_seconds"][q]),
                        "policy_seconds": float(trace["policy_seconds"][q])}
                       for q, step in enumerate(trace["query_steps"])]
            if any(not np.isfinite(t["call_seconds"]) or t["call_seconds"] < 0
                   or not np.isfinite(t["policy_seconds"]) or t["policy_seconds"] < 0
                   or t["call_seconds"] < t["policy_seconds"] for t in timings):
                raise ValueError("Invalid recorded inference timings")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=avg_frame_rate,nb_read_frames,width,height",
            "-of", "json", str(folder / "rollout.mp4")], text=True))["streams"][0]
        if (Fraction(probe["avg_frame_rate"]) != 20
                or int(probe["nb_read_frames"]) != result["policy_steps"] + 1):
            raise ValueError("Video must contain the initial frame and one frame per control at 20fps")
        digest = hashes[f"{episode}/rollout.mp4"]
        entries.append({"id": digest, "episode": episode, "file": f"videos/{digest}.mp4",
                        "sha256": digest, "trace_sha256": hashes[f"{episode}/trace.npz"],
                        "fps": 20, "frames": int(probe["nb_read_frames"]),
                        "query_timing": timings,
                        "width": probe["width"], "height": probe["height"], "timebase": "simulation"})
    # Validate all inputs before publishing a manifest. Existing frames stay unchanged.
    prior = video_entries(directory)
    merged = {entry["episode"]: entry for entry in prior}
    for entry in entries:
        (directory / "videos").mkdir(exist_ok=True)
        target = directory / entry["file"]
        if target.exists() and sha(target) != entry["sha256"]:
            raise ValueError("Existing replay video differs from its SHA")
        if not target.exists():
            shutil.copyfile(results / entry["episode"] / "rollout.mp4", target)
        if sha(target) != entry["sha256"]:
            raise ValueError("Copied video SHA mismatch")
        merged[entry["episode"]] = entry
    temporary = directory / "videos.json.tmp"
    temporary.write_text(json.dumps({"videos": list(merged.values())}, indent=2) + "\n")
    temporary.replace(directory / "videos.json")
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("session", "results", "audit"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    args = parser.parse_args()
    entries = attach_videos(args.session, args.results, args.audit, args.episodes)
    print(json.dumps({"videos": entries, "new_policy_calls": 0}))


if __name__ == "__main__":
    main()

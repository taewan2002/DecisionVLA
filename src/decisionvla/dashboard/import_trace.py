"""Import verified LIBERO query images and probabilities as an explicit replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..confidence import summarize
from .telemetry import SessionWriter


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["observations", "results", "metadata", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    args = parser.parse_args()
    prepared = json.loads((args.observations / "completion.json").read_text())
    if not prepared.get("all_query_images_exact") or prepared["source_contract_sha256"] != sha(
            args.results / "contract.json"):
        raise ValueError("Query image provenance disagrees with rollout contract")
    metadata = json.loads(args.metadata.read_text())
    metadata.update(source="recorded_inference", contract_sha256=sha(args.results / "contract.json"),
                    observation_manifest_sha256=sha(args.observations / "records.json"))
    writer = SessionWriter(args.output, metadata)
    records = json.loads((args.observations / "records.json").read_text())
    cache, found = {}, set()
    for row in records:
        episode = f'task-{row["task"]:02d}-init-{row["initial"]:02d}'
        if episode not in args.episodes:
            continue
        found.add(episode)
        if episode not in cache:
            folder = args.results / episode
            if sha(folder / "trace.npz") != row["trace_sha256"]:
                raise ValueError("Trace SHA mismatch")
            with np.load(folder / "trace.npz", allow_pickle=False) as archive:
                trace = {key: archive[key] for key in archive.files}
            result = json.loads((folder / "result.json").read_text())
            cache[episode] = trace, result
        trace, result = cache[episode]
        image_path = (args.observations / row["file"]).resolve()
        if not image_path.is_relative_to(args.observations.resolve()):
            raise ValueError("Image path outside observations")
        image = Image.open(image_path).convert("RGB")
        if hashlib.sha256(np.asarray(image).tobytes()).hexdigest() != row["image_sha256"]:
            raise ValueError("Observation SHA mismatch")
        q = row["query"]
        if (int(trace["query_seeds"][q]) != row["seed"] or
                str(trace["image_sha256"][q]) != row["image_sha256"] or
                result["instruction"] != row["instruction"]):
            raise ValueError("Query identity disagrees with the recorded trace")
        diagnostics = summarize(trace["probabilities"][q])
        np.testing.assert_allclose(diagnostics["chunk_confidence"], trace["chunk_confidence"][q],
                                   atol=2e-6, rtol=0)
        writer.write(image, row["instruction"], {"actions": trace["chunks"][q], "diagnostics": diagnostics},
                     float(trace["policy_seconds"][q]), seed=row["seed"], steps=4,
                     context={"episode": episode, "query": q, "control_step": int(trace["query_steps"][q]),
                              "recorded_success": bool(result["success"]), "trace_sha256": row["trace_sha256"],
                              "image_sha256": row["image_sha256"]})
    if found != set(args.episodes):
        raise ValueError("Some requested episodes were not found")
    print(json.dumps({"queries": writer.count, "episodes": sorted(found), "new_policy_calls": 0}))


if __name__ == "__main__":
    main()

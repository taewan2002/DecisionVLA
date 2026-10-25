import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from decisionvla.dashboard.import_trace import sha
from decisionvla.dashboard.server import make_server
from decisionvla.dashboard.telemetry import SessionWriter
from decisionvla.dashboard.video import attach_videos, byte_range
from test_dashboard import prediction


@pytest.mark.parametrize(("header", "expected"), [
    (None, (0, 9)), ("bytes=0-1", (0, 1)), ("bytes=5-", (5, 9)),
    ("bytes=-3", (7, 9)), ("bytes=2-999", (2, 9)), ("bytes=-999", (0, 9)),
])
def test_browser_byte_ranges(header, expected):
    assert byte_range(header, 10) == expected


@pytest.mark.parametrize("header", ["bytes=10-", "bytes=3-2", "bytes=-0", "bytes=-",
                                   "bytes=0-1,3-4", "chars=0-1"])
def test_unsatisfiable_or_unsupported_ranges(header):
    with pytest.raises(ValueError):
        byte_range(header, 10)


def test_video_stream_ranges_head_and_private_path_boundary(tmp_path):
    folder = tmp_path / "session"
    SessionWriter(folder, {})
    (folder / "video.mp4").write_bytes(b"0123456789")
    entry = {"id": "a" * 64, "file": "video.mp4", "episode": "test", "fps": 20,
             "timebase": "simulation"}
    manifest = folder / "videos.json"
    manifest.write_text(json.dumps({"videos": [entry]}))
    server = make_server(folder, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f"http://127.0.0.1:{server.server_port}"
    url = root + "/api/video?id=" + entry["id"]
    try:
        with urlopen(root + "/api/state") as response:
            assert "file" not in json.load(response)["videos"][0]
        with urlopen(Request(url, headers={"Range": "bytes=2-5"})) as response:
            assert response.status == 206 and response.read() == b"2345"
            assert response.headers["Content-Range"] == "bytes 2-5/10"
            assert response.headers["Content-Type"] == "video/mp4"
        with urlopen(Request(url, method="HEAD")) as response:
            assert response.read() == b"" and response.headers["Content-Length"] == "10"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={"Range": "bytes=99-"}))
        assert error.value.code == 416
        with pytest.raises(HTTPError) as error:
            urlopen(root + "/api/video?id=../video.mp4")
        assert error.value.code == 404
        private = tmp_path / "private.mp4"
        private.write_bytes(b"private")
        (folder / "link.mp4").symlink_to(private)
        entry["file"] = "link.mp4"
        manifest.write_text(json.dumps({"videos": [entry]}))
        with pytest.raises(HTTPError) as error:
            urlopen(url)
        assert error.value.code == 422
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_video_attachment_requires_same_audited_query_and_frame_count(tmp_path, monkeypatch):
    results, session = tmp_path / "rollout", tmp_path / "session"
    folder = results / "task-00-init-00"
    folder.mkdir(parents=True)
    (results / "contract.json").write_text("{}")
    (folder / "rollout.mp4").write_bytes(b"synthetic test fixture")
    (folder / "result.json").write_text(json.dumps({"policy_steps": 1, "policy_queries": 1,
                                                   "success": False}))
    np.savez(folder / "trace.npz", actions=np.zeros((1, 7)), query_steps=[0], query_seeds=[42],
             query_seconds=[.12], policy_seconds=[.1],
             image_sha256=["test-image"])
    hashes = {f"{folder.name}/{p.name}": sha(p) for p in folder.iterdir()}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"artifacts_sha256": hashes}))
    writer = SessionWriter(session, {"contract_sha256": sha(results / "contract.json")})
    writer.write(Image.new("RGB", (128, 128)), "test", prediction(), .1, seed=42, context={
        "episode": folder.name, "query": 0, "control_step": 0, "recorded_success": False,
        "trace_sha256": sha(folder / "trace.npz"), "image_sha256": "test-image"})
    frame_path = session / "frames/000000.json"
    original = frame_path.read_bytes()
    probe = {"streams": [{"avg_frame_rate": "20/1", "nb_read_frames": "3",
                          "width": 128, "height": 128}]}
    monkeypatch.setattr("decisionvla.dashboard.video.subprocess.check_output",
                        lambda *args, **kwargs: json.dumps(probe))
    with pytest.raises(ValueError, match="one frame per control"):
        attach_videos(session, results, audit, [folder.name])
    assert not (session / "videos.json").exists()
    probe["streams"][0]["nb_read_frames"] = "2"
    altered = json.loads(original)
    altered["context"]["control_step"] = 1
    frame_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="query identity"):
        attach_videos(session, results, audit, [folder.name])
    frame_path.write_bytes(original)
    entries = attach_videos(session, results, audit, [folder.name])
    assert sha(session / entries[0]["file"]) == hashes[f"{folder.name}/rollout.mp4"]
    assert frame_path.read_bytes() == original
    (folder / "rollout.mp4").write_bytes(b"different video")
    with pytest.raises(ValueError, match="SHA mismatch"):
        attach_videos(session, results, audit, [folder.name])

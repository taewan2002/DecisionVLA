import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from PIL import Image

from decisionvla.codec import ActionCodec
from decisionvla.confidence import summarize
from decisionvla.dashboard.server import make_server
from decisionvla.dashboard.telemetry import ObservedPolicy, Recorder, SessionWriter


def prediction():
    # A deliberately synthetic fixture, never a measured policy result.
    bins = np.full((8, 7), 128)
    bins[:, 6] = 0
    return {"actions": ActionCodec(256).decode(bins), "diagnostics": summarize(np.eye(256)[bins])}


def test_observer_calls_once_preserves_return_and_rng(tmp_path):
    class Policy:
        calls = 0

        def predict_with_confidence(self, image, instruction, steps, seed):
            self.calls += 1
            self.value = prediction()
            return self.value

    policy = Policy()
    image = Image.new("RGB", (128, 128))
    before = np.random.get_state()
    with Recorder(tmp_path / "live", {"source": "synthetic_test"}) as recorder:
        wrapped = ObservedPolicy(policy, recorder)
        value = wrapped.predict_with_confidence(image, "test", 4, 123)
        assert value is policy.value and policy.calls == 1
        # Caller mutation after return cannot corrupt the queued snapshot.
        value["diagnostics"]["probabilities"][0][0][128] = 0
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    saved = json.loads((tmp_path / "live/frames/000000.json").read_text())
    assert saved["diagnostics"]["probabilities"][0][0][128] == 1
    assert saved["seed"] == 123
    assert recorder.errors == 0


def test_rpc_result_keeps_server_latency(tmp_path):
    class Client:
        def predict_with_confidence(self, *args, **kwargs):
            p = prediction()
            self.value = p["actions"], .188, p["diagnostics"]
            return self.value

    client = Client()
    with Recorder(tmp_path / "rpc", {}) as recorder:
        result = ObservedPolicy(client, recorder).predict_with_confidence(Image.new("RGB", (128, 128)), "test")
        assert result is client.value
    saved = json.loads((tmp_path / "rpc/frames/000000.json").read_text())
    assert saved["policy_seconds"] == .188


def test_invalid_probabilities_never_appear_in_viewer(tmp_path):
    value = prediction()
    value["diagnostics"]["probabilities"][0][0][128] = .5
    with Recorder(tmp_path / "invalid", {}) as recorder:
        recorder.submit(Image.new("RGB", (128, 128)), "test", value, .1)
    index = json.loads((tmp_path / "invalid/index.json").read_text())
    assert index["frames"] == [] and recorder.errors == 1
    assert "normalization" in recorder.last_error


def test_slow_writer_drops_telemetry_without_blocking_predictions(tmp_path):
    recorder = Recorder(tmp_path / "bounded", {}, queue_size=1)
    entered, release = threading.Event(), threading.Event()
    original = recorder.writer.write

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    recorder.writer.write = slow
    image = Image.new("RGB", (128, 128))
    try:
        assert recorder.submit(image, "one", prediction(), .1)
        assert entered.wait(2)
        assert recorder.submit(image, "two", prediction(), .1)
        assert not recorder.submit(image, "dropped", prediction(), .1)
        assert recorder.dropped == 1
    finally:
        release.set()
        recorder.close()
    assert recorder.writer.count == 2 and recorder.errors == 0


def test_http_live_updates_read_only_routes_and_private_path_boundary(tmp_path):
    folder = tmp_path / "session"
    writer = SessionWriter(folder, {"source": "synthetic_test"})
    server = make_server(folder, port=0, mode="live")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(root + path, timeout=3) as response:
            return json.load(response)

    try:
        state = get("/api/state")
        assert state["mode"] == "live" and state["total"] == 0
        writer.write(Image.new("RGB", (128, 128)), "test", prediction(), .1)
        assert get("/api/state")["total"] == 1
        assert get("/api/state")["outcomes"] == {}
        (folder / "episodes.json").write_text(json.dumps({"omy-lift": {"success": False}}))
        assert get("/api/state")["outcomes"]["omy-lift"]["success"] is False
        assert get("/api/frame?id=0")["diagnostics"]["calibrated"] is False
        with urlopen(root, timeout=3) as response:
            assert b"DecisionVLA" in response.read()
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        for path in ["/session.json", "/../session.json", "/api/frame?id=../../session"]:
            with pytest.raises(HTTPError):
                get(path)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(root + "/api/frame", data=b"{}", method="POST"), timeout=3)
        assert error.value.code == 501
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_existing_session_cannot_be_overwritten(tmp_path):
    SessionWriter(tmp_path / "session", {})
    with pytest.raises(FileExistsError):
        SessionWriter(tmp_path / "session", {})

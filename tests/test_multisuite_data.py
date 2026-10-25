"""Guard against publishing a four-suite run that silently loads only Spatial."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from decisionvla.data import LiberoDataset, dataset_files, task_instruction
from decisionvla.continuation import validate_resume_config


def make_task(root, suite, name="test_task"):
    path = root / suite / f"{name}_demo.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in (0, 9):
            demo = data.create_group(f"demo_{i}")
            actions = np.zeros((4, 7), dtype=np.float32)
            actions[:, 0] = np.arange(4) / 10
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("obs/agentview_rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8))


def config():
    return json.loads(Path("configs/libero_spatial.json").read_text())


def test_requires_every_suite_and_exact_task_set(tmp_path):
    cfg = config()
    cfg.pop("suite")
    cfg.update(suites=["libero_spatial", "libero_object"],
               expected_tasks={s: ["test_task"] for s in ("libero_spatial", "libero_object")})
    make_task(tmp_path, "libero_spatial")
    with pytest.raises(ValueError, match="Missing dataset suite"):
        LiberoDataset(tmp_path, cfg)
    make_task(tmp_path, "libero_object")
    train, val = LiberoDataset(tmp_path, cfg), LiberoDataset(tmp_path, cfg, "validation")
    assert len(train) == len(val) == 6
    assert train[0]["key"].startswith("libero_spatial/")
    assert train[3]["key"].startswith("libero_object/")
    np.testing.assert_allclose(train[3]["actions"][:3, 0], [0.1, 0.2, 0.3])
    assert {e[0:2] for e in train.episodes}.isdisjoint({e[0:2] for e in val.episodes})
    assert {e["file"].split("/")[0] for e in train.manifest()["episodes"]} == set(cfg["suites"])
    train.close()
    val.close()
    make_task(tmp_path, "libero_object", "unexpected_task")
    with pytest.raises(ValueError, match="Task manifest mismatch"):
        dataset_files(tmp_path, cfg)


def test_legacy_spatial_identity_and_demo_coverage(tmp_path):
    cfg = config()
    make_task(tmp_path, cfg["suite"])
    train = LiberoDataset(tmp_path, cfg)
    assert train[0]["key"] == "test_task_demo.hdf5/demo_0/0"
    assert train[0]["task"] == "test task"
    episodes = [{"file": "test_task_demo.hdf5", "demo": "demo_0", "frames": 4}]
    assert train.manifest()["sha256"] == hashlib.sha256(
        json.dumps(episodes, sort_keys=True).encode()).hexdigest()
    train.close()
    with pytest.raises(ValueError, match="Demonstration manifest mismatch"):
        LiberoDataset(tmp_path, {**cfg, "expected_demonstrations_per_task": 50})


@pytest.mark.parametrize("name,expected", [
    ("pick_up_the_black_bowl_demo.hdf5", "pick up the black bowl"),
    ("KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5", "turn on the stove"),
    ("KITCHEN_SCENE10_close_the_drawer_demo.hdf5", "close the drawer"),
    ("LIVING_ROOM_SCENE2_put_the_bowl_demo.hdf5", "put the bowl"),
])
def test_official_language(name, expected):
    assert task_instruction(Path(name)) == expected


@pytest.mark.parametrize("suites", [[], ["libero_spatial", "libero_spatial"], ["../data"], [None]])
def test_invalid_suite_scope(tmp_path, suites):
    with pytest.raises(ValueError):
        dataset_files(tmp_path, {"suites": suites})


def test_candidate_covers_forty_tasks_and_cannot_silently_resume_spatial():
    candidate = json.loads(Path("configs/libero_40_r32_b64_100k_candidate.json").read_text())
    assert candidate["suites"] == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    assert "suite" not in candidate
    assert set(candidate["expected_tasks"]) == set(candidate["suites"])
    assert all(len(set(names)) == len(names) == 10 for names in candidate["expected_tasks"].values())
    parent = json.loads(Path("configs/libero_spatial_r32_b64_100k_cublas.json").read_text())
    with pytest.raises(ValueError, match="identical config"):
        validate_resume_config({**parent, "step": 6500}, candidate)


def test_balanced_sampling_and_validation_reproducibility(tmp_path):
    cfg = config()
    cfg.pop("suite")
    cfg.update(suites=["libero_spatial", "libero_object"], sampling="uniform_task",
               validation_samples=4, validation_samples_per_task=2)
    for suite in cfg["suites"]:
        make_task(tmp_path, suite)
    data = LiberoDataset(tmp_path, cfg)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    indices = data.sample_indices(a, 6400)
    np.testing.assert_array_equal(indices, data.sample_indices(b, 6400))
    assert 3000 < int((indices < 3).sum()) < 3400
    assert a.bit_generator.state == b.bit_generator.state
    held = data.validation_indices()
    assert len(set(held.tolist())) == 4 and (held < 3).sum() == 2
    np.testing.assert_array_equal(held, data.validation_indices())
    state = a.bit_generator.state
    next_indices = data.sample_indices(a, 64)
    b.bit_generator.state = state
    np.testing.assert_array_equal(next_indices, data.sample_indices(b, 64))
    data.close()

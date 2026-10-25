import importlib.util
from pathlib import Path


def test_publication_requires_a_whole_task_with_leader_margin():
    spec = importlib.util.spec_from_file_location(
        "assist_evaluation", Path(__file__).parents[1] / "scripts/assist_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"task_ids": list(range(10)), "initial_state_ids": list(range(50))}
    assert module.publication_safe(config, {"status": "running", "completed_episodes": 395}, 8)
    assert not module.publication_safe(config, {"status": "running", "completed_episodes": 396}, 8)
    assert not module.publication_safe(config, {"status": "running", "completed_episodes": 401}, 8)
    assert not module.publication_safe(config, {"status": "error", "completed_episodes": 300}, 8)

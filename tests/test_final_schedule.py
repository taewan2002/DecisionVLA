import copy
import json
from pathlib import Path

import pytest

from decisionvla.continuation import learning_rate_at, training_boundary
from decisionvla.schedule import select_checkpoint, write_selection


def plan():
    return json.loads(Path('configs/training_schedule_b64_100k_finalonly.json').read_text())


def report(step=100000):
    return {'status': 'completed', 'episodes': 500, 'successes': 0,
            'protocol': {'checkpoint_step': step},
            'tasks': [{'task_id': i, 'episodes': 50, 'failures': 50} for i in range(10)],
            'primary_within_task_auroc': None, 'primary_ci95': [None, None]}


def test_final_checkpoint_is_fixed_even_with_zero_successes(tmp_path):
    result = write_selection(plan(), {100000: report()}, tmp_path)
    assert result['selected_step'] == 100000
    assert result['independent_test'] is False
    assert result['checkpoints'][0]['within_task_failure_auroc'] is None
    assert (tmp_path / 'final-evaluation.csv').exists()
    assert not (tmp_path / 'learning-curve.csv').exists()


@pytest.mark.parametrize('reports', [{}, {10000: report(10000)},
                                   {10000: report(10000), 100000: report()},
                                   {100000: {**report(), 'episodes': 499}}])
def test_missing_extra_or_incomplete_evaluations_rejected(reports):
    with pytest.raises(ValueError):
        select_checkpoint(plan(), reports)


def test_fixed_final_plan_rejects_posthoc_checkpoint_search():
    changed = copy.deepcopy(plan())
    changed['evaluation_steps'] = [10000, 100000]
    changed['selection']['candidate_steps'] = [10000, 100000]
    with pytest.raises(ValueError, match='Fixed-final'):
        select_checkpoint(changed, {10000: report(10000), 100000: report()})


def test_execution_segments_keep_original_lr_and_final_protocol():
    current = plan()
    previous = json.loads(Path('configs/training_schedule_b64_100k_cublas.json').read_text())
    assert current['training_config'] == previous['training_config']
    for key in ('total_steps', 'checkpoint_every', 'validation_every', 'validation_samples',
                'gpu_count', 'global_batch_size', 'same_seed', 'episodes_per_checkpoint'):
        assert current[key] == previous[key]
    config = json.loads(Path(current['training_config']).read_text())
    steps = []
    first = 0
    for stop in current['training_segment_steps']:
        end = training_boundary(config, first, stop)
        steps.extend(range(first + 1, end + 1))
        first = end
    assert steps == list(range(1, 100001))
    assert [learning_rate_at(s, config) for s in steps] == [
        learning_rate_at(s, config) for s in range(1, 100001)]
    assert current['evaluation_steps'] == [100000]
    assert current['selection']['fixed_step'] == 100000

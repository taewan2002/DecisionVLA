import copy
import itertools
import json
from pathlib import Path

import pytest

from decisionvla.continuation import learning_rate_at, training_boundary, validate_extension


def test_50k_schedule_continues_across_all_evaluation_boundaries():
    config = json.loads(Path('configs/libero_spatial_r32_corr_50k.json').read_text())
    previous = json.loads(Path('configs/libero_spatial_r32_corr_10k.json').read_text())
    validate_extension({**previous, 'step': 10000}, config)
    original = copy.deepcopy(config)
    segmented = []
    first = 10000
    for stop in (20000, 30000, 40000, 50000):
        end = training_boundary(config, first, stop)
        segmented.extend(learning_rate_at(step, config) for step in range(first + 1, end + 1))
        first = end
    assert segmented == [learning_rate_at(step, config) for step in range(10001, 50001)]
    assert config == original
    assert segmented[0] == pytest.approx(5.09e-6)
    assert segmented[499] == pytest.approx(5e-5)
    assert segmented[-1] == pytest.approx(5e-6)
    assert all(a >= b for a, b in itertools.pairwise(segmented[499:]))
    for stop in (10000, 50001):
        with pytest.raises(ValueError):
            training_boundary(config, 10000, stop)


def test_five_rollouts_preserve_protocol_except_checkpoint_identity():
    folder = Path('configs')
    plan = json.loads((folder / 'training_schedule_50k.json').read_text())
    original = json.loads((folder / 'eval_libero_spatial_confidence_10k.json').read_text())
    for step in plan['evaluation_steps']:
        rollout = json.loads((folder / f'eval_libero_spatial_confidence_{step // 1000}k.json').read_text())
        analysis = json.loads((folder / f'confidence_analysis_{step // 1000}k.json').read_text())
        assert rollout == {**original, 'expected_checkpoint_step': step}
        assert analysis['checkpoint_step'] == step
        assert len(rollout['task_ids']) * len(rollout['initial_state_ids']) == analysis['episodes'] == 500
    assert plan['selection']['candidate_steps'] == plan['evaluation_steps']
    assert plan['selection']['tie_break'] == 'earliest_step'


def test_selection_requires_all_evaluations_and_uses_earliest_tie():
    from decisionvla.schedule import select_checkpoint
    plan=json.loads(Path('configs/training_schedule_50k.json').read_text())
    reports={}
    for step, successes in zip(plan['evaluation_steps'], (14, 40, 50, 50, 45)):
        failures=[50]*10
        for i in range(successes):
            failures[i % 10] -= 1
        reports[step]={'status':'completed','episodes':500,'successes':successes,
            'protocol':{'checkpoint_step':step},
            'tasks':[{'task_id':i,'episodes':50,'failures':f} for i,f in enumerate(failures)],
            'primary_within_task_auroc':None,'primary_ci95':[None,None]}
    result=select_checkpoint(plan,reports)
    assert result['selected_step'] == 30000
    # Equal successes distributed differently across tasks must still be an exact tie.
    reports[40000]['tasks'] = [{'task_id':i,'episodes':50,'failures':0 if i == 0 else 50}
                              for i in range(10)]
    assert select_checkpoint(plan,reports)['selected_step'] == 30000
    assert result['independent_test'] is False
    with pytest.raises(ValueError,match='Every planned'):
        select_checkpoint(plan,{k:v for k,v in reports.items() if k != 50000})
    reports[50000]['episodes']=499
    with pytest.raises(ValueError,match='Incomplete'):
        select_checkpoint(plan,reports)

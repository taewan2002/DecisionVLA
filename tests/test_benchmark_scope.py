import json
from pathlib import Path

import numpy as np
import pytest

from decisionvla.analyze_confidence import pair_counts
from decisionvla.benchmark import base_task_index, physical_bddl, task_config


def test_fixed_scope_and_suite_horizons():
    p = json.loads(Path('configs/eval_libero40_plus_100k.json').read_text())
    assert sum(p['original']['task_counts'].values()) * 50 == 2000
    assert sum(p['plus']['task_counts'].values()) == 10030
    for suite, horizon in zip(p['suites'], [220, 280, 300, 520]):
        c = task_config(p, 'original', suite, 3, 100000, False)
        assert c['max_policy_steps'] == horizon and len(c['initial_state_ids']) == 50
        assert task_config(p, 'plus', suite, 3, 100000, False)['initial_state_ids'] == [0]
    assert task_config(p, 'plus', p['suites'][0], 0, 3, True)['max_policy_steps'] == 8


def test_virtual_path_and_task_mapping():
    assert base_task_index('pick_a_table_1', ['pick', 'pick_a']) == 1
    with pytest.raises(ValueError):
        base_task_index('unseen', ['pick'])
    assert str(physical_bddl('/tasks/pick_view_30_0_100_0_0_initstate_0_noise_10.bddl')) == '/tasks/pick.bddl'
    assert str(physical_bddl('/tasks/pick_table_1.bddl')) == '/tasks/pick_table_1.bddl'


def test_sorted_auc_exactly_matches_pairwise_reference_with_ties():
    rng = np.random.default_rng(21)
    for n in [1, 10, 1000]:
        scores = rng.integers(0, 20, n) / 20
        labels = rng.integers(0, 2, n).astype(bool)
        differences = scores[labels, None] - scores[~labels][None, :]
        expected = (float((differences > 0).sum() + .5 * (differences == 0).sum()), differences.size)
        assert pair_counts(labels, scores) == expected


def test_nonspatial_probability_audit_and_horizon_guard(tmp_path):
    from decisionvla.analyze_confidence import audit_results
    from decisionvla.codec import ActionCodec
    from decisionvla.confidence import METRICS, SCHEMA, summarize
    from decisionvla.rollout import digest
    probs = np.zeros((8, 7, 256), dtype=np.float64)
    probs[:, :6, 128] = 1
    probs[:, 6, 0] = 1
    stats = summarize(probs)
    actions = ActionCodec().decode(np.asarray(stats['action_bins']))
    episode = tmp_path / 'task-00-init-00'
    episode.mkdir()
    trace = dict(probabilities=probs[None], action_bins=np.asarray(stats['action_bins'])[None],
                 actions=actions[:2], chunks=actions[None], states=np.zeros((3, 4)),
                 query_steps=np.array([0]), chunk_confidence=np.array([stats['chunk_confidence']]))
    trace.update({k: np.asarray(stats['metrics'][k])[None] for k in METRICS})
    np.savez_compressed(episode / 'trace.npz', **trace)
    result = dict(task_id=0, initial_state_id=0, success=False, confidence_schema=SCHEMA,
                  policy_queries=1, policy_steps=2, artifacts_sha256={'trace.npz': digest(episode / 'trace.npz')})
    (episode / 'result.json').write_text(json.dumps(result))
    (tmp_path / 'contract.json').write_text(json.dumps(dict(kind='closed_loop_evaluation',
        config=dict(suite='libero_goal', record_confidence=True, max_policy_steps=300,
                    task_ids=[0], initial_state_ids=[0]), policy={'step': 100000})))
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='completed', episodes=1, successes=0)))
    protocol = dict(checkpoint_step=100000, suite='libero_goal', episodes=1, max_policy_steps=300)
    rows, hashes = audit_results(tmp_path, protocol)
    assert rows[0]['failure'] == 1 and len(hashes) == 2
    with pytest.raises(AssertionError):
        audit_results(tmp_path, {**protocol, 'max_policy_steps': 220})

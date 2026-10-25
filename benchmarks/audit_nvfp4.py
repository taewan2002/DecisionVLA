"""CPU-only checks of fixed-input action fidelity and paired rollout artifacts.

This is a development comparison of a frozen checkpoint, not an equivalence
or noninferiority test. Changes include runtime, merge, quantization and kernels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def validate_arrays(actions, probabilities):
    assert actions.shape == (8, 7) and probabilities.shape == (8, 7, 256)
    assert np.isfinite(actions).all() and np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-6, rtol=0)
    assert (probabilities[:, 6, 1:-1] == 0).all()
    decoded = (2 * probabilities.argmax(-1) / 255 - 1).astype(np.float32)
    np.testing.assert_allclose(actions, decoded, atol=2e-7, rtol=0)


def fixed_comparison(reference, candidate):
    rows = []
    with np.load(reference, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as b:
        for i in range(10):
            suffix = f'{i}_rep0' if f'actions_{i}_rep0' in b else str(i)
            x, p = a[f'actions_{i}'], a[f'probabilities_{i}']
            y, q = b[f'actions_{suffix}'], b[f'probabilities_{suffix}']
            validate_arrays(x, p)
            validate_arrays(y, q)
            if f'actions_{i}_rep1' in b:
                np.testing.assert_array_equal(y, b[f'actions_{i}_rep1'])
                np.testing.assert_array_equal(q, b[f'probabilities_{i}_rep1'])
            rows.append({
                'observation': i,
                'continuous_action_mae': float(np.abs(x[:, :6] - y[:, :6]).mean()),
                'max_action_difference': float(np.abs(x - y).max()),
                'gripper_agreements': int((x[:, 6] == y[:, 6]).sum()),
                'probability_mean_tv': float(np.abs(p - q).sum(-1).mean() / 2),
            })
    return {
        'observations': 10, 'continuous_actions': 480, 'gripper_actions': 80,
        'continuous_action_mae': float(np.mean([r['continuous_action_mae'] for r in rows])),
        'max_action_difference': max(r['max_action_difference'] for r in rows),
        'gripper_agreements': sum(r['gripper_agreements'] for r in rows),
        'probability_mean_tv': float(np.mean([r['probability_mean_tv'] for r in rows])),
        'rows': rows, 'reference_sha256': digest(reference), 'candidate_sha256': digest(candidate),
        'interpretation': 'Total deployment-path difference; not quantization-only error or a success-rate estimate',
    }


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center-radius, center+radius]


def paired_rollouts(reference, candidate):
    from decisionvla.analyze_confidence import audit_results

    protocol = {'checkpoint_step': 4500, 'suite': 'libero_spatial', 'episodes': 20}
    a, ah = audit_results(reference, protocol)
    b, bh = audit_results(candidate, protocol)
    ac = json.loads((Path(reference)/'contract.json').read_text())
    bc = json.loads((Path(candidate)/'contract.json').read_text())
    assert {k: v for k, v in ac['config'].items() if k != 'expected_policy_kind'} == {
        k: v for k, v in bc['config'].items() if k != 'expected_policy_kind'}
    assert ac['policy']['checkpoint_sha256'] == bc['policy']['checkpoint_sha256']
    assert ac['runtime'] == bc['runtime'] and ac['tasks'] == bc['tasks']
    pairs = []
    for x, y in zip(a, b, strict=True):
        task, initial = x['task_id'], x['initial_state_id']
        assert (task, initial) == (y['task_id'], y['initial_state_id'])
        sub = f'task-{task:02d}-init-{initial:02d}'
        with np.load(Path(reference)/sub/'trace.npz') as ta, np.load(Path(candidate)/sub/'trace.npz') as tb:
            np.testing.assert_array_equal(ta['initial_state'], tb['initial_state'])
            np.testing.assert_array_equal(ta['states'][0], tb['states'][0])
            assert ta['image_sha256'][0] == tb['image_sha256'][0]
            assert ta['query_seeds'][0] == tb['query_seeds'][0]
        pairs.append({'task_id': task, 'initial_state_id': initial,
                      'reference_success': 1-x['failure'], 'candidate_success': 1-y['failure']})
    original = sum(r['reference_success'] for r in pairs)
    quant = sum(r['candidate_success'] for r in pairs)
    changes = np.array([r['candidate_success']-r['reference_success'] for r in pairs]).reshape(10, 2)
    # Descriptive paired hierarchical bootstrap, task then initial state;
    # chosen before complete paired outcomes. No adoption threshold is inferred.
    rng = np.random.default_rng(20261025)
    boot = []
    for _ in range(20000):
        tasks = rng.integers(0, 10, 10)
        initials = rng.integers(0, 2, (10, 2))
        boot.append(changes[tasks[:, None], initials].mean())
    return {
        'status': 'audited_complete', 'episodes_per_arm': 20,
        'reference_successes': original, 'candidate_successes': quant,
        'reference_wilson_ci95': wilson(original, 20), 'candidate_wilson_ci95': wilson(quant, 20),
        'paired_success_rate_difference': (quant-original)/20,
        'paired_hierarchical_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
        'recovered': int((changes == 1).sum()), 'lost': int((changes == -1).sum()),
        'pairs': pairs, 'reference_artifacts_sha256': ah, 'candidate_artifacts_sha256': bh,
        'reference_contract_sha256': digest(Path(reference)/'contract.json'),
        'candidate_contract_sha256': digest(Path(candidate)/'contract.json'),
        'interpretation': '20 reused development initial states, one checkpoint; neither quality equivalence nor superiority is established',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-fixed', type=Path, required=True)
    parser.add_argument('--candidate-fixed', type=Path, required=True)
    parser.add_argument('--reference-rollouts', type=Path)
    parser.add_argument('--candidate-rollouts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'fixed_inputs': fixed_comparison(args.reference_fixed, args.candidate_fixed)}
    if args.reference_rollouts or args.candidate_rollouts:
        assert args.reference_rollouts and args.candidate_rollouts
        result['paired_rollouts'] = paired_rollouts(args.reference_rollouts, args.candidate_rollouts)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()

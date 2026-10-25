"""Independently connect graph-output hashes to audited eager rollout traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_nvfp4 import digest, validate_arrays


def main():
    parser = argparse.ArgumentParser()
    for name in ['results', 'replay', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.replay.read_text())
    assert report['status'] == 'completed'
    assert report['contract_sha256'] == digest(args.results/'contract.json')
    rows = {(r['task'], r['initial'], r['query']): r for r in report['rows']}
    assert len(rows) == len(report['rows']) == report['audited_queries'] == report['expected_queries']
    checked = 0
    for folder in sorted(args.results.glob('task-*-init-*')):
        result = json.loads((folder/'result.json').read_text())
        task, initial = result['task_id'], result['initial_state_id']
        trace_sha = digest(folder/'trace.npz')
        assert trace_sha == result['artifacts_sha256']['trace.npz']
        with np.load(folder/'trace.npz', allow_pickle=False) as trace:
            for q, (actions, probabilities) in enumerate(zip(trace['chunks'], trace['probabilities'], strict=True)):
                row = rows.pop((task, initial, q))
                validate_arrays(actions, probabilities)
                assert row['actions_probabilities_metrics_exact']
                assert row['trace_sha256'] == trace_sha
                assert row['image_sha256'] == trace['image_sha256'][q]
                assert row['action_sha256'] == hashlib.sha256(actions.tobytes()).hexdigest()
                assert row['probability_sha256'] == hashlib.sha256(probabilities.tobytes()).hexdigest()
                checked += 1
    assert not rows and checked == report['audited_queries']
    output = {'status': 'passed', 'mode': report['mode'], 'queries': checked,
              'distributions': checked*56, 'replay_report_sha256': digest(args.replay),
              'all_image_action_probability_hashes_match_reference': True,
              'probability_normalization_gripper_support_and_action_decode_passed': True,
              'interpretation': 'Independent trace/hash audit; these are existing episodes, not new success trials'}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()

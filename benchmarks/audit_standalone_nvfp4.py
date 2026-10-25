"""Recompute the completed 5090 experiment from private raw arrays and public reports.

Run from the repository root with PYTHONPATH=benchmarks and NumPy 1.26.4.
Raw arrays must be supplied separately; this tool does not download data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_nvfp4 import validate_arrays

ROOT = Path('reports/inference-nvfp4-5090-20261025')


def main():
    rows = []
    for label in ['w4a16-eager', 'w4a16-graph', 'w4a4-eager', 'w4a4-graph']:
        mode = label.split('-')[0]
        report_path = ROOT/'final'/(label+'-latency.json')
        raw_path = ROOT/'private'/('standalone-'+label+'.npz')
        report = json.loads(report_path.read_text())
        assert report['status'] == 'completed' and report['export_reload_exact']
        assert report['repeated_actions_and_probabilities_exact']
        times = np.asarray([r['seconds'] for r in report['rows']])
        assert len(times) == 20 and np.isfinite(times).all() and (times > 0).all()
        assert float(np.percentile(times, 50)) == report['p50_seconds']
        assert float(np.percentile(times, 95)) == report['p95_seconds']
        with np.load(raw_path, allow_pickle=False) as raw, np.load(
            ROOT/'private'/(mode+'-fixed.npz'), allow_pickle=False
        ) as fixed:
            for i in range(10):
                suffix = f'{i}_rep0' if f'actions_{i}_rep0' in fixed else str(i)
                for a, p in [(f'eager_actions_{i}', f'eager_probabilities_{i}'),
                             (f'actions_{i}_rep0', f'probabilities_{i}_rep0'),
                             (f'actions_{i}_rep1', f'probabilities_{i}_rep1')]:
                    validate_arrays(raw[a], raw[p])
                    np.testing.assert_array_equal(raw[a], fixed['actions_'+suffix])
                    np.testing.assert_array_equal(raw[p], fixed['probabilities_'+suffix])
        rows.append({'path': label, 'predictions': 30, 'distributions': 1680,
                     'all_actions_probabilities_exact_to_frozen_candidate': True,
                     'p50_seconds': report['p50_seconds'], 'p95_seconds': report['p95_seconds'],
                     'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                     'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()})
    output = {'status': 'passed', 'predictions': 120, 'distributions': 6720,
              'rows': rows, 'scope': 'Independent aggregate latency and saved-array fidelity audit; memory is runtime-reported'}
    output['auditor_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/'final/standalone-independent-audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == "__main__":
    main()

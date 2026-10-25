"""Check every pinned benchmark task mapping, BDDL and initial state on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from decisionvla.benchmark import base_task_index, load_benchmark, physical_bddl
from decisionvla.rollout import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--kind', choices=['original', 'plus'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = []
    for s in plan['suites']:
        suite, categories = load_benchmark(plan, args.kind, s)
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            bddl = physical_bddl(suite.get_task_bddl_file_path(i))
            states = np.asarray(suite.get_task_init_states(i))
            assert states.ndim == 2 and states.shape[0] >= (50 if args.kind == 'original' else 1)
            assert np.isfinite(states).all()
            rows.append({'suite': s, 'task': i, 'base_task': base_task_index(task.name, plan['original_tasks'][s]),
                         'bddl_sha256': digest(bddl), 'initial_shape': list(states.shape),
                         'category': categories.get(i)})
    write_json(args.output, {'status': 'passed', 'kind': args.kind, 'tasks': len(rows),
                            'plan_sha256': digest(args.plan), 'rows': rows})
    print(json.dumps({'status': 'passed', 'tasks': len(rows)}))


if __name__ == '__main__':
    main()

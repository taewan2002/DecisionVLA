"""CPU-only coverage and fixed validation baselines for a completed pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from decisionvla.data import LiberoDataset
from decisionvla.rollout import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'run.json').read_text())
    config = metadata['config']
    rows = [json.loads(line) for line in (args.run / 'events.jsonl').read_text().splitlines()]
    updates = [r for r in rows if r['event'] == 'update']
    keys = [k for r in updates for k in r['sample_keys']]
    lengths = {(e['file'], e['demo']): e['frames'] for e in metadata['train']['episodes']}
    targets = set()
    for key in keys:
        file, demo, start = key.split('/')
        begin = int(start) + config['action_offset']
        for index in range(begin, min(begin + config['horizon'], lengths[(file, demo)])):
            targets.add((file, demo, index))
    dataset = LiberoDataset(args.data, config, 'validation')
    indices = np.random.default_rng(9001).choice(len(dataset), config['validation_samples'], replace=False)
    actions, validation_keys = [], []
    try:
        for index in indices:
            item = dataset[int(index)]
            valid = item['valid'].reshape(-1, 7)[:, 0]
            actions.extend(item['actions'][valid])
            validation_keys.append(item['key'])
    finally:
        dataset.close()
    actions = np.asarray(actions)
    validation = [r for r in rows if r['event'].startswith('validation')]
    report = {'kind': 'posthoc_training_diagnostic', 'training_updates': len(updates),
              'anchor_presentations': len(keys), 'unique_anchor_count': len(set(keys)),
              'train_anchors_available': metadata['train']['frames'],
              'equivalent_anchor_epochs': len(keys) / metadata['train']['frames'],
              'unique_supervised_action_steps': len(targets), 'validation_samples': len(indices),
              'validation_curve': validation, 'validation_keys': validation_keys,
              'zero_continuous_action_validation_mae': float(np.abs(actions[:, :6]).mean()),
              'zero_action_mae_by_axis': np.abs(actions[:, :6]).mean(0).tolist(),
              'validation_gripper_majority_accuracy': float(max(np.mean(actions[:, 6] > 0),
                                                               np.mean(actions[:, 6] <= 0))),
              'fixed_validation_last': validation[-1],
              'source_sha256': {'events.jsonl': digest(args.run / 'events.jsonl'),
                                'run.json': digest(args.run / 'run.json')},
              'numpy_version': np.__version__}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['validation_curve', 'validation_keys']}))


if __name__ == '__main__':
    main()

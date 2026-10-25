"""Check graph outputs against every query in recorded eager LIBERO trajectories.

Images must come from replay_policy_observations.py. This is a parity audit of
existing episodes, not a new closed-loop success cohort or latency benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from latency_nvfp4 import sha
from PIL import Image

import decisionvla.model as dm
from decisionvla.checkpoints import load_adapter
from decisionvla.codec import ActionCodec
from decisionvla.confidence import METRICS, validate_diagnostics
from decisionvla.policy import DecisionVLA


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for name in ['model', 'adapter', 'packed', 'manifest', 'observations', 'results', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=['w4a16', 'w4a4'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((args.observations/'completion.json').read_text())
    assert prepared['status'] == 'completed' and prepared['all_query_images_exact']
    assert prepared['source_contract_sha256'] == sha(args.results/'contract.json')
    records = json.loads((args.observations/'records.json').read_text())
    assert len(records) == prepared['queries']
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(24*2**30/torch.cuda.get_device_properties(0).total_memory)
    policy = DecisionVLA.__new__(DecisionVLA)
    policy.config = json.loads((args.adapter/'adapter_config.json').read_text())
    assert policy.config['step'] == 4500
    policy.model, policy.processor, _ = dm.load_model(
        str(args.model), {**policy.config, 'expert_kernel': 'torch_grouped_mm'}, 'cpu')
    load_adapter(policy.model, args.adapter)
    policy.model.eval()
    policy.model.gradient_checkpointing_disable()
    manifest = json.loads(args.manifest.read_text())['export_sha256']
    if args.mode == 'w4a16':
        from resident_nvfp4 import load_packed_experts
        load_packed_experts(policy.model, args.packed, manifest)
    else:
        from resident_nvfp4_a4 import load_packed_a4
        load_packed_a4(policy.model, args.packed, manifest)
    policy.device, policy.codec = 'cuda', ActionCodec(policy.config['bins'])
    assert all(t.device.type == 'cuda' for t in [*policy.model.parameters(), *policy.model.buffers()])
    from decoder_graph import bind_policy_graphs
    bind_policy_graphs(policy, prefix=True)
    rows = []
    report = {'status': 'running', 'mode': args.mode, 'checkpoint_step': 4500,
              'records_sha256': sha(args.observations/'records.json'),
              'contract_sha256': sha(args.results/'contract.json'),
              'export_manifest_sha256': sha(args.manifest), 'expected_queries': len(records),
              'interpretation': 'Graph parity on recorded eager trajectories; not additional episodes',
              'rows': rows}
    try:
        for i, record in enumerate(records):
            folder = args.results/f'task-{record["task"]:02d}-init-{record["initial"]:02d}'
            assert sha(folder/'trace.npz') == record['trace_sha256']
            image = Image.open(args.observations/record['file']).convert('RGB')
            assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == record['image_sha256']
            out = policy.predict_with_confidence(image, record['instruction'], 4, record['seed'])
            validate_diagnostics(out['diagnostics'], out['actions'])
            p = np.asarray(out['diagnostics']['probabilities'], dtype=np.float32)
            with np.load(folder/'trace.npz', allow_pickle=False) as trace:
                q = record['query']
                np.testing.assert_array_equal(out['actions'], trace['chunks'][q])
                np.testing.assert_array_equal(p, trace['probabilities'][q])
                np.testing.assert_array_equal(out['diagnostics']['action_bins'], trace['action_bins'][q])
                for key in METRICS:
                    np.testing.assert_array_equal(
                        np.asarray(out['diagnostics']['metrics'][key], dtype=np.float32), trace[key][q])
                assert out['diagnostics']['chunk_confidence'] == trace['chunk_confidence'][q]
            rows.append({'index': i, 'task': record['task'], 'initial': record['initial'], 'query': q,
                         'action_sha256': hashlib.sha256(out['actions'].tobytes()).hexdigest(),
                         'probability_sha256': hashlib.sha256(p.tobytes()).hexdigest(),
                         'trace_sha256': record['trace_sha256'], 'image_sha256': record['image_sha256'],
                         'actions_probabilities_metrics_exact': True})
            if (i+1) % 25 == 0:
                (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps({'completed_queries': i+1, 'total': len(records)}), flush=True)
        report.update(status='completed', all_actions_probabilities_metrics_exact=True,
                      audited_queries=len(rows),
                      audited_episodes=len({(r['task'], r['initial']) for r in records}))
    except Exception as error:
        report.update(status='error', error=repr(error))
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

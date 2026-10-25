"""Measure a verified NVFP4 export on one SM120 GPU; seeded sampler is unchanged.

Run with PYTHONPATH=src:benchmarks in the isolated, pinned inference runtime.
Private model weights and observation images are supplied explicitly. Each
observation is warmed before timing; graph compilation/capture is separate.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import decisionvla.model as dm
from decisionvla.checkpoints import load_adapter
from decisionvla.codec import ActionCodec
from decisionvla.confidence import validate_diagnostics
from decisionvla.policy import DecisionVLA


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['w4a16', 'w4a4'], required=True)
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--prefix-graph', action='store_true')
    parser.add_argument('--expected-fixed', type=Path, help='Earlier frozen eager outputs to verify after reload')
    parser.add_argument('--gpu-budget-gib', type=float, default=24)
    parser.add_argument('--expected-step', type=int, default=4500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(
        args.gpu_budget_gib*2**30 / torch.cuda.get_device_properties(0).total_memory)
    policy = DecisionVLA.__new__(DecisionVLA)
    policy.config = json.loads((args.adapter/'adapter_config.json').read_text())
    assert policy.config['step'] == args.expected_step
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
    samples = json.loads((args.inputs/'samples.json').read_text())
    assert len(samples) == 10
    raw, rows, cold = {}, [], []
    report = {'status': 'running', 'mode': args.mode, 'graph': args.graph, 'prefix_graph': args.prefix_graph,
              'prefix_kv_copy_scope': 'once per observation' if args.graph else 'eager decoder reads prefix',
              'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'checkpoint_step': args.expected_step, 'manifest_sha256': sha(args.manifest),
              'samples_sha256': sha(args.inputs/'samples.json'),
              'images_sha256': {s['file']: sha(args.inputs/s['file']) for s in samples},
              'scope': 'batch1, one128x128 RGB and instruction, four denoising passes, eight7D actions plus full probability/confidence conversion; no CPU offload',
              'standalone': 'caller must verify exclusive compute on this GPU',
              'rows': rows, 'cold_or_capture_rows': cold}

    def predict(s):
        image = Image.open(args.inputs/s['file']).convert('RGB')
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = policy.predict_with_confidence(image, s['instruction'], 4, s['seed'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        validate_diagnostics(out['diagnostics'], out['actions'])
        return out['actions'], np.asarray(out['diagnostics']['probabilities']), elapsed

    def persist():
        np.savez_compressed(args.output/'raw.npz', **raw)
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    # Keep an eager result for every input before enabling graph capture.
    for i, s in enumerate(samples):
        a, p, t = predict(s)
        raw[f'eager_actions_{i}'], raw[f'eager_probabilities_{i}'] = a, p
        cold.append({'i': i, 'phase': 'eager_first', 'seconds': t})
    persist()
    if args.expected_fixed:
        with np.load(args.expected_fixed, allow_pickle=False) as expected:
            for i in range(10):
                suffix = f'{i}_rep0' if f'actions_{i}_rep0' in expected else str(i)
                np.testing.assert_array_equal(raw[f'eager_actions_{i}'], expected[f'actions_{suffix}'])
                np.testing.assert_array_equal(raw[f'eager_probabilities_{i}'], expected[f'probabilities_{suffix}'])
        report['export_reload_exact'] = True
    if args.graph:
        from decoder_graph import bind_policy_graphs
        bind_policy_graphs(policy, prefix=args.prefix_graph)
    elif args.prefix_graph:
        from prefix_graph import PrefixLanguageGraph
        model = policy.model.model.encoder.language_model
        model.forward = PrefixLanguageGraph(model)
    try:
        for i, s in enumerate(samples):
            a, p, t = predict(s)
            cold.append({'i': i, 'phase': 'graph_capture' if args.graph else 'eager_warmup', 'seconds': t})
            np.testing.assert_array_equal(a, raw[f'eager_actions_{i}'])
            np.testing.assert_array_equal(p, raw[f'eager_probabilities_{i}'])
            for rep in range(2):
                a, p, t = predict(s)
                raw[f'actions_{i}_rep{rep}'], raw[f'probabilities_{i}_rep{rep}'] = a, p
                rows.append({'i': i, 'repeat': rep, 'seconds': t, 'action_sha256': hashlib.sha256(a.tobytes()).hexdigest()})
                persist()
                np.testing.assert_array_equal(a, raw[f'eager_actions_{i}'])
                np.testing.assert_array_equal(p, raw[f'eager_probabilities_{i}'])
                print(json.dumps(rows[-1]), flush=True)
        times = [r['seconds'] for r in rows]
        report.update(status='completed', repeated_actions_and_probabilities_exact=True,
                      p50_seconds=float(np.percentile(times, 50)), p95_seconds=float(np.percentile(times, 95)),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        persist()
    except Exception as error:
        report.update(status='error', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    main()

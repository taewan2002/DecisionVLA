"""Calibrate native FP4 activation scales on a frozen training-only image list.

Collect from the stable W4A16 policy; do not use rollout outcomes or the held-out
fixed-input fidelity cohort to choose scales. The output contains no images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from resident_nvfp4 import load_packed_experts

from decisionvla.checkpoints import load_adapter
from decisionvla.codec import ActionCodec
from decisionvla.model import load_model
from decisionvla.policy import DecisionVLA


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    policy = DecisionVLA.__new__(DecisionVLA)
    policy.config = json.loads((args.adapter/'adapter_config.json').read_text())
    policy.model, policy.processor, _ = load_model(
        str(args.model), {**policy.config, 'expert_kernel': 'torch_grouped_mm'}, 'cpu')
    load_adapter(policy.model, args.adapter)
    policy.model.eval()
    policy.model.gradient_checkpointing_disable()
    load_packed_experts(policy.model, args.packed, json.loads(args.manifest.read_text())['export_sha256'])
    policy.device, policy.codec = 'cuda', ActionCodec(policy.config['bins'])
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import _get_cached_workspace
    stats, hooks = {}, []
    for i, layer in enumerate(policy.model.model.encoder.language_model.layers):
        stats[i] = {'input_amax': torch.zeros((), device='cuda'),
                    'fc2_amax': torch.zeros((), device='cuda'), 'calls': 0}

        def record(module, inputs, output, i=i):
            x, ids, _ = inputs
            r, md = ids.numel(), module._metadata
            workspace = _get_cached_workspace(
                backend='w4a16', state_E=md['num_experts'], weight_E=md['num_experts'],
                routed_rows=r, k=md['hidden_size'], n=md['intermediate_size'],
                num_topk=ids.shape[-1], device=x.device, quant_mode='w4a16', activation='gelu_tanh')
            mid = workspace.intermediate_cache2.flatten()[:r*md['intermediate_size']]
            stats[i]['input_amax'] = torch.maximum(stats[i]['input_amax'], x.float().abs().amax())
            stats[i]['fc2_amax'] = torch.maximum(stats[i]['fc2_amax'], mid.float().abs().amax())
            stats[i]['calls'] += 1

        hooks.append(layer.experts.register_forward_hook(record))
    samples = json.loads((args.inputs/'samples.json').read_text())
    assert len(samples) == 32 and len({s['key'] for s in samples}) == 32
    try:
        for s in samples:
            demo = int(s['key'].split('/')[1].split('_')[1])
            assert demo % policy.config['validation_demo_modulus'] != policy.config['validation_demo_remainder']
            image = Image.open(args.inputs/s['file']).convert('RGB')
            assert hashlib.sha256(np.asarray(image).tobytes()).hexdigest() == s['image_sha256']
            policy.predict(image, s['instruction'], 4, s['seed'])
    finally:
        for hook in hooks:
            hook.remove()
    layers = {str(i): {
        'input_amax': float(v['input_amax']), 'fc2_amax': float(v['fc2_amax']), 'calls': v['calls'],
        'input_global_scale': float(v['input_amax'])*1.1/2688,
        'fc2_global_scale': float(v['fc2_amax'])*1.1/2688,
    } for i, v in stats.items()}
    assert all(r['calls'] == 160 and np.isfinite([r['input_amax'], r['fc2_amax']]).all()
               and min(r['input_amax'], r['fc2_amax']) > 0 for r in layers.values())
    report = {'method': '32 evenly spaced TRAIN observations, W4A16 activations, four denoising passes, per-layer amax *1.1 /2688; no rollout or validation inputs',
              'samples': samples, 'layers': layers}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

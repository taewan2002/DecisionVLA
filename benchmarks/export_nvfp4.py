"""Convert an isolated adapter/base copy to private NVFP4 expert assets.

Base weights, exported weights and calibration observations must not be added
to this Git repository. This conversion never updates the original checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from decisionvla.checkpoints import load_adapter
from decisionvla.model import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['w4a16', 'w4a4'], required=True)
    parser.add_argument('--calibration', type=Path, help='Required for W4A4; frozen training-only scale report')
    args = parser.parse_args()
    if (args.mode == 'w4a4') != bool(args.calibration):
        parser.error('Provide --calibration for W4A4 only')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    config = json.loads((args.adapter/'adapter_config.json').read_text())
    model, _, _ = load_model(str(args.model), {**config, 'expert_kernel': 'torch_grouped_mm'}, 'cpu')
    load_adapter(model, args.adapter)
    model.eval()
    model.gradient_checkpointing_disable()
    if args.mode == 'w4a16':
        from resident_nvfp4 import quantize_model
        placement = quantize_model(model)
    else:
        from resident_nvfp4_a4 import quantize_model_a4
        calibration = json.loads(args.calibration.read_text())
        placement = quantize_model_a4(model, calibration)
        placement['calibration_sha256'] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    directory = args.output/'packed-experts'
    directory.mkdir()
    hashes = {}
    for i, layer in enumerate(model.model.encoder.language_model.layers):
        module = layer.experts
        metadata = None
        if args.mode == 'w4a16':
            settings = {**module._metadata, 'params_dtype': str(module._metadata['params_dtype'])}
            metadata = {'settings': json.dumps(settings)}
        else:
            metadata = {'strides': json.dumps({k: list(v.stride()) for k, v in module.named_buffers()})}
        path = directory/f'layer-{i:02d}.safetensors'
        save_file({k: v.contiguous() for k, v in module.named_buffers()}, path, metadata=metadata)
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'status': 'exported_not_quality_validated', 'mode': args.mode,
              'checkpoint_step': config['step'], 'placement': placement,
              'adapter_sha256': json.loads((args.adapter/'sha256.json').read_text()),
              'export_sha256': hashes, 'torch': torch.__version__}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()

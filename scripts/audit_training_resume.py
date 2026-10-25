"""Compare disposable continuous and segmented training runs before a long run."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from decisionvla.checkpoints import adapter_files, file_sha256


def audit(reference, resumed, output):
    maximum = 0.0
    def compare(a, b, floating=False):
        nonlocal maximum
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and a.dtype == b.dtype
            if floating and a.is_floating_point():
                error = float((a - b).abs().max()) if a.numel() else 0.0
                maximum = max(maximum, error)
                torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
            else:
                assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key], floating)
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                compare(left, right, floating)
        else:
            assert a == b, (a, b)
    manifests = {}
    for path in (reference, resumed):
        manifest = json.loads((path / 'sha256.json').read_text())
        for name, expected in manifest.items():
            assert Path(name).name == name and file_sha256(path / name) == expected
        manifests[str(path.name) + ('_reference' if path == reference else '_resumed')] = file_sha256(path / 'sha256.json')
    left, right = adapter_files(reference), adapter_files(resumed)
    assert [p.name for p in left] == [p.name for p in right]
    for a, b in zip(left, right):
        compare(load_file(a), load_file(b), floating=True)
    parameter_error = maximum
    a = torch.load(reference / 'training_state.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed / 'training_state.pt', map_location='cpu', weights_only=False)
    compare({k:v for k,v in a.items() if k != 'optimizer'},
            {k:v for k,v in b.items() if k != 'optimizer'})
    maximum = 0.0
    compare(a['optimizer'], b['optimizer'], floating=True)
    result = {'status':'passed', 'step':a['step'], 'parameter_atol':1e-5,
              'max_parameter_abs_error':parameter_error, 'max_optimizer_tensor_abs_error':maximum,
              'rng_exact':True, 'bitwise_equal':parameter_error == 0 and maximum == 0,
              'checkpoint_manifests':manifests, 'scope':'disposable segment/resume check; no smoke weights reused'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.reference,args.resumed,args.output)))

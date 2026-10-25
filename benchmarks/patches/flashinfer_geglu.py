"""Add GeGLU-tanh to the pinned FlashInfer SM120 W4A16 implementation.

Only an isolated inference image is patched. Formula matches the existing
FlashInfer W4A4 GeGLU path; BF16 activation/multiply rounding is retained.
"""
import hashlib
from pathlib import Path

import flashinfer

assert flashinfer.__version__ == '0.7.2.post1'
root = Path(flashinfer.__file__).parent / 'fused_moe/cute_dsl/blackwell_sm12x'
meta = root / 'moe_w4a16_activations.py'
kernel = root / 'moe_w4a16_kernel.py'
a, b = meta.read_text(), kernel.read_text()
assert 'frozenset({"silu", "relu2"})' in a
assert b.count('                sigmoid_arg = gate\n') == 2
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (meta, kernel)}
a = a.replace('Only silu and relu2 are supported.', 'SiLU, GeGLU-tanh and ReLU2 are supported.')
a = a.replace('frozenset({"silu", "relu2"})', 'frozenset({"silu", "gelu_tanh", "relu2"})')
a = a.replace('frozenset({"silu"})', 'frozenset({"silu", "gelu_tanh"})')
b = b.replace('                sigmoid_arg = gate\n', '''                sigmoid_arg = gate
                if cutlass.const_expr(self.activation == "gelu_tanh"):
                    sigmoid_arg = cutlass.Float32(1.5957691216057308) * (
                        gate + cutlass.Float32(0.044715) * gate * gate * gate
                    )
''')
meta.write_text(a)
kernel.write_text(b)
print({'before': before, 'after': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (meta,kernel)}})

"""Apply a bounded vLLM 0.30.0 image-path fix in a disposable container only.

DiffusionGemma reuses Gemma4's _process_image_input but omits initialization of
_enable_mm_lora. This benchmark uses no LoRA, so the intended branch is False.
No weights, tensor operations or sampling rules are changed.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import vllm

assert vllm.__version__ == "0.30.0", "Review compatibility fix for a new version"
path = Path(vllm.__file__).parent / "model_executor/models/diffusion_gemma.py"
source = path.read_text()
needle = "    hf_to_vllm_mapper = WeightsMapper("
assert source.count(needle) == 1
assert "    _enable_mm_lora = False\n" not in source
patched = source.replace(needle, "    _enable_mm_lora = False\n\n" + needle)
compile(patched, str(path), "exec")
path.write_text(patched)
record = {
    "vllm": vllm.__version__, "file": "vllm/model_executor/models/diffusion_gemma.py",
    "before_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "after_sha256": hashlib.sha256(patched.encode()).hexdigest(),
    "change": "DiffusionGemmaForConditionalGeneration._enable_mm_lora = False",
    "condition": "No LoRA used in benchmark; merged/base weights only",
}
Path("/output/runtime-patch.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])

"""BF16 reference inference. This path needs >50 GiB; NVFP4 serving is a later export stage."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .adapters import load_adapter_state
from .codec import ActionCodec
from .data import prepare_observation
from .model import encode_prefix, load_model, sample_actions
from .train import to_device


class DecisionVLA:
    def __init__(self, base_model, adapter, device="cuda"):
        adapter = Path(adapter)
        self.config = json.loads((adapter / "adapter_config.json").read_text())
        self.model, self.processor, _ = load_model(base_model, self.config, device)
        load_adapter_state(self.model, load_file(adapter / "adapter.safetensors"))
        self.model.eval()
        self.device = device
        self.codec = ActionCodec(self.config["bins"])

    @torch.inference_mode()
    def predict(self, image, instruction, steps=4, seed=42):
        """Return [8, 7] normalized LIBERO controls. Image must already be rotated 180 degrees."""
        inputs = prepare_observation(self.processor, image, instruction, self.config["visual_tokens"])
        cache = encode_prefix(self.model, to_device(inputs, self.device))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        ids = sample_actions(self.model, cache, self.config, steps, generator)
        return self.codec.decode(ids.cpu().numpy().reshape(self.config["horizon"], 7))

"""Historical BF16 pilot (>50 GiB). Experimental NVFP4 runtimes live in benchmarks/."""
import json
from pathlib import Path

import torch

from .checkpoints import load_adapter
from .codec import ActionCodec
from .confidence import summarize
from .data import prepare_observation
from .model import encode_prefix, load_model, sample_actions
from .train import to_device


class DecisionVLA:
    def __init__(self, base_model, adapter, device="cuda"):
        adapter = Path(adapter)
        self.config = json.loads((adapter / "adapter_config.json").read_text())
        self.model, self.processor, _ = load_model(base_model, self.config, device)
        load_adapter(self.model, adapter)
        self.model.eval()
        self.device = device
        self.codec = ActionCodec(self.config["bins"])

    @torch.inference_mode()
    def predict(self, image, instruction, steps=4, seed=42):
        """Return [8, 7] normalized LIBERO controls. Image must already be rotated 180 degrees."""
        return self._predict(image, instruction, steps, seed, diagnostics=False)

    @torch.inference_mode()
    def predict_with_confidence(self, image, instruction, steps=4, seed=42):
        """Return actions and uncalibrated final-step probabilities/diagnostics."""
        return self._predict(image, instruction, steps, seed, diagnostics=True)

    def _predict(self, image, instruction, steps, seed, diagnostics):
        inputs = prepare_observation(self.processor, image, instruction, self.config["visual_tokens"])
        cache = encode_prefix(self.model, to_device(inputs, self.device))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = sample_actions(self.model, cache, self.config, steps, generator,
                                return_probabilities=diagnostics)
        ids, probabilities = result if diagnostics else (result, None)
        actions = self.codec.decode(ids.cpu().numpy().reshape(self.config["horizon"], 7))
        if diagnostics:
            return {"actions": actions,
                    "diagnostics": summarize(probabilities.cpu().numpy().reshape(8, 7, 256))}
        return actions

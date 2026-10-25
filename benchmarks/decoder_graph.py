"""One-shape CUDA graph for the unchanged four-pass action-logit decoder.

Only the decoder is captured. The native sampler and its seeded RNG remain
outside the graph. Prefix KV values are copied for each new observation.
"""

import copy

import torch
from torch.nn import functional as F


class DecoderGraph:
    def __init__(self, model, config, eager):
        self.model, self.config, self.eager = model, config, eager
        self.graph = self.cache = self.shape = None
        self.captures = 0
        self.active_cache = None
        self.copy_prefix = True

    def begin_observation(self, cache):
        """Declare a fresh prefix, immutable during this sampling call only."""
        self.active_cache = cache
        self.copy_prefix = True

    def _forward(self):
        c = self.config
        count = c["horizon"] * c["action_dim"]
        canvas = torch.full(
            (1, c["canvas_length"]), self.model.config.text_config.pad_token_id,
            device="cuda", dtype=torch.long,
        )
        canvas[:, :count] = self.bins + c["action_token_start"]
        hidden = self.model.model.decoder(
            decoder_input_ids=canvas, past_key_values=self.cache,
            decoder_attention_mask=self.masks, decoder_position_ids=self.positions,
        ).last_hidden_state[:, :count]
        weight = self.model.model.decoder.embed_tokens.action_weight()
        logits = F.linear(hidden, weight).float()
        cap = self.model.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
        return logits.masked_fill(~self.allowed[None], -torch.inf)

    def _prepare(self, cache, bins):
        # At most one graph/KV shape is retained, bounding VRAM across tasks.
        self.graph = self.cache = self.output = None
        self.cache = copy.deepcopy(cache)
        self.bins = bins.clone()
        c = self.config
        count = c["horizon"] * c["action_dim"]
        prefix = cache.get_seq_length()
        attention = torch.ones((1, prefix + c["canvas_length"]), dtype=torch.bool, device="cuda")
        attention[:, prefix + count:] = False
        dummy = torch.empty((1, c["canvas_length"], self.model.config.text_config.hidden_size), dtype=torch.bfloat16, device="cuda")
        self.masks = self.model.model.decoder.create_diffusion_decoder_attention_mask(
            config=self.model.config.text_config, inputs_embeds=dummy,
            past_key_values=self.cache, decoder_attention_mask=attention,
        )
        self.positions = torch.arange(prefix, prefix + c["canvas_length"], device="cuda")[None]
        self.allowed = torch.ones((count, c["bins"]), dtype=torch.bool, device="cuda")
        self.allowed[6::7, 1:-1] = False
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._forward()
        self.shape = tuple(tuple(layer.keys.shape) for layer in cache.layers)
        self.captures += 1

    def __call__(self, model, cache, bins, config, *, prefix_attention_mask=None):
        if model is not self.model or config != self.config or prefix_attention_mask is not None:
            raise ValueError("Graph decoder contract changed")
        if bins.shape != (1, config["horizon"] * config["action_dim"]):
            raise ValueError("Only single-observation action chunks are supported")
        shape = tuple(tuple(layer.keys.shape) for layer in cache.layers)
        if self.graph is None or shape != self.shape:
            self._prepare(cache, bins)
        else:
            if cache is not self.active_cache or self.copy_prefix:
                for source, dest in zip(cache.layers, self.cache.layers, strict=True):
                    dest.keys.copy_(source.keys)
                    dest.values.copy_(source.values)
            self.bins.copy_(bins)
        if cache is self.active_cache:
            self.copy_prefix = False
        else:
            self.copy_prefix = True
        self.graph.replay()
        # Sampling consumes this result before the next replay. No random
        # operation or mutable external KV cache is captured.
        return self.output


def bind_policy_graphs(policy, *, prefix=False):
    """Install graphs in one isolated, serial policy process.

    Mark every encoded observation explicitly, including when the prefix graph
    reuses its cache object. Three later denoising passes need no repeated KV
    copies. Calls outside this policy wrapper retain conservative copying.
    """
    import decisionvla.model as model_module
    import decisionvla.policy as policy_module

    if isinstance(model_module.action_logits, DecoderGraph):
        raise TypeError('Install policy graphs only once per isolated process')
    graph = DecoderGraph(policy.model, policy.config, model_module.action_logits)
    eager_encode = policy_module.encode_prefix

    def encode_and_mark(model, inputs):
        cache = eager_encode(model, inputs)
        graph.begin_observation(cache)
        return cache

    policy_module.encode_prefix = encode_and_mark
    model_module.action_logits = graph
    if prefix:
        from prefix_graph import PrefixLanguageGraph
        language_model = policy.model.model.encoder.language_model
        language_model.forward = PrefixLanguageGraph(language_model)
    return graph

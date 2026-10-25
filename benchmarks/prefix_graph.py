"""One-shape CUDA graph for the prefix language model, after eager vision.

The original encoder still computes image features and validates placeholder
counts. Masks, positions and embeddings are copied for every observation.
"""
import torch


def tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(fn, v) for k, v in value.items()}
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise TypeError(f'Unsupported graph argument: {type(value).__name__}')
    return value


def copy_tree(source, target):
    if isinstance(source, torch.Tensor):
        target.copy_(source)
    elif isinstance(source, dict):
        assert source.keys() == target.keys()
        for key in source:
            copy_tree(source[key], target[key])
    else:
        assert source == target


class PrefixLanguageGraph:
    def __init__(self, language_model):
        self.eager = language_model.forward
        self.config = language_model.config
        self.graph = self.signature = self.arguments = self.output = None
        self.captures = 0

    def _fresh_cache(self):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.config)
        # HF initializes this scalar on CPU. Transfer it before capture;
        # the unchanged lazy initialization then performs no host copy.
        for layer in cache.layers:
            if hasattr(layer, '_sliding_window_tensor'):
                layer._sliding_window_tensor = layer._sliding_window_tensor.to('cuda')
        return cache

    def __call__(self, *args, **kwargs):
        if args or kwargs.get('past_key_values') is not None:
            raise ValueError('Only a fresh prefix with keyword inputs is supported')
        if not isinstance(kwargs.get('attention_mask'), dict):
            raise TypeError('Explicit vision/text masks must be built before capture')
        signature = tree_map(lambda t: (t.shape, t.dtype, t.device), kwargs)
        if self.graph is None or signature != self.signature:
            self.graph = self.output = self.arguments = None
            self.arguments = tree_map(lambda t: t.detach().clone(), kwargs)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.eager(**{**self.arguments, 'past_key_values': self._fresh_cache()})
            torch.cuda.current_stream().wait_stream(stream)
            cache = self._fresh_cache()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.eager(**{**self.arguments, 'past_key_values': cache})
            self.signature = signature
            self.captures += 1
        else:
            copy_tree(kwargs, self.arguments)
        self.graph.replay()
        return self.output

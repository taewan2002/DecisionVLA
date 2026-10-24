"""Fixed [-1, 1] LIBERO control bins; no fitted statistics or future information."""
import numpy as np


class ActionCodec:
    def __init__(self, bins=256):
        self.bins = bins

    def encode(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[-1] != 7 or not np.isfinite(actions).all():
            raise ValueError("Expected finite [..., 7] LIBERO actions")
        if np.max(np.abs(actions)) > 1.0001:
            raise ValueError("LIBERO OSC actions must be in [-1, 1]")
        tokens = np.rint((np.clip(actions, -1, 1) + 1) * (self.bins - 1) / 2).astype(np.int64)
        tokens[..., 6] = np.where(actions[..., 6] > 0, self.bins - 1, 0)
        return tokens

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.shape[-1] != 7 or np.any(tokens < 0) or np.any(tokens >= self.bins):
            raise ValueError("Invalid action token IDs")
        actions = tokens.astype(np.float32) * (2 / (self.bins - 1)) - 1
        actions[..., 6] = np.where(tokens[..., 6] >= self.bins / 2, 1, -1)
        return actions
